#!/usr/bin/env python3
"""Writes a canonical text description of ONNX models, using only the public API of the onnx package (onnx.load and the
message fields). tests/OnnxSweep writes the same text from our importer; the two are compared with diff.

    onnx_dump.py sweep <root> <outdir>        every model.onnx under <root>
    onnx_dump.py dump <outdir> <model.onnx>...

Floats are written as bit patterns and tensor data as SHA-256 of its little-endian bytes (string tensors: each element
prefixed with its 4-byte length), so no formatting can differ.
"""
import hashlib
import os
import struct
import sys

import onnx


def bits(f):
    return "%08x" % struct.unpack("<I", struct.pack("<f", f))[0]


def tstr(tp):
    k = tp.WhichOneof("value")
    if k is None:
        return "none"
    if k in ("tensor_type", "sparse_tensor_type"):
        t = getattr(tp, k)
        s = ("tensor(%d)" if k == "tensor_type" else "sparse(%d)") % t.elem_type
        if t.HasField("shape"):
            dims = []
            for d in t.shape.dim:
                dk = d.WhichOneof("value")
                if dk == "dim_value":
                    dims.append("?" if d.dim_value == -1 else str(d.dim_value))
                elif dk == "dim_param":
                    dims.append(d.dim_param)
                else:
                    dims.append("?")
            s += "[" + ",".join(dims) + "]"
        return s
    if k in ("sequence_type", "optional_type"):
        t = getattr(tp, k)
        inner = tstr(t.elem_type) if t.HasField("elem_type") else ""
        inner = "" if inner == "none" else inner
        return ("seq<%s>" if k == "sequence_type" else "opt<%s>") % inner
    if k == "map_type":
        t = tp.map_type
        inner = tstr(t.value_type) if t.HasField("value_type") else ""
        inner = "" if inner == "none" else inner
        return "map<%d,%s>" % (t.key_type, inner)
    return "opaque(%s:%s)" % (tp.opaque_type.domain, tp.opaque_type.name)


def vtype(vi):
    return tstr(vi.type) if vi.HasField("type") else "none"


def pack6(values):
    out = bytearray((len(values) * 6 + 7) // 8)
    for i, v in enumerate(values):
        v &= 0x3F
        bit = i * 6
        out[bit >> 3] |= (v << (bit & 7)) & 0xFF
        if (bit & 7) > 2:
            out[(bit >> 3) + 1] |= v >> (8 - (bit & 7))
    return bytes(out)


def tensor_bytes(t):
    dt = t.data_type
    if dt == 8:
        return b"".join(struct.pack("<I", len(s)) + s for s in t.string_data)
    if t.HasField("raw_data"):
        return t.raw_data
    if dt in (1, 14):
        return struct.pack("<%df" % len(t.float_data), *t.float_data)
    if dt in (11, 15):
        return struct.pack("<%dd" % len(t.double_data), *t.double_data)
    if dt == 7:
        return struct.pack("<%dq" % len(t.int64_data), *t.int64_data)
    if dt == 13:
        return struct.pack("<%dQ" % len(t.uint64_data), *t.uint64_data)
    if dt == 12:
        return struct.pack("<%dI" % len(t.uint64_data), *[v & 0xFFFFFFFF for v in t.uint64_data])
    v = list(t.int32_data)
    if dt == 6:
        return struct.pack("<%di" % len(v), *v)
    if dt in (4, 5, 10, 16):
        return struct.pack("<%dH" % len(v), *[x & 0xFFFF for x in v])
    if dt in (27, 28):
        return pack6(v)
    if dt == 9:
        return bytes(1 if x else 0 for x in v)
    return bytes(x & 0xFF for x in v)


def tensor(t):
    if t.data_type == 8:
        h = hashlib.sha256(tensor_bytes(t)).hexdigest()
    else:
        h = hashlib.sha256(tensor_bytes(t)).hexdigest()
    return "%s dtype=%d dims=[%s] sha256=%s" % (t.name, t.data_type, ",".join(str(d) for d in t.dims), h)


def sparse(s):
    return "dims=[%s] values=%s indices=%s" % (",".join(str(d) for d in s.dims), tensor(s.values), tensor(s.indices))


def attr_type(a):
    if a.type:
        return a.type
    for name, code in (("f", 1), ("i", 2), ("s", 3), ("t", 4), ("g", 5), ("sparse_tensor", 11), ("tp", 13)):
        if a.HasField(name):
            return code
    for name, code in (("floats", 6), ("ints", 7), ("strings", 8), ("tensors", 9), ("graphs", 10), ("sparse_tensors", 12), ("type_protos", 14)):
        if len(getattr(a, name)):
            return code
    return 0


def dump_attr(out, a, ind):
    p = " " * ind
    ty = attr_type(a)
    head = "%sattr %s type=%d ref=%s" % (p, a.name, ty, a.ref_attr_name)
    if a.ref_attr_name:
        out.append(head)
        return
    if ty == 1:
        out.append("%s %s" % (head, bits(a.f)))
    elif ty == 2:
        out.append("%s %d" % (head, a.i))
    elif ty == 3:
        out.append("%s %s" % (head, a.s.hex().upper()))
    elif ty == 4:
        out.append("%s %s" % (head, tensor(a.t)))
    elif ty == 11:
        out.append("%s %s" % (head, sparse(a.sparse_tensor)))
    elif ty == 13:
        out.append("%s %s" % (head, tstr(a.tp)))
    elif ty == 6:
        out.append("%s [%s]" % (head, ",".join(bits(f) for f in a.floats)))
    elif ty == 7:
        out.append("%s [%s]" % (head, ",".join(str(i) for i in a.ints)))
    elif ty == 8:
        out.append("%s [%s]" % (head, ",".join(s.hex().upper() for s in a.strings)))
    elif ty == 9:
        out.append(head)
        for t in a.tensors:
            out.append("%s %s" % (p, tensor(t)))
    elif ty == 12:
        out.append(head)
        for t in a.sparse_tensors:
            out.append("%s %s" % (p, sparse(t)))
    elif ty == 14:
        out.append("%s [%s]" % (head, ";".join(tstr(t) for t in a.type_protos)))
    elif ty == 5:
        out.append(head)
        dump_graph(out, a.g, ind + 1)
    elif ty == 10:
        out.append(head)
        for g in a.graphs:
            dump_graph(out, g, ind + 1)
    else:
        out.append(head)


def dump_node(out, n, ind):
    p = " " * ind
    out.append("%snode op=%s domain=%s name=%s overload=%s in=[%s] out=[%s]" % (
        p, n.op_type, n.domain, n.name, n.overload, ",".join(n.input), ",".join(n.output)))
    for a in n.attribute:
        dump_attr(out, a, ind + 1)


def dump_graph(out, g, ind):
    p = " " * ind
    out.append("%sgraph name=%s" % (p, g.name))
    for v in g.input:
        out.append("%s input %s %s" % (p, v.name, vtype(v)))
    for v in g.output:
        out.append("%s output %s %s" % (p, v.name, vtype(v)))
    defined = set(v.name for v in g.input)
    defined.update(t.name for t in g.initializer)
    defined.update(s.values.name for s in g.sparse_initializer)
    for n in g.node:
        defined.update(o for o in n.output if o)
    skip = set(v.name for v in g.input) | set(v.name for v in g.output)
    seen = {}
    for vi in g.value_info:
        if vi.name in defined and vi.name not in skip and vi.name not in seen and vi.HasField("type") and vi.type.WhichOneof("value"):
            seen[vi.name] = tstr(vi.type)
    for name in sorted(seen, key=lambda s: s.encode("utf-8")):
        out.append("%s valueinfo %s %s" % (p, name, seen[name]))
    for t in g.initializer:
        out.append("%s init %s" % (p, tensor(t)))
    for s in g.sparse_initializer:
        out.append("%s sparse %s" % (p, sparse(s)))
    for n in g.node:
        dump_node(out, n, ind + 1)


def dump_model(path):
    m = onnx.load(path, load_external_data=True)
    out = ["model ir=%d producer=%s|%s domain=%s version=%d" % (m.ir_version, m.producer_name, m.producer_version, m.domain, m.model_version)]
    for o in m.opset_import:
        out.append("opset %s=%d" % (o.domain, o.version))
    for kv in m.metadata_props:
        out.append("meta %s=%s" % (kv.key, kv.value))
    dump_graph(out, m.graph, 0)
    for f in m.functions:
        out.append("function %s|%s|%s in=[%s] out=[%s] attrs=[%s] nodes=%d" % (
            f.domain, f.name, f.overload, ",".join(f.input), ",".join(f.output), ",".join(f.attribute), len(f.node)))
        for o in f.opset_import:
            out.append("  fopset %s=%d" % (o.domain, o.version))
        for a in f.attribute_proto:
            dump_attr(out, a, 2)
        for n in f.node:
            dump_node(out, n, 2)
    return "\n".join(out) + "\n"


def write(target, path):
    try:
        text = dump_model(path)
    except Exception as e:  # a model the package cannot read is recorded as an error, like ours
        print("%s: %s" % (path, str(e)[:200]))
        text = "ERROR\n"
    with open(target, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def main(argv):
    if len(argv) >= 4 and argv[1] == "sweep":
        root, outdir = argv[2], argv[3]
        os.makedirs(outdir, exist_ok=True)
        files = []
        for d, _, names in os.walk(root):
            files += [os.path.join(d, n) for n in names if n == "model.onnx"]
        for f in sorted(files):
            key = os.path.relpath(f, root).replace(os.sep, "_") + ".txt"
            write(os.path.join(outdir, key), f)
        return 0
    if len(argv) >= 4 and argv[1] == "dump":
        os.makedirs(argv[2], exist_ok=True)
        for f in argv[3:]:
            write(os.path.join(argv[2], os.path.basename(f) + ".txt"), f)
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
