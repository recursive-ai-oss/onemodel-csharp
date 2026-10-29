#!/usr/bin/env python3
"""Generates randomized operator cases for the convolution / pooling / normalisation / resize family and records ONNX Runtime's
outputs as the expected values.

Usage: ops_conv_oracle.py OUTDIR [--seed N] [--count K]

Each case is a directory in the layout of onnx/backend/test/data/node (model.onnx, test_data_set_0/input_*.pb, output_*.pb), so the
C# tests read generated cases and the frozen backend cases with the same code. The generator is seeded; the same seed gives the same
cases. ONNX Runtime is used as a black box through its public Python API (version pinned in requirements-oracles.txt).
"""
import argparse
import os
import shutil

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

NAN = np.float32("nan")


class Gen:
    def __init__(self, out, seed):
        self.out = out
        self.rng = np.random.RandomState(seed)
        self.count = 0
        self.skipped = 0
        self.diverged = 0
        self.fails = []

    def r(self, *shape, special=False):
        """Random normal data; special=True adds NaN, infinities and -0, special="nonan" the same without NaN."""
        a = self.rng.randn(*shape).astype(np.float32)
        if special and a.size > 8:
            for v in ((np.inf, -np.inf, -0.0) if special == "nonan" else (NAN, np.inf, -np.inf, -0.0)):
                a.flat[self.rng.randint(a.size)] = v
        return a

    def case(self, name, op_type, inputs, n_out, opset, expect_shape=None, **attrs):
        """inputs: list of (array or None, is_initializer). None = omitted optional input."""
        feeds = {}
        names, init, graph_in = [], [], []
        for k, item in enumerate(inputs):
            if item is None:
                names.append("")
                continue
            arr, as_init = item
            nm = f"in{k}"
            names.append(nm)
            if as_init:
                init.append(numpy_helper.from_array(arr, nm))
            else:
                graph_in.append(helper.make_tensor_value_info(nm, helper.np_dtype_to_tensor_dtype(arr.dtype), list(arr.shape)))
                feeds[nm] = arr
        outs = [f"out{k}" for k in range(n_out)]
        node = helper.make_node(op_type, names, outs, **attrs)
        graph = helper.make_graph([node], name, graph_in, [helper.make_empty_tensor_value_info(o) for o in outs], initializer=init)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
        model.ir_version = 9
        try:
            sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
            results = sess.run(None, feeds)
        except Exception as e:  # ORT rejects the combination: not a case
            self.skipped += 1
            return
        if expect_shape is not None and list(results[0].shape) != list(expect_shape):
            # ONNX Runtime and the operator text disagree about the output size: not a case (counted and reported).
            self.diverged += 1
            return
        # Record the output types and shapes ORT found, so the model is a complete one.
        graph = helper.make_graph([node], name, graph_in, [
            helper.make_tensor_value_info(o, helper.np_dtype_to_tensor_dtype(r.dtype), list(r.shape)) for o, r in zip(outs, results)],
            initializer=init)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
        model.ir_version = 9
        d = os.path.join(self.out, f"{name}_{self.count:04d}")
        os.makedirs(os.path.join(d, "test_data_set_0"))
        onnx.save(model, os.path.join(d, "model.onnx"))
        gi = 0
        for nm in [g.name for g in graph_in]:
            with open(os.path.join(d, "test_data_set_0", f"input_{gi}.pb"), "wb") as f:
                f.write(numpy_helper.from_array(feeds[nm]).SerializeToString())
            gi += 1
        for k, res in enumerate(results):
            with open(os.path.join(d, "test_data_set_0", f"output_{k}.pb"), "wb") as f:
                f.write(numpy_helper.from_array(res).SerializeToString())
        self.count += 1


def spec_pool_shape(x, k, s, dil, pads, auto, ceil):
    """Output shape from the operator text: ceil(in / stride) for SAME_*, else floor/ceil((in + pads - eff) / stride) + 1."""
    out = list(x.shape[:2])
    for i, n in enumerate(x.shape[2:]):
        eff = dil[i] * (k[i] - 1) + 1
        if auto in ("SAME_UPPER", "SAME_LOWER"):
            out.append(-(-n // s[i]))
            continue
        pb, pe = (0, 0) if auto == "VALID" else (pads[i], pads[i + len(k)])
        span = n + pb + pe - eff
        if span < 0:
            return [-1]
        e = (-(-span // s[i]) if ceil else span // s[i]) + 1
        if ceil and (e - 1) * s[i] >= n + pb:
            e -= 1
        out.append(e)
    return out


def conv_cases(g, count):
    rng = g.rng
    # Shapes as the reference models use them, small.
    fixed = [
        # (N, Cin, spatial, Cout, kernel, stride, pads, dil, group)
        (1, 3, (32, 32), 8, (7, 7), (2, 2), (3, 3, 3, 3), (1, 1), 1),   # ResNet stem
        (2, 8, (9, 9), 16, (1, 1), (1, 1), (0, 0, 0, 0), (1, 1), 1),    # pointwise
        (1, 8, (16, 16), 8, (3, 3), (2, 2), (1, 1, 1, 1), (1, 1), 8),   # MobileNet depthwise stride 2
        (1, 6, (12, 12), 6, (5, 5), (1, 1), (2, 2, 2, 2), (1, 1), 6),   # depthwise 5x5
        (1, 4, (20,), 6, (3,), (1,), (1, 1), (1,), 1),                  # Whisper conv1d
        (1, 6, (20,), 6, (3,), (2,), (1, 1), (1,), 1),                  # Whisper conv1d stride 2
        (1, 1, (8, 24), 4, (3, 3), (1, 1), (1, 1, 1, 1), (1, 1), 1),    # CRNN first layer, one channel
        (1, 2, (7, 5), 3, (3, 3), (1, 1), (0, 0, 0, 0), (1, 1), 1),
    ]
    for (n, cin, sp, cout, k, s, p, dil, grp) in fixed:
        for bias in (True, False):
            for init in (True, False):
                x = g.r(n, cin, *sp)
                w = g.r(cout, cin // grp, *k) * 0.3
                ins = [(x, False), (w, init)] + ([(g.r(cout), init)] if bias else [])
                g.case("conv", "Conv", ins, 1, 17, kernel_shape=list(k), strides=list(s), pads=list(p), dilations=list(dil), group=grp)
    for _ in range(count):
        r1d = rng.rand() < 0.3
        nsp = 1 if r1d else 2
        grp = int(rng.choice([1, 1, 2, 3]))
        cin = grp * int(rng.randint(1, 4))
        cout = grp * int(rng.randint(1, 4))
        if rng.rand() < 0.15:
            grp, cin = cin, cin
            cout = cin * int(rng.randint(1, 3))
        n = int(rng.choice([1, 1, 2]))
        k = tuple(int(v) for v in rng.randint(1, 5, size=nsp))
        s = tuple(int(v) for v in rng.randint(1, 4, size=nsp))
        dil = tuple(int(v) for v in rng.randint(1, 3, size=nsp))
        sp = tuple(int(v) for v in rng.randint(1, 12, size=nsp))
        attrs = dict(kernel_shape=list(k), strides=list(s), dilations=list(dil), group=grp)
        mode = rng.choice(["pads", "pads", "SAME_UPPER", "SAME_LOWER", "VALID"])
        if mode == "pads":
            attrs["pads"] = [int(v) for v in rng.randint(0, 3, size=2 * nsp)]
        else:
            attrs["auto_pad"] = str(mode)
        x = g.r(n, cin, *sp, special=rng.rand() < 0.1)
        w = g.r(cout, cin // grp, *k) * 0.3
        ins = [(x, False), (w, bool(rng.rand() < 0.5))]
        if rng.rand() < 0.6:
            ins.append((g.r(cout), bool(rng.rand() < 0.5)))
        g.case("conv_rand", "Conv", ins, 1, 17, **attrs)
    g.case("conv_empty_batch", "Conv", [(np.zeros((0, 2, 5, 5), np.float32), False), (g.r(3, 2, 3, 3), True)], 1, 17, kernel_shape=[3, 3])


def convt_cases(g, count):
    rng = g.rng
    for _ in range(count):
        nsp = 1 if rng.rand() < 0.3 else 2
        grp = int(rng.choice([1, 1, 2]))
        cin = grp * int(rng.randint(1, 4))
        cout_g = int(rng.randint(1, 4))
        n = int(rng.choice([1, 2]))
        k = tuple(int(v) for v in rng.randint(1, 5, size=nsp))
        s = tuple(int(v) for v in rng.randint(1, 4, size=nsp))
        dil = tuple(int(v) for v in rng.randint(1, 3, size=nsp))
        sp = tuple(int(v) for v in rng.randint(1, 7, size=nsp))
        attrs = dict(kernel_shape=list(k), strides=list(s), dilations=list(dil), group=grp)
        mode = rng.choice(["pads", "pads", "opad", "SAME_UPPER", "SAME_LOWER", "VALID", "oshape"])
        if mode == "pads":
            attrs["pads"] = [int(v) for v in rng.randint(0, 2, size=2 * nsp)]
        elif mode == "opad":
            attrs["output_padding"] = [int(rng.randint(0, st)) for st in s]
        elif mode == "oshape":
            attrs["output_shape"] = [int(((sp[i] - 1) * s[i]) + (dil[i] * (k[i] - 1) + 1) - rng.randint(0, 3)) for i in range(nsp)]
        else:
            attrs["auto_pad"] = str(mode)
        x = g.r(n, cin, *sp)
        w = g.r(cin, cout_g, *k) * 0.3
        ins = [(x, False), (w, bool(rng.rand() < 0.5))]
        if rng.rand() < 0.5:
            ins.append((g.r(cout_g * grp), True))
        g.case("convtranspose", "ConvTranspose", ins, 1, 17, **attrs)
    # DBNet-style 2x2 stride 2 upsampler
    g.case("convtranspose_dbnet", "ConvTranspose", [(g.r(1, 4, 5, 6), False), (g.r(4, 4, 2, 2), True), (g.r(4), True)], 1, 17,
           kernel_shape=[2, 2], strides=[2, 2])


def pool_cases(g, count):
    rng = g.rng
    g.case("maxpool_resnet", "MaxPool", [(g.r(1, 3, 17, 17), False)], 1, 12, kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1])
    for _ in range(count):
        nsp = int(rng.choice([1, 2, 2, 3]))
        k = tuple(int(v) for v in rng.randint(1, 5, size=nsp))
        s = tuple(int(v) for v in rng.randint(1, 4, size=nsp))
        dil = tuple(int(v) for v in rng.randint(1, 3, size=nsp))
        sp = tuple(int(v) for v in rng.randint(1, 10, size=nsp))
        pads = [int(rng.randint(0, kk)) for kk in k] + [int(rng.randint(0, kk)) for kk in k]
        attrs = dict(kernel_shape=list(k), strides=list(s))
        mode = rng.choice(["pads", "pads", "SAME_UPPER", "SAME_LOWER", "VALID"])
        if mode == "pads":
            attrs["pads"] = pads
        else:
            attrs["auto_pad"] = str(mode)
        if rng.rand() < 0.4:
            attrs["ceil_mode"] = 1
        x = g.r(int(rng.choice([1, 2])), int(rng.randint(1, 4)), *sp, special="nonan" if rng.rand() < 0.25 else False)
        # MaxPool, with and without Indices, either storage order (2-D only for column-major).
        mattrs = dict(attrs)
        if rng.rand() < 0.5 and "auto_pad" not in attrs:  # SAME_* with dilation: ONNX Runtime pads by the undilated kernel (docs/ops-conv.md)
            mattrs["dilations"] = list(dil)
        if nsp == 2 and rng.rand() < 0.3:
            mattrs["storage_order"] = 1
        z = [0] * (2 * nsp)
        g.case("maxpool", "MaxPool", [(x, False)], int(rng.choice([1, 2])), 12, **mattrs, expect_shape=spec_pool_shape(
            x, k, s, mattrs.get("dilations", [1] * nsp), mattrs.get("pads", z), mattrs.get("auto_pad", "NOTSET"), mattrs.get("ceil_mode", 0)))
        aattrs = dict(attrs)
        if rng.rand() < 0.5:
            aattrs["count_include_pad"] = 1
        opset = 11
        if rng.rand() < 0.5 and "auto_pad" not in attrs:
            aattrs["dilations"] = list(dil)
            opset = 19
        g.case("avgpool", "AveragePool", [(x, False)], 1, opset, **aattrs, expect_shape=spec_pool_shape(
            x, k, s, aattrs.get("dilations", [1] * nsp), aattrs.get("pads", z), aattrs.get("auto_pad", "NOTSET"), aattrs.get("ceil_mode", 0)))
    for shape in [(1, 3, 7, 7), (2, 4, 5), (1, 2, 3, 4, 5), (3, 5, 1, 1), (0, 3, 4, 4)]:
        x = g.r(*shape, special=len(shape) > 3 and shape[0] > 0)
        g.case("globalavg", "GlobalAveragePool", [(x, False)], 1, 17)
        xm = g.r(*shape, special="nonan" if len(shape) > 3 and shape[0] > 0 else False)  # ORT's NaN handling differs by CPU
        g.case("globalmax", "GlobalMaxPool", [(xm, False)], 1, 17)
    g.case("maxpool_uint8", "MaxPool", [(rng.randint(0, 255, size=(1, 2, 6, 6)).astype(np.uint8), False)], 1, 12, kernel_shape=[2, 2], strides=[2, 2])
    g.case("maxpool_int8", "MaxPool", [(rng.randint(-128, 127, size=(1, 2, 6, 6)).astype(np.int8), False)], 1, 12, kernel_shape=[3, 3], pads=[1, 1, 1, 1])


def bn_cases(g, count):
    rng = g.rng
    for i in range(count):
        shape = [int(v) for v in rng.randint(1, 6, size=int(rng.randint(2, 5)))]
        c = shape[1]
        ins = [(g.r(*shape, special=i % 5 == 0), False), (g.r(c), bool(rng.rand() < 0.5)), (g.r(c), True),
               (g.r(c), True), (np.abs(g.r(c)) + 0.1, True)]
        g.case("batchnorm", "BatchNormalization", ins, 1, int(rng.choice([9, 14, 15])), epsilon=float(rng.choice([1e-5, 1e-3, 0.1])))
    g.case("batchnorm_real", "BatchNormalization", [(g.r(2, 8, 5, 5), False), (g.r(8) + 1, True), (g.r(8), True), (g.r(8), True), (np.abs(g.r(8)) + 0.5, True)], 1, 17)


def resize_cases(g, count):
    rng = g.rng
    modes = ["nearest", "linear", "cubic"]
    coords = ["half_pixel", "half_pixel_symmetric", "pytorch_half_pixel", "align_corners", "asymmetric"]
    nearests = ["round_prefer_floor", "round_prefer_ceil", "floor", "ceil"]
    # Model-shaped: nearest x2 asymmetric/floor (torch's default), bilinear half_pixel and align_corners.
    x = g.r(1, 3, 5, 6)
    sc = np.array([1, 1, 2, 2], np.float32)
    g.case("resize_torch_nearest", "Resize", [(x, False), (np.array([], np.float32), True), (sc, True)], 1, 17, mode="nearest",
           coordinate_transformation_mode="asymmetric", nearest_mode="floor")
    g.case("resize_torch_bilinear", "Resize", [(x, False), (np.array([], np.float32), True), (sc, True)], 1, 17, mode="linear",
           coordinate_transformation_mode="half_pixel")
    g.case("resize_torch_bilinear_ac", "Resize", [(x, False), (np.array([], np.float32), True), (sc, True)], 1, 17, mode="linear",
           coordinate_transformation_mode="align_corners")
    for i in range(count):
        rank = int(rng.choice([1, 2, 3, 4, 4]))
        shape = [int(v) for v in rng.randint(1, 7, size=rank)]
        mode = str(rng.choice(modes))
        x = g.r(*shape, special=(rng.rand() < 0.1 and mode == "nearest"))  # ORT interpolates infinities and NaN in its own order
        attrs = dict(mode=mode, coordinate_transformation_mode=str(rng.choice(coords + ["tf_crop_and_resize"] if rng.rand() < 0.15 else coords)))
        if mode == "nearest":
            attrs["nearest_mode"] = str(rng.choice(nearests))
        if mode == "cubic":
            attrs["cubic_coeff_a"] = float(rng.choice([-0.75, -0.5]))
            attrs["exclude_outside"] = int(rng.rand() < 0.3)
        use_sizes = rng.rand() < 0.4
        if attrs["coordinate_transformation_mode"] == "tf_crop_and_resize":
            roi = np.concatenate([rng.rand(rank) * 0.4, 0.6 + rng.rand(rank) * 0.4]).astype(np.float32)
            attrs["extrapolation_value"] = float(rng.choice([0.0, 1.5]))
        else:
            roi = np.array([], np.float32)
        opset = int(rng.choice([13, 17, 19]))
        axes = None
        if opset >= 18 and rng.rand() < 0.3 and rank > 1:
            axes = sorted(int(a) for a in rng.choice(rank, size=int(rng.randint(1, rank)), replace=False))
            attrs["axes"] = axes
        n = len(axes) if axes else rank
        as_init = bool(rng.rand() < 0.7)
        if use_sizes:
            sizes = rng.randint(1, 10, size=n).astype(np.int64)
            if opset >= 18 and rng.rand() < 0.3:
                attrs["keep_aspect_ratio_policy"] = str(rng.choice(["not_larger", "not_smaller", "stretch"]))
            ins = [(x, False), (roi, True), None, (sizes, as_init)]
        else:
            scales = rng.choice([0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0], size=n).astype(np.float32)
            ins = [(x, False), (roi, True), (scales, as_init)]
        g.case("resize", "Resize", ins, 1, opset, **attrs)
    # Older opsets and Upsample.
    for mode in ("nearest", "linear"):
        x = g.r(1, 2, 4, 5)
        sc = np.array([1, 1, 2, 1.5], np.float32)
        g.case("resize10", "Resize", [(x, False), (sc, True)], 1, 10, mode=mode)
        g.case("upsample9", "Upsample", [(x, False), (sc, True)], 1, 9, mode=mode)
        g.case("upsample7", "Upsample", [(x, False)], 1, 7, mode=mode, scales=[1.0, 1.0, 2.0, 3.0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--count", type=int, default=60, help="random cases per family")
    a = ap.parse_args()
    if os.path.exists(a.out):
        shutil.rmtree(a.out)
    os.makedirs(a.out)
    g = Gen(a.out, a.seed)
    conv_cases(g, a.count)
    convt_cases(g, a.count)
    pool_cases(g, a.count)
    bn_cases(g, a.count // 3)
    resize_cases(g, a.count * 2)
    print(f"wrote {g.count} cases to {a.out}; ONNX Runtime refused {g.skipped} combinations and disagreed with the operator text on the output size of {g.diverged}; onnxruntime {ort.__version__}")


if __name__ == "__main__":
    main()
