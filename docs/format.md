# The OneModel model format

A model file with the graph and its weights in one place, laid out so the weights can be used straight from a memory map. This
text is normative: a reader or writer in another language can be built from it alone. **Format version 1. Unstable until 1.0:**
version 1 files may stop loading, or change meaning, before the library reaches 1.0.

The format stores an ONNX graph (the same model that the ONNX importer produces, see [onnx-import.md](onnx-import.md)) without
protobuf: a fixed header, a JSON manifest and the raw tensor bytes. It carries no optimisation results. What is saved is the
imported graph before any pass, so a saved file optimises and runs exactly as the ONNX file it came from, and bit-identically
(`graph-passes.md` says what passes do at load).

## 1. File layout

All integers are little endian. Offsets are from the start of the file.

```
offset 0                 header                64 bytes
offset 64                manifest              manifest_length bytes, UTF-8 JSON
offset 64+manifest_length   zero padding        up to the next multiple of 64
offset data_offset       data region           data_length bytes
end of file                                    data_offset + data_length == file length
```

### 1.1 Header (64 bytes)

| Offset | Size | Field | Value |
|---|---|---|---|
| 0 | 8 | magic | the ASCII bytes `ONEMODEL` |
| 8 | 4 | version | u32, `1` |
| 12 | 4 | flags | u32, `0`. A reader rejects any other value |
| 16 | 8 | manifest_offset | u64, `64` |
| 24 | 8 | manifest_length | u64, bytes of manifest |
| 32 | 8 | data_offset | u64, a multiple of 64, at least `64 + manifest_length`, at most the file length |
| 40 | 8 | data_length | u64, `file length - data_offset` |
| 48 | 8 | file_length | u64, the exact length of the file |
| 56 | 8 | reserved | zero. A reader rejects anything else |

All u64 fields must be at most 2^63 − 1. `file_length` lets a reader notice a truncated or extended file before it trusts any other
field. A reader checks every field above against the real file length before it reads the manifest or allocates from any value.

### 1.2 Manifest

UTF-8 JSON (RFC 8259) without a byte-order mark, one object. Reading it never needs more than a streaming JSON parser. Readers
limit its size (the C# reader: 512 MiB) and its nesting depth (1024 JSON levels).

**Integers** in the manifest are JSON numbers written without a fraction or exponent and fit in a signed 64-bit integer. A reader
must parse them as integers: a JSON parser that turns every number into a double loses bits above 2^53 (offsets and lengths stay
below that in any real model, but `int64` attributes may not).

**Floats** are never written as decimal numbers. A 32-bit float attribute is the unsigned integer of its IEEE-754 bit pattern
(`1.0` is `1065353216`), so NaN payloads, infinities and signed zeros survive.

**Bytes** (text attributes, string tensor elements) are base64 (RFC 4648, standard alphabet, padded) in a JSON string.

**Absent fields.** A field marked *optional* defaults to `""` (strings), `0` (integers) or empty (arrays). Unknown fields are
ignored by readers, so later versions can add fields and keep the version number only if old readers still see the same model.

### 1.3 Data region

The raw bytes of tensors, each starting at a multiple of 64 from the start of the file (`data_offset` is a multiple of 64, so
tensor offsets are multiples of 64 within the region too), with zero bytes between. Tensors are stored in the layout of ONNX's
`TensorProto.raw_data` for their element type, little endian: row-major, no stride, no header. Several tensors may point at the
same bytes. The writer in this library gives each distinct tensor its own bytes and places them in this order: graph
initializers, then each sparse initializer's values and indices, then, for each node in order, the tensors in its attributes
(descending into subgraphs in place), then the same for each function's nodes and attribute defaults. A reader does not depend on
the order.

## 2. Manifest schema

### 2.1 Model (the top-level object)

| Field | JSON | Notes |
|---|---|---|
| `format` | string | required, `"onemodel"` |
| `version` | integer | required, equal to the header's version |
| `ir_version` | integer | required, at least 3 (ONNX IR version) |
| `producer_name`, `producer_version`, `domain`, `doc` | string | optional |
| `model_version` | integer | optional |
| `opsets` | array of `{domain: string, version: integer}` | operator set imports. Domain `""` and `"ai.onnx"` are the default domain |
| `metadata` | array of `{key: string, value: string}` | optional, order kept |
| `graph` | Graph | required |
| `functions` | array of Function | optional |

### 2.2 Graph

| Field | JSON | Notes |
|---|---|---|
| `name`, `doc` | string | optional |
| `inputs` | array of TypedName | graph inputs in order. An input that also has an initializer is listed here and in `initializers` |
| `outputs` | array of TypedName | graph outputs in order |
| `value_info` | array of TypedName | declared types of other values |
| `initializers` | array of Tensor | |
| `sparse_initializers` | array of SparseTensor | |
| `nodes` | array of Node | in execution order when the writer sorted them; a reader must not rely on it (section 3) |

**TypedName**: `{name: string (required), type: Type (optional)}`.

### 2.3 Node

`{name, op (required), domain, overload, doc: string, inputs: [string], outputs: [string], attributes: [Attribute]}`.
An empty string in `inputs` or `outputs` is an omitted optional operand, as in ONNX. A node's operator set version is the
version the model's `opsets` give for the node's domain.

### 2.4 Attribute

`{name (required), type (required), doc, ref, value}`.

`type` is ONNX's `AttributeProto.AttributeType` code, and `value` depends on it:

| type | code | `value` |
|---|---|---|
| FLOAT | 1 | unsigned integer, the float's bits |
| INT | 2 | integer |
| STRING | 3 | base64 string |
| TENSOR | 4 | Tensor |
| GRAPH | 5 | Graph |
| FLOATS | 6 | array of unsigned integers (bits) |
| INTS | 7 | array of integers |
| STRINGS | 8 | array of base64 strings |
| TENSORS | 9 | array of Tensor |
| GRAPHS | 10 | array of Graph |
| SPARSE_TENSOR | 11 | SparseTensor |
| SPARSE_TENSORS | 12 | array of SparseTensor |
| TYPE_PROTO | 13 | Type |
| TYPE_PROTOS | 14 | array of Type |

If `ref` is a non-empty string the attribute is a reference to an attribute of the enclosing function (ONNX
`ref_attr_name`) and has no `value`; `type` is then the type of the referenced attribute. Otherwise `type` 0 is an error and
`value` is required.

### 2.5 Tensor

`{name, dtype (required), dims (required), doc, data | strings}`.

* `dtype` is ONNX's `TensorProto.DataType` code. Every code except 0 (undefined) and the string type 8 has a fixed size in bits:

  | codes | bits per element |
  |---|---|
  | 1 float, 6 int32, 12 uint32 | 32 |
  | 2 uint8, 3 int8, 9 bool, 17–20 float8 types, 24 float8e8m0 | 8 |
  | 4 uint16, 5 int16, 10 float16, 16 bfloat16 | 16 |
  | 7 int64, 11 double, 13 uint64, 14 complex64 | 64 |
  | 15 complex128 | 128 |
  | 21 uint4, 22 int4, 23 float4e2m1 | 4 |
  | 25 uint2, 26 int2 | 2 |
  | 27, 28 | 6 |

  Any other code is rejected as unsupported.
* `dims` is an array of non-negative integers (empty for a scalar). The element count is their product, which must stay below 2^60.
* `data` (every dtype except 8) is `{offset: integer, length: integer}`: the bytes are `length` bytes at `data_offset + offset`.
  `length` must equal `ceil(count × bits / 8)`, `offset` must be a multiple of 64 when `length > 0`, and
  `offset + length` must not exceed `data_length`. Elements narrower than a byte are packed as in ONNX `raw_data`. This library
  supports tensors of up to 2^31 − 1 bytes and rejects larger ones.
* `strings` (dtype 8) is an array of base64 strings, one per element, `count` of them.

### 2.6 SparseTensor

`{values: Tensor, indices: Tensor, dims: [integer]}` as ONNX `SparseTensorProto`. The `values` tensor's `name` is the sparse
initializer's name.

### 2.7 Type

An object with a `kind`:

| kind | other fields |
|---|---|
| `"tensor"`, `"sparse_tensor"` | `elem`: integer element type code (required). `shape`: optional array; present (even empty) means the rank is known. Each entry is a non-negative integer (a fixed size), a string (a named dimension), or `null` (unknown) |
| `"sequence"`, `"optional"` | `inner`: Type |
| `"map"` | `key`: integer element type code; `inner`: Type (the value type) |
| `"opaque"` | `domain`, `name`: strings |

Any other kind is rejected as unsupported.

### 2.8 Function

`{name, domain, overload, doc, inputs: [string], outputs: [string], attribute_names: [string], attribute_defaults: [Attribute], nodes: [Node], opsets: [{domain, version}]}`.
Functions are carried, not expanded.

## 3. Reading rules

After the manifest is parsed, names are resolved exactly as for an ONNX file:

1. Within one graph every name defines at most one value. Graph inputs, initializers, sparse initializers (by their values'
   name) and non-empty node outputs define values; a second definition of a name is an error. An initializer whose name matches
   a graph input attaches to that input instead.
2. A node input or graph output names a value defined in the same graph or in an enclosing graph (subgraphs read the values of
   the graphs around them). An undefined name is an error.
3. Nodes are put in dependency order (stable: nodes already in order keep their positions). A cycle is an error.
4. A node whose domain has no entry in `opsets` is an error.
5. `value_info` and the `type` of inputs and outputs set the declared type of the value they name.

### 3.1 Limits and checks

A reader checks, before it allocates or slices from any value in the file:

* the header against the real file length (section 1.1);
* the manifest size;
* every tensor range against the data region, in 64-bit arithmetic, and the byte length against dtype and dims;
* element counts: no negative dimension, no product at 2^60 or more;
* nesting: graphs and types are counted together; more than 64 open at once is rejected (a graph inside a node's attribute
  counts one level, a type inside a graph or a type one more).

Every failure is reported as one of four kinds, with the manifest path of the field (such as `manifest.graph.nodes[3].inputs[1]`)
or the header field name, and the file offset when there is one:

| Kind | Meaning |
|---|---|
| Malformed | not a well-formed file: bad magic, truncation, inconsistent header, invalid JSON, a value of the wrong JSON type, a range outside the file |
| InvalidModel | well formed but impossible: duplicate or undefined names, a byte length that does not match dims and dtype, a negative dimension |
| Unsupported | something this reader does not handle: a newer version, nonzero flags, an unknown dtype, attribute type or type kind, too much nesting, a tensor over 2^31 − 1 bytes |
| Io | the file could not be read or written |

## 4. Writing rules

A writer produces the same bytes for the same model: a single pass assigns offsets in the order of section 1.3, then writes the
manifest with no insignificant whitespace, then the header, the padding and the tensors. The C# writer emits fields in the order
of section 2's tables, writes optional strings `doc` only when non-empty, and always writes `name`, `domain`, `overload`,
`inputs`, `outputs` and `attributes` of a node. Other writers need not match its manifest byte for byte; the data region, the
header's fields and the meaning of the manifest are what the format fixes.

Weights that an ONNX file keeps in external data files are copied into the data region.

## 5. Loading

`Model.Load`, `Session.Load` and `ModelFormat.Load` read the file. `Model.Load` and `Session.Load` look at the first eight bytes to
choose between this format and ONNX. Files of 1 MiB or more are memory-mapped (where the platform allows it) and each tensor is a
view of the mapping, copied only when a session builds its weights; smaller files, streams and platforms without mapping read the
file into memory. Alignment to 64 bytes lets a consumer use mapped weights in place with any vector width.

Saving: `ModelFormat.Save(model, path | stream)` and `ModelFormat.Convert(onnxPath, outputPath)`.

## 6. Checks that back this text

* `tests/OneModel.Tests/Format/`: round trips (dtypes, strings, subgraphs, sparse, external data, mapped files), a hand-assembled
  file written from this text alone, and 64 hostile files in `testdata/format/hostile`, each named `<Kind>-<case>`.
* `tools/format_read.py`: a reader in Python with numpy only, written from this text; `tools/format_check.py` compares its view of a
  converted model with the ONNX file byte for byte.
* A fuzz target (`format`) in `fuzz/`.
