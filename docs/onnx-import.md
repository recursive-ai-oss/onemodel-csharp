# ONNX import

How a model file becomes the in-memory graph. A port can follow this text without the C# code. The schema is `onnx.proto` and the
rules are ONNX's `IR.md`; the wire format is the protobuf encoding specification. Nothing here depends on a protobuf library.

## 1. Protobuf reading

The reader works on one contiguous byte range and never reads outside it.

* **Keys.** A key is a varint of at most 32 bits: `field << 3 | wire`. Field 0 is an error. Wire types 0 (varint), 1 (64-bit),
  2 (length-delimited) and 5 (32-bit) are read. Wire types 3 and 4 (groups) and 6 and 7 are errors.
* **Varints.** At most 10 bytes. The tenth byte may be 0 or 1; anything larger overflows 64 bits and is an error. A varint cut off by
  the end of the data is an error. Overlong encodings (`0x81 0x80 0x00`) are accepted, as the specification allows.
* **Lengths.** A length is read as 64 bits and compared with the bytes that remain *before* any slice is taken. A length larger than
  the remainder is an error. Nothing is allocated from a length.
* **Integers.** `int32` and enum fields keep the low 32 bits of the varint, as protobuf does. `int64` and `uint64` keep all 64.
* **Strings** (`string` fields) must be valid UTF-8; invalid bytes are an error. `bytes` fields are never decoded.
* **Repeated scalars** are accepted packed (one length-delimited field) or unpacked (one key per element), mixed in any order.
* **Unknown fields**, and known fields that arrive with the wrong wire type, are skipped.
* **Repeated occurrences** of a singular field: the last one wins. Singular message fields are not merged.
* **Nesting.** Each nested message is one level. Reading stops with an error beyond `MaxNesting` levels (default 128, range 8 to 250).
  A graph inside a node's attribute inside a graph costs three levels.
* **Size.** A model file can be at most 2^31 − 1 bytes (protobuf's own limit). Bigger models keep their weights in external data.

## 2. Messages and fields used

| Message | Fields read (number) |
|---|---|
| ModelProto | ir_version 1, producer_name 2, producer_version 3, domain 4, model_version 5, doc_string 6, graph 7, opset_import 8, metadata_props 14, functions 25. `training_info` and `configuration` are skipped |
| OperatorSetIdProto | domain 1, version 2 |
| GraphProto | node 1, name 2, initializer 5, doc_string 10, input 11, output 12, value_info 13, sparse_initializer 15. `quantization_annotation` and `metadata_props` are skipped |
| NodeProto | input 1, output 2, name 3, op_type 4, attribute 5, doc_string 6, domain 7, overload 8. `metadata_props` and `device_configurations` are skipped |
| AttributeProto | name 1, f 2, i 3, s 4, t 5, g 6, floats 7, ints 8, strings 9, tensors 10, graphs 11, doc_string 13, tp 14, type_protos 15, type 20, ref_attr_name 21, sparse_tensor 22, sparse_tensors 23 |
| TensorProto | dims 1, data_type 2, segment 3, float_data 4, int32_data 5, string_data 6, int64_data 7, name 8, raw_data 9, double_data 10, uint64_data 11, doc_string 12, external_data 13, data_location 14 |
| SparseTensorProto | values 1, indices 2, dims 3 |
| ValueInfoProto | name 1, type 2 |
| TypeProto | tensor_type 1, sequence_type 4, map_type 5, optional_type 9, sparse_tensor_type 8, opaque_type 7 |
| TensorShapeProto.Dimension | dim_value 1, dim_param 2 (last one set wins) |
| FunctionProto | name 1, input 4, output 5, attribute 6, node 7, doc_string 8, opset_import 9, domain 10, attribute_proto 11, overload 13. `value_info` and `metadata_props` are skipped |

**Model checks.** `ir_version` must be present and at least 3; a larger version than any known is accepted (new fields are skipped).
A model without a graph is an error.

**Attributes.** If `type` is absent the type is inferred from the field that is set (`f`, `i`, `s`, `t`, `g`, `sparse_tensor`, `tp`,
then the repeated fields in the order floats, ints, strings, tensors, graphs, sparse_tensors, type_protos). An attribute with
neither a type nor a value (and no `ref_attr_name`) is an error. A scalar attribute whose field is absent reads as 0 or empty. An
attribute declared as tensor, graph, sparse tensor or type whose field is absent is an error. An attribute with `ref_attr_name`
(function bodies) carries no value.

**Types.** A `TypeProto` with no member set is "no type". A tensor type that has a `shape` field has a shape (possibly rank 0);
one without it has unknown rank. A dimension with neither `dim_value` nor `dim_param` is unknown.

## 3. Tensors

A tensor has an element type code (1 to 28, `TensorProto.DataType`), dimensions, and its values as **little-endian bytes** in the
raw layout of `TensorProto.raw_data`.

* Dimensions must be non-negative. The element count is their product, which must not exceed 2^60.
* The expected byte length is `ceil(count × bits / 8)` with bits: 32 for float, int32, uint32; 8 for uint8, int8, bool, the 8-bit
  floats; 16 for uint16, int16, float16, bfloat16; 64 for int64, uint64, double, complex64; 128 for complex128; 4 for the 4-bit
  types; 2 for the 2-bit types; 6 for the 6-bit floats. Sub-byte elements are packed as `onnx.proto` describes. A tensor that would
  need more than 2^31 − 1 bytes is *unsupported*, and this is decided from the dimensions alone, before any data is read.
* **Where the values come from**, in this order: `data_location = EXTERNAL` (section 5); else `raw_data`; else the typed field
  for the element type. Mixing sources is an error. A tensor with elements and no source is an error (its byte length is zero).
* **Typed fields**, converted to the raw layout:
  * `float_data` (float, complex64): 4 bytes per entry. `double_data` (double, complex128): 8 bytes. `int64_data` (int64): 8 bytes.
  * `uint64_data` serves uint64 (8 bytes) and uint32 (the low 32 bits, 4 bytes).
  * `int32_data` serves every other numeric type, one element per entry: int32 as is; 16-bit types keep the low 16 bits; 8-bit types
    and bool keep the low 8 bits (bool: any non-zero entry becomes 1). For 4-bit and 2-bit types an entry's low byte is one *packed*
    byte. For the 6-bit types an entry's low 6 bits are one element and the elements are packed into the 4-in-3-bytes bit stream.
  * A typed field that does not belong to the element type is an error.
  * The resulting byte length must equal the expected length.
* **Strings** (type 8) live only in `string_data`, one entry per element; the count must equal the element count.
* **Segments** (`segment`) are not supported.
* `raw_data` must have exactly the expected length. Values are not byte-swapped; raw data is little-endian by definition.

A sparse tensor is imported as written (`values`, `indices`, dense `dims`) and is not converted to dense.

## 4. The graph

Reading produces the parsed graph; resolution then builds the graph proper.

* **Values.** Per graph, in this order: each input (name must be non-empty and unique); each initializer (an initializer whose name is
  an input that has no initializer yet becomes that input's default value, which is how IR versions before 4 list initializers; any other
  repeated name is an error); each sparse initializer; each non-empty node output. A name defined twice is an error. An empty name
  among inputs or outputs of a *node* means "omitted optional" and gives no value.
* **Scope.** A subgraph (an attribute of type graph or graphs) resolves after its enclosing graph has defined all its values, and
  sees them; names it defines itself shadow outer names. A subgraph's `Captures` list the outer values it (or its own subgraphs) reads.
* **References.** Every non-empty node input, and every graph output, must name a defined value. Otherwise: error, naming the node.
* **Opsets.** A node's opset version is the version imported for its domain; the empty domain and `ai.onnx` are the same domain.
  A node whose domain is not imported is an error. Function bodies are not resolved.
* **Types** come from input and output declarations first, and from `value_info` for values that still have none. A `value_info` for a
  name that is not defined is ignored.
* **Order.** IR.md requires nodes in topological order. Input in that order is kept as is. Otherwise nodes are sorted: repeatedly take
  the node with the smallest file index whose inputs (including values read by its subgraphs) are all produced; the flag
  `NodesWereReordered` records that this happened. A cycle is an error.
* **Consumers.** For each value, the nodes that read it in sorted order, each once; a node counts as a reader of what its subgraphs read.
* **Functions** are imported as written (name, domain, overload, formal inputs, outputs and attributes, attribute defaults, body nodes
  by name, opset imports). They are not expanded.

## 5. External data

`data_location = EXTERNAL` takes the values from a file. Entries of `external_data`: `location` (required), `offset` (default 0),
`length` (default: the tensor's expected length), `checksum` (ignored).

1. `location` must be a relative path: not empty, no NUL, not starting with `/` or `\`, not `X:`-prefixed, not rooted on the platform,
   and with no `..` segment when split at either `/` or `\`.
2. It is joined to the **external-data directory** (for a file path, the model's directory; for a stream or byte array, only if the
   caller names one) and every symbolic link on the way is resolved. The resolved file must lie inside the resolved directory and be a
   file. Otherwise: external-data error.
3. `offset` and `length` must be non-negative decimal integers. `length`, if given, must equal the expected length. `offset + length`
   must not exceed the file's size (checked in 64 bits).
4. The file is memory-mapped where the platform allows it (one mapping per file, shared by its tensors) and the tensor is a view into
   it. Where mapping is unavailable, only the requested range is read into memory.

A swap of a file or link after the check is not defended against.

## 6. Memory

Initializer bytes are views: into the mapped model file, into a byte array the caller supplied, or into a small array holding converted
typed-field data. Importing therefore allocates only the graph itself. Disposing the model releases the mappings; reading tensor data
afterwards throws. Copying a tensor out (`ToTensor`) is the only step that allocates its bytes. Models up to 1 MiB are read instead of
mapped. 

Indicative figures from one Apple M2 (arm64, .NET 10, warm file cache, three runs, the steady ones), importing the whole graph and
touching no tensor data. Peak resident size is the operating system's figure for the process, including the runtime (about 45 MiB):

| Model (file size) | Mapped: load, peak RSS | Read into memory: load, peak RSS |
|---|---|---|
| BERT-base (416 MiB) | 17 ms, 48 MiB | 68 ms, 463 MiB |
| MiniLM-L6 (86 MiB) | 17 ms, 46 MiB | 66 ms, 132 MiB |
| Whisper-small decoder (738 MiB) | 19 ms, 51 MiB | 110 ms, 787 MiB |
| Whisper-small encoder (336 MiB) | 18 ms, 48 MiB | 55 ms, 383 MiB |

On an x64 desktop (Ryzen 9 7950X3D, Linux) mapped loads took 21 to 25 ms and peaked at 181 to 281 MiB, because Linux counts the file pages
it reads ahead around each header; reading took 140 to 300 ms and 370 to 790 MiB. The managed heap after loading is 3 to 5 MiB when mapped.
These are single-machine, indicative figures, not a gated benchmark.

## 7. Errors

Everything wrong with the input is one exception type carrying a kind (`Malformed` for bytes that are not valid protobuf, `InvalidModel`
for valid protobuf that is not a valid model, `Unsupported`, `ExternalData`, `Io`), the input's name, the byte offset when one is known,
and the field path (`ModelProto.graph.node[3].attribute[1].t`). Offsets are from the start of the model file or byte array.

## 8. How the importer is checked

* `tests/OnnxSweep` imports every `model.onnx` of the ONNX backend test data and writes a structural dump of each (producer, opsets,
  every node with its attributes, value types, initializer element type, dimensions and SHA-256 of the data). `tools/onnx_dump.py` writes
  the same dump from the `onnx` Python package using only `onnx.load` and the message fields, and `tools/compare_onnx_dump.sh`
  diffs the two. Floats are compared as bit patterns.
* The hostile inputs in `testdata/onnx/hostile` are named `<kind>-<case>.onnx`; the test suite regenerates them from code and expects
  each to be refused with that kind.
* `fuzz/` holds libFuzzer targets for the model reader and the tensor reader; `tools/setup_fuzz.sh` and `tools/run_fuzz.sh` run them.
