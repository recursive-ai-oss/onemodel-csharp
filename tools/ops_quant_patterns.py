import sys, collections, onnx
for path in sys.argv[1:]:
    m = onnx.load(path, load_external_data=False)
    g = m.graph
    prod = {}
    cons = collections.defaultdict(list)
    for n in g.node:
        for o in n.output: prod[o] = n
        for i in n.input: cons[i].append(n)
    init = {i.name for i in g.initializer}
    print("==", path.split('/')[-1], len(g.node), "nodes")
    pats = collections.Counter()
    for n in g.node:
        if n.op_type in ("Conv","Gemm","MatMul","Add","Mul","Relu","HardSwish","HardSigmoid","MaxPool","GlobalAveragePool","Flatten"):
            ins = []
            for i in n.input:
                if i in init: ins.append("init")
                elif i in prod: ins.append(prod[i].op_type)
                else: ins.append("graph")
            outs = sorted({c.op_type for o in n.output for c in cons[o]})
            pats[(n.op_type, tuple(ins), tuple(outs))] += 1
    for k, v in sorted(pats.items(), key=lambda kv: -kv[1]): print(v, k)
    # DQ/Q detail
    c = collections.Counter()
    for n in g.node:
        if n.op_type in ("QuantizeLinear","DequantizeLinear"):
            ax = [a.i for a in n.attribute if a.name=="axis"]
            c[(n.op_type, "axis%s"%ax, tuple("init" if i in init else "dyn" for i in n.input))] += 1
    print(dict(c))
    print({i.name: (i.data_type, list(i.dims)) for i in g.initializer if i.name.endswith('zero_point') or i.name.endswith('scale')} and "has scale/zp inits")
    print("opsets", [(o.domain, o.version) for o in m.opset_import])
