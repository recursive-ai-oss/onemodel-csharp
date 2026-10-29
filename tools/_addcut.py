import sys, os, glob, numpy as np, onnx, onnxruntime as ort
from onnx import numpy_helper as nh
path = os.path.expanduser('~/onemodel/cache/models/resnet50/resnet50_int8_qdq.onnx')
ref = os.path.expanduser('~/onemodel/cache/reference/linux-x86_64/resnet50/int8_qdq/b1')
m = onnx.load(path)
init = {i.name: nh.to_array(i) for i in m.graph.initializer}
prod = {o: n for n in m.graph.node for o in n.output}
cons = {}
for n in m.graph.node:
    for i in n.input: cons.setdefault(i, []).append(n)
adds = [n for n in m.graph.node if n.op_type == 'Add']
add = adds[int(sys.argv[1]) if len(sys.argv) > 1 else 0]
dqa, dqb = prod[add.input[0]], prod[add.input[1]]
qy = cons[add.output[0]][0]
print('add', add.output[0], 'Q', qy.output[0])
def params(dq): return float(init[dq.input[1]]), int(init[dq.input[2]])
sa, za = params(dqa); sb, zb = params(dqb); sy, zy = float(init[qy.input[1]]), int(init[qy.input[2]])
print('sa za', sa, za, 'sb zb', sb, zb, 'sy zy', sy, zy, init[qy.input[2]].dtype)
inputs = [i.name for i in m.graph.input]
feeds = {os.path.basename(f)[:-4]: np.load(f) for f in glob.glob(ref + '/inputs/*.npy')}
tmp = os.path.expanduser('~/onemodel/u14/addcut.onnx')
onnx.utils.extract_model(path, tmp, inputs, [dqa.input[0], dqb.input[0], qy.output[0]])
so = ort.SessionOptions(); so.intra_op_num_threads = 1
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
a, b, yd = ort.InferenceSession(tmp, so, providers=['CPUExecutionProvider']).run(None, feeds)
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
a2, b2, ya = ort.InferenceSession(tmp, so, providers=['CPUExecutionProvider']).run(None, feeds)
print('inputs identical across levels', (a == a2).all(), (b == b2).all(), 'y diff DISABLE vs ALL', int((yd != ya).sum()), 'of', yd.size)
f32 = np.float32
da = (a.astype(f32) - f32(za)) * f32(sa); db = (b.astype(f32) - f32(zb)) * f32(sb)
lo, hi = (-128, 127) if yd.dtype == np.int8 else (0, 255)
c1 = np.clip(np.rint((da + db) / f32(sy)) + zy, lo, hi).astype(yd.dtype)
print('float chain vs DISABLE', int((c1 != yd).sum()), 'vs ALL (fused, from same inputs)', int((c1 != ya).sum()))
