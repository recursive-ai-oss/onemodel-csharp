import os, sys, glob, shutil, numpy as np, onnx, onnxruntime as ort
from onnx import helper as h
src = os.path.expanduser('~/onemodel/u14/cutadd'); out = os.path.expanduser('~/onemodel/u14/cutadd2')
shutil.rmtree(out, ignore_errors=True); os.makedirs(out + '/ref/outputs')
shutil.copytree(src + '/ref/inputs', out + '/ref/inputs')
m = onnx.load(src + '/model.onnx')
add = next(n for n in m.graph.node if n.op_type == 'Add')
for name, t in (('t_a', add.input[0]), ('t_b', add.input[1]), ('t_sum', add.output[0])):
    m.graph.node.append(h.make_node('Identity', [t], [name]))
    m.graph.output.append(h.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None))
onnx.save(m, out + '/model.onnx')
feeds = {os.path.basename(f)[:-4]: np.load(f) for f in glob.glob(out + '/ref/inputs/*.npy')}
so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess = ort.InferenceSession(out + '/model.onnx', so, providers=['CPUExecutionProvider'])
for o, v in zip(sess.get_outputs(), sess.run(None, feeds)):
    np.save(out + f'/ref/outputs/{o.name}.npy', v)
print([o.name for o in sess.get_outputs()])
