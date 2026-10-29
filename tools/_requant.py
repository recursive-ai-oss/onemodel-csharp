import os, sys, glob, numpy as np, onnx, onnxruntime as ort
from onnx import numpy_helper as nh
path = os.path.expanduser('~/onemodel/cache/models/resnet50/resnet50_int8_qdq.onnx')
ref = os.path.expanduser('~/onemodel/cache/reference/linux-x86_64/resnet50/int8_qdq/b1')
m = onnx.load(path)
init = {i.name: nh.to_array(i) for i in m.graph.initializer}
prod = {o: n for n in m.graph.node for o in n.output}
cons = {}
for n in m.graph.node:
    for i in n.input: cons.setdefault(i, []).append(n)
heavy = [n for n in m.graph.node if n.op_type == 'Conv']
conv = heavy[int(sys.argv[1])]
dqx, dqw, dqb = (prod[i] for i in conv.input[:3])
q = cons[conv.output[0]][0]
xq_name = dqx.input[0]
sx, zx = float(init[dqx.input[1]]), int(init[dqx.input[2]])
W = init[dqw.input[0]].astype(np.int64); sw = init[dqw.input[1]].astype(np.float32); zw = init[dqw.input[2]].astype(np.int64)
B = init[dqb.input[0]].astype(np.int64)
sy, zy = float(init[q.input[1]]), int(init[q.input[2]])
attrs = {a.name: onnx.helper.get_attribute_value(a) for a in conv.attribute}
print('conv', conv.output[0], 'W', W.shape, attrs, 'sx', sx, 'zx', zx, 'sy', sy, 'zy', zy, 'sw0', sw[:3])
feeds = {os.path.basename(f)[:-4]: np.load(f) for f in glob.glob(ref + '/inputs/*.npy')}
tmp = os.path.expanduser('~/onemodel/u14/rq.onnx')
onnx.utils.extract_model(path, tmp, [i.name for i in m.graph.input], [xq_name, q.output[0]])
res = {}
for lvl in ('DISABLE_ALL', 'ENABLE_ALL'):
    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    so.graph_optimization_level = getattr(ort.GraphOptimizationLevel, 'ORT_' + lvl)
    res[lvl] = ort.InferenceSession(tmp, so, providers=['CPUExecutionProvider']).run(None, feeds)
x = res['ENABLE_ALL'][0].astype(np.int64)
ya, yd = res['ENABLE_ALL'][1], res['DISABLE_ALL'][1]
print('ORT ALL vs DISABLE on this layer', int((ya != yd).sum()), 'of', ya.size)
# accumulate exactly
N, C, H, Wd = x.shape
M, Cg, KH, KW = W.shape
pads = attrs.get('pads', [0] * 4); st = attrs.get('strides', [1, 1]); dil = attrs.get('dilations', [1, 1])
xp = np.pad(x - zx, ((0, 0), (0, 0), (pads[0], pads[2]), (pads[1], pads[3])))
OH = (xp.shape[2] - dil[0] * (KH - 1) - 1) // st[0] + 1; OW = (xp.shape[3] - dil[1] * (KW - 1) - 1) // st[1] + 1
acc = np.zeros((N, M, OH, OW), np.int64)
for ky in range(KH):
    for kx in range(KW):
        patch = xp[:, :, ky * dil[0]: ky * dil[0] + (OH - 1) * st[0] + 1: st[0], kx * dil[1]: kx * dil[1] + (OW - 1) * st[1] + 1: st[1]]
        acc += np.einsum('nchw,mc->nmhw', patch, (W[:, :, ky, kx] - zw[:, None]))
acc += B[None, :, None, None]
f = np.float32
lo, hi = -128, 127
def Q(v): return np.clip(v + zy, lo, hi).astype(np.int8)
a32 = acc.astype(f)
swv = sw[None, :, None, None]
swd = sw.astype(np.float64)[None, :, None, None]
import itertools
cands = {
  'float(double sx*sw/sy)': Q(np.rint(a32 * (np.float64(sx) * swd / np.float64(sy)).astype(f))),
  'float(double sx*(sw/sy))': Q(np.rint(a32 * (np.float64(sx) * (swd / np.float64(sy))).astype(f))),
  'float(double (sx/sy)*sw)': Q(np.rint(a32 * ((np.float64(sx) / np.float64(sy)) * swd).astype(f))),
  'float: sw*sx then /sy (f32)': Q(np.rint(a32 * (swv * f(sx)) / f(sy))),
  'float: (sx/sy) f32 * sw': Q(np.rint(a32 * (f(sx) / f(sy)) * swv)),
  'float: acc*sx*(sw/sy)': Q(np.rint((a32 * f(sx)) * (swv / f(sy)))),
  'float: sx*(sw/sy) f32 mult': Q(np.rint(a32 * (f(sx) * (swv / f(sy))))),

  'ours: rint(acc*(sx*sw/sy))': Q(np.rint(a32 * (f(sx) * swv / f(sy)))),
  'rint(acc*(sx*sw)/sy)': Q(np.rint(a32 * (f(sx) * swv) / f(sy))),
  'rint((acc*sx*sw)/sy)': Q(np.rint(((a32 * f(sx)) * swv) / f(sy))),
  'rint(acc*(sx/sy*sw))': Q(np.rint(a32 * ((f(sx) / f(sy)) * swv))),
  'rint(acc*sw*(sx/sy))': Q(np.rint((a32 * swv) * (f(sx) / f(sy)))),
  'double': Q(np.rint(acc.astype(np.float64) * float(sx) * sw.astype(np.float64)[None, :, None, None] / float(sy))),
  'double2': Q(np.rint(acc.astype(np.float64) * (float(sx) * sw.astype(np.float64)[None, :, None, None] / float(sy)))),
}
for k, v in cands.items():
    print(f'{k:32s} vs ORT ALL: {int((v != ya).sum()):6d}   vs ORT DISABLE: {int((v != yd).sum()):6d}')
