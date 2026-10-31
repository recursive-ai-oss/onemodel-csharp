#!/usr/bin/env python3
"""Makes a patched copy of a CRNN line-recognizer training tree so a fine-tune can be run as a black-box reference.

    prepare_ref.py --src TRAINING_DIR --dst COPY_DIR

The source tree is only read. The copy gets these changes to train/run.py, each asserted so a changed source fails loudly:

* the device comes from REF_DEVICE (default "cpu") and bf16 autocast is off on the CPU, so the reference is a plain fp32 run;
* every step dumps the batch the model sees (after augmentation, rounded to bytes, padded) to REF_DUMP/batch-<step>.bin,
  and the step's loss, gradient norm and learning rate to REF_DUMP/steps.jsonl;
* REF_TF32=0 disables TF32 in cuDNN convolutions;
* REF_STEPS overrides the step count and REF_EVAL_EVERY the evaluation interval.

Batch file layout (little-endian): int32 B, int32 W, int32 widths[B], int32 target_lengths[B], int32 targets[sum],
then B*32*W bytes (uint8, row-major [B][32][W]).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def patch(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, f"expected exactly one occurrence of: {old!r} (found {text.count(old)})"
    return text.replace(old, new)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(".venv", "__pycache__", ".git"))

    p = dst / "train" / "run.py"
    t = p.read_text(encoding="utf-8")
    t = patch(t, 'dev = "cuda"', 'dev = os.environ.get("REF_DEVICE", "cpu")')
    assert t.count(".pin_memory()") == 2
    t = t.replace(".pin_memory()", "")
    t = patch(t, 'with torch.autocast("cuda", dtype=torch.bfloat16):',
              'with (torch.autocast("cuda", dtype=torch.bfloat16) if dev == "cuda" and os.environ.get("REF_BF16", "1") == "1" else __import__("contextlib").nullcontext()):')
    # cuDNN convolutions use TF32 by default on recent GPUs; REF_TF32=0 turns that off so the run is plain fp32
    t = patch(t, "    random.seed(seed)\n", "    random.seed(seed)\n    torch.backends.cudnn.allow_tf32 = os.environ.get(\"REF_TF32\", \"1\") == \"1\"\n")
    # dump the batch the model sees
    t = patch(t, "            x = to_input(x, dev)\n",
              "            _dump_batch(step, x, widths, tgt, tlen)\n            x = to_input(x, dev)\n")
    t = patch(t, "        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)\n",
              "        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0))\n"
              "        _dump_step(step, float(loss), gnorm, lr_at(step, tc))\n")
    t = patch(t, "def lr_at(step: int, tc: dict) -> float:", '''def _dump_batch(step, x, widths, tgt, tlen):
    d = os.environ.get("REF_DUMP")
    if not d:
        return
    xb = x.to(torch.uint8).contiguous().cpu().numpy()
    b, _, h, w = xb.shape
    assert h == 32
    hdr = np.array([b, w], dtype="<i4").tobytes()
    with open(Path(d) / f"batch-{step:05d}.bin", "wb") as f:
        f.write(hdr)
        f.write(widths.to(torch.int32).numpy().astype("<i4").tobytes())
        f.write(tlen.to(torch.int32).numpy().astype("<i4").tobytes())
        f.write(tgt.to(torch.int32).numpy().astype("<i4").tobytes())
        f.write(xb.tobytes())


def _dump_step(step, loss, gnorm, lr):
    d = os.environ.get("REF_DUMP")
    if d:
        with open(Path(d) / "steps.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"step": step, "loss": loss, "grad_norm": gnorm, "lr": lr}) + "\\n")


def lr_at(step: int, tc: dict) -> float:''')
    # step count and evaluation interval from the environment
    t = patch(t, '    tc = cfg["train"]\n',
              '    tc = cfg["train"]\n'
              '    tc["steps"] = int(os.environ.get("REF_STEPS", tc["steps"]))\n'
              '    tc["eval_every"] = int(os.environ.get("REF_EVAL_EVERY", tc["eval_every"]))\n')
    p.write_text(t, encoding="utf-8")


if __name__ == "__main__":
    main()
