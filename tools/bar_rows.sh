#!/usr/bin/env bash
# The reference-set rows shared by tools/gpu_browser.sh and tools/bar.sh. Source it: `. tools/bar_rows.sh` (needs tools/env.sh first).
# Plain variables and indirect expansion, so it also runs under the bash 3.2 that macOS ships.
# row = model file under the model cache | GpuReport arguments. Image rows take the size of their reference inputs.
ROWS=""
row() { eval "FILE_$1=\$2; ARGS_$1=\$3"; ROWS="$ROWS $1"; }
file_of() { local v="FILE_$1"; echo "${!v:-}"; }
args_of() { local v="ARGS_$1"; echo "${!v:-}"; }
row minilm_fp32_b1   minilm_l6/minilm_l6.onnx                       "--batch 1 --seq 128"
row minilm_fp32_b8   minilm_l6/minilm_l6.onnx                       "--batch 8 --seq 128"
row minilm_fp16_b1   minilm_l6/minilm_l6_fp16.onnx                  "--batch 1 --seq 128"
row minilm_int8_b1   minilm_l6/minilm_l6_int8_dyn.onnx              "--batch 1 --seq 128"
row bert_fp32_b1     bert_base/bert_base.onnx                       "--batch 1 --seq 128"
row bert_fp32_b8     bert_base/bert_base.onnx                       "--batch 8 --seq 128"
row bert_int8_b1     bert_base/bert_base_int8_dyn.onnx              "--batch 1 --seq 128"
row resnet50_fp32_b1 resnet50/resnet50.onnx                         "--batch 1"
row resnet50_fp32_b8 resnet50/resnet50.onnx                         "--batch 8"
row resnet50_fp16_b1 resnet50/resnet50_fp16.onnx                    "--batch 1"
row resnet50_int8_b1 resnet50/resnet50_int8_qdq.onnx                "--batch 1"
row mobilenet_fp32_b1 mobilenet_v3_large/mobilenet_v3_large.onnx    "--batch 1"
row mobilenet_fp32_b8 mobilenet_v3_large/mobilenet_v3_large.onnx    "--batch 8"
row mobilenet_int8_b1 mobilenet_v3_large/mobilenet_v3_large_int8_qdq.onnx "--batch 1"
row dbnet_fp32_b1    dbnet/dbnet_lite_v1.onnx                       "--batch 1 --size 640x640"
# The recognizer takes a real line image (the reference set's first sample) so that its decoded ids are not empty.
crnn_input="$(ls "$ONEMODEL_CACHE"/reference/*/crnn/fp32/b1-val-00000539/inputs/input.npy 2>/dev/null | head -1)"
row crnn_fp32_b1     crnn/crnn_v0.onnx                              "--batch 1 ${crnn_input:+--input $crnn_input}"
row whisper_enc_b1   whisper_small/whisper_small_encoder.onnx       "--batch 1"
row whisper_dec_b1   whisper_small/whisper_small_decoder.onnx       "--batch 1 --seq 8"
row whisper_enc_int8_b1 whisper_small/whisper_small_encoder_int8_dyn.onnx "--batch 1"
row whisper_dec_int8_b1 whisper_small/whisper_small_decoder_int8_dyn.onnx "--batch 1 --seq 8"
# A greedy decode through the decoder on encoder states made by the CPU, free-running tokens against the CPU exact mode.
row whisper_greedy   whisper_small/whisper_small_decoder.onnx       "--greedy-encoder $ONEMODEL_CACHE/models/whisper_small/whisper_small_encoder.onnx --steps 8 --batch 1 --runs 1 --threads 8"

