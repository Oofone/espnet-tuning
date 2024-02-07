#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_clean_100"
valid_set="dev"
test_sets="test_clean test_other dev_clean dev_other"

# asr_config=conf/train_asr.yaml
asr_config=conf/tuning/train_asr_whisper_full.yaml
# inference_config=conf/decode_asr.yaml
inference_config=conf/decode_asr_whisper_noctc_greedy.yaml

./asr.sh \
    --lang en \
    --ngpu 0 \
    --nj 16 \
    --stage 1 \
    --stop_stage 13 \
    --gpu_inference false \
    --inference_nj 4 \
    --token_type whisper_multilingual \
    --feats_normalize '' \
    --max_wav_duration 30 \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --audio_format "flac.ark" \
    --feats_type raw \
    --use_lm false \
    --cleaner whisper_en \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --inference_asr_model valid.acc.ave.pth \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
