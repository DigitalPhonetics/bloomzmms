#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_t
valid_set=dev_fleurs
test_sets="
test_fleurs
test_voxpopuli_cs
test_voxpopuli_de
test_voxpopuli_en
test_voxpopuli_es
test_voxpopuli_et
test_voxpopuli_fi
test_voxpopuli_fr
test_voxpopuli_hr
test_voxpopuli_hu
test_voxpopuli_it
test_voxpopuli_lt
test_voxpopuli_nl
test_voxpopuli_pl
test_voxpopuli_ro
test_voxpopuli_sk
test_voxpopuli_sl
test_mls_de
test_mls_en
test_mls_es
test_mls_fr
test_mls_it
test_mls_nl
test_mls_pl
test_mls_pt
"

asr_config=conf/tuning/train_asr_e_branchformer_mms1b-asr_bloomz7b_ctc.yaml
inference_config=conf/decode_asr_hf.yaml

./asr.sh \
    --ngpu 2 \
    --use_lm false \
    --token_type hugging_face \
    --hugging_face_model_name_or_path bigscience/bloomz-7b1 \
    --max_wav_duration 30 \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --audio_format "flac.ark" \
    --feats_normalize utterance_mvn \
    --inference_nj 1 \
    --gpu_inference true \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --ignore_init_mismatch true \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --cleaner whisper_basic \
    --hyp_cleaner whisper_basic \
    --lm_train_text "data/${train_set}/text" "$@"
