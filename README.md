# Teaching a Multilingual Large Language Model to Understand Multilingual Speech via Multi-Instructional Training

## Installation

1. Install the custom version of ESPnet:
```
git clone --branch v.202304 --depth 1 git@github.com:espnet/espnet.git /path/to/espnet-bloomzmms
```
2. Copy the modifications:
```
rsync -av espnet-bloomzmms-files/ /path/to/espnet-bloomzmms/
```
3. Follow the ESPnet installation instructions.

## Data Preparation

1. Run the data preparation stages:
```
cd /path/to/espnet-bloomzmms/egs2/fleurs/asr1
./run.sh --stop-stage 5
```

2. Download the synthetic multi-instractional training targets and create additional training data directories:
```
wget 'https://zenodo.org/records/10900287/files/tmi_training_targets.tar.gz?download=1' -O downloads/tmi_training_targets.tar.gz
mkdir downloads/tmi_training_targets
tar xf downloads/tmi_training_targets.tar.gz -C downloads/tmi_training_targets

mkdir dump/raw/train_tmi_sp dump/raw/train_mi_sp

cp -v downloads/tmi_training_targets/* dump/raw/train_tmi_sp/
cp -v dump/raw/train_t_sp/feats_type dump/raw/train_tmi_sp/
for f in wav.scp utt2spk; do
    for i in {1..3}; do
        perl -p -e 's/^(\S+)/$1-'$i'/' dump/raw/train_t_sp/$f >> dump/raw/train_tmi_sp/$f
    done
done
utils/fix_data_dir.sh dump/raw/train_tmi_sp

cp -v downloads/tmi_training_targets/* dump/raw/train_mi_sp/
cp -v dump/raw/train_t_sp/feats_type dump/raw/train_mi_sp/
for f in wav.scp utt2spk; do
    for i in {1..2}; do
        perl -p -e 's/^(\S+)/$1-'$i'/' dump/raw/train_t_sp/$f >> dump/raw/train_mi_sp/$f
    done
done
utils/fix_data_dir.sh dump/raw/train_mi_sp

```

## Training

Note: "pretrained" models in the following commands are outputs of the previous steps, it's not
necessary to download the pretrained models from the next section to perform training.
If you wish to skip training, you can download the pretrained models from the next section.

1. CTC pretraining:
```
./run.sh \
    --stage 6 \
    --stop-stage 11 \
    --pretrained_model downloads/bloomz_token_embeddings.pth
```

2. AED training:
```
./run.sh \
    --stage 11 \
    --stop-stage 11 \
    --ngpu 4 \
    --asr_config conf/tuning/train_asr_e_branchformer_mms1b-asr_bloomz7b_aed.yaml \
    --train_set train_tmi \
    --auxiliary_data_tags "decoder_prefix decoder_postfix" \
    --train_text_type text_int \
    --pretrained_model exp/asr_train_asr_e_branchformer_mms1b-asr_bloomz7b_ctc_raw_hugging_face_bigscience-bloomz-7b1_sp/valid.cer_ctc.best.pth:::ctc
```

### Pretrained models

- CTC: (akreal/bloomzmms-ctc)[https://huggingface.co/akreal/bloomzmms-ctc]
- AED:
  - T: (akreal/bloomzmms-aed-t)[https://huggingface.co/akreal/bloomzmms-aed-t]
  - MI: (akreal/bloomzmms-aed-mi)[https://huggingface.co/akreal/bloomzmms-aed-mi]
  - TMI: (akreal/bloomzmms-aed-tmi)[https://huggingface.co/akreal/bloomzmms-aed-tmi]

## Speech Recognition Inference

TBD

## Speech Translation Inference

TBD

## SpeechGLUE Inference

TBD

## SpeechXNLI Inference

TBD
