# Teaching a Multilingual Large Language Model to Understand Multilingual Speech via Multi-Instructional Training
[[Paper]](https://aclanthology.org/2024.findings-naacl.52/)

## Installation

1. Install the custom version of ESPnet:
```
git clone --branch v.202304 --depth 1 git@github.com:espnet/espnet.git /path/to/espnet-bloomzmms
```
2. Copy the modifications:
```bash
rsync -av espnet-bloomzmms-files/ /path/to/espnet-bloomzmms/
```
3. Follow the ESPnet installation instructions.

## Data Preparation

1. Run the data preparation stages:
```bash
cd /path/to/espnet-bloomzmms/egs2/fleurs/asr1
./run.sh --stop-stage 5
```

2. Download the synthetic multi-instractional training targets and create additional training data directories:
```bash
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
```bash
./run.sh \
    --stage 6 \
    --stop-stage 11 \
    --pretrained_model downloads/bloomz_token_embeddings.pth
```

2. AED training:
```bash
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

- CTC: [akreal/bloomzmms-ctc](https://huggingface.co/akreal/bloomzmms-ctc)
- CE:
  - T: [akreal/bloomzmms-ce-t](https://huggingface.co/akreal/bloomzmms-ce-t)
  - MI: [akreal/bloomzmms-ce-mi](https://huggingface.co/akreal/bloomzmms-ce-mi)
  - TMI: [akreal/bloomzmms-ce-tmi](https://huggingface.co/akreal/bloomzmms-ce-tmi)

## Speech Recognition Inference

```bash
./run.sh \
    --stage 12 \
    --asr_config conf/tuning/train_asr_e_branchformer_mms1b-asr_bloomz7b_aed.yaml
```

## Speech Translation Inference

1. Run the data preparation steps.

For CoVoST 2, run:
```bash
./run.sh \
    --stage 3 \
    --stop-stage 3 \
    --test_sets test_covost2_en-de
```

For FLEURS, run:
```bash
local/prepare_fleurs_translate.py en-us de-de
```

2. Generate a config with language-specific instructions:
```bash
perl -p \
    -e 's/prefix: "Repeat the sentence: "/prefix: "Translate the following text from English to German\\n"/;' \
    -e 's/postfix: ". "/postfix: "\\n"/;'  -e 's/keep_nbest_models: 3/keep_nbest_models: 1/;' \
    -e 's/max_epoch: 70/max_epoch: 0/;' \
        conf/tuning/train_asr_e_branchformer_mms1b-asr_bloomz7b_aed.yaml \
        > conf/tuning/train_asr_e_branchformer_mms1b-asr_bloomz7b_aed_translate_en-de.yaml
```

3. Run the model construction and inference stages:
```bash
./run.sh \
    --stage 11 \
    --test_sets test_covost2_en-de \
    --pretrained_model exp/asr_train_asr_e_branchformer_mms1b-asr_bloomz7b_aed_raw_hugging_face_bigscience-bloomz-7b1_sp/valid.acc.ave.pth
    --asr_config conf/tuning/train_asr_e_branchformer_mms1b-asr_bloomz7b_aed_translate_en-de.yaml
```

## SpeechGLUE Inference

1. Prepare the SpeechGLUE data: [https://github.com/ashi-ta/speechGLUE?tab=readme-ov-file#data-preparation](https://github.com/ashi-ta/speechGLUE?tab=readme-ov-file#data-preparation).

2. Run the inference script:
```bash
mkdir speechglue_output

. path.sh

for task in cola mnli_matched mnli_mismatched mrpc qnli qqp rte sst2 stsb wnli; do
    local/speechglue_inference.py \
        /path/to/speechGLUE/dump \
        ${task} \
        exp/asr_train_asr_e_branchformer_mms1b-asr_bloomz7b_aed_raw_hugging_face_bigscience-bloomz-7b1_sp \
        speechglue_output
done
```

## SpeechXNLI Inference

1. Download the SpeechXNLI data:
```bash
wget https://zenodo.org/records/10900287/files/speech_xnli.tar.gz?download=1 -O speech_xnli.tar.gz
tar xvf speech_xnli.tar.gz
```

2. Run the inference script:
```bash
mkdir speechxnli_output

. path.sh

for lang in ar bg de el en es fr hi ru sw th tr ur vi zh; do
    local/speechxnli_inference.py \
        /path/to/speechXNLI \
        ${lang} \
        exp/asr_train_asr_e_branchformer_mms1b-asr_bloomz7b_aed_raw_hugging_face_bigscience-bloomz-7b1_sp \
        speechxnli_output
done
```

## Citation
```
@inproceedings{denisov-vu-2024-teaching,
    title = "Teaching a Multilingual Large Language Model to Understand Multilingual Speech via Multi-Instructional Training",
    author = "Denisov, Pavel  and
      Vu, Thang",
    editor = "Duh, Kevin  and
      Gomez, Helena  and
      Bethard, Steven",
    booktitle = "Findings of the Association for Computational Linguistics: NAACL 2024",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-naacl.52",
    pages = "814--834",
}
```
