#!/usr/bin/env python3

import torch
import soundfile
import json
import sys

import numpy as np
import os
from datasets import load_dataset
from promptsource.templates import DatasetTemplates

from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet2.tasks.asr import ASRTask
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.token_id_converter import TokenIDConverter
from espnet2.torch_utils.device_funcs import to_device

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.file_utils import ModelOutput

from local.mms_glue import task2list

xnli_dir = sys.argv[1]
lang = sys.argv[2]
asr_expdir = sys.argv[3]
odir = sys.argv[4]

os.makedirs(odir, exist_ok=True)

sentence2wav = {}
with open(f"{xnli_dir}/sentences_{lang}.txt", encoding="utf-8") as f:
    i = 0
    for l in f:
        sentence2wav[
            l.strip("\n")
        ] = f"{xnli_dir}/audios/{lang}/xnli_validation_{i:04d}.flac"
        i += 1

dataset = load_dataset("xnli", lang, split="validation")

asr_train_config = f"{asr_expdir}/config.yaml"
asr_model_file = f"{asr_expdir}/valid.acc.ave.pth"
ngpu = 1
dtype = "float32"

if ngpu >= 1:
    device = "cuda:0"
else:
    device = "cpu"

task = ASRTask

asr_model, asr_train_args = task.build_model_from_file(
    asr_train_config, asr_model_file, device
)

asr_model.to(dtype=getattr(torch, dtype)).eval()

decoder = asr_model.decoder

token_type = asr_train_args.token_type
bpemodel = asr_train_args.bpemodel
tokenizer = build_tokenizer(token_type=token_type, bpemodel=bpemodel)
token_list = asr_model.token_list
converter = TokenIDConverter(token_list=token_list)

model_name_or_path = asr_train_args.decoder_conf["model_name_or_path"]
load_in_8bit = True

if torch.cuda.device_count() > 1:
    device_map = "balanced_low_0"
else:
    device_map = "auto"

hugging_face_model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    load_in_8bit=load_in_8bit,
    device_map=device_map,
)

word_embeddings = hugging_face_model.transformer.word_embeddings

hugging_face_linear_in = decoder.linear_in

speechglue_data = {}

prompts = DatasetTemplates("xnli/en")
template = prompts.templates["172b73dc-d045-491c-9dc2-76bf6566c8ee"]

answer_options = set()

for e in dataset:
    example = template.apply(e)
    answer_options.add(example[1])

print(f"Answer options: {answer_options}")
hf_tokenizer = AutoTokenizer.from_pretrained(bpemodel)
force_words_ids = hf_tokenizer.batch_encode_plus(
    list(answer_options), return_attention_mask=False
)["input_ids"]
max_answer_length = max([len(x) for x in force_words_ids])

refs = []
hyps = []

for i, l in enumerate(dataset):
    if i % 10 == 0:
        print(f"Processing XNLI {lang} sample {i}")

    refs.append(l["label"])

    speech = []
    text_fields = ["premise", "hypothesis"]

    for text_field in text_fields:
        sentence = l[text_field]
        wav_file = sentence2wav[sentence]
        wav, rate = soundfile.read(wav_file)
        wav = torch.tensor(wav)
        wav = wav.to(getattr(torch, dtype))
        speech.append(wav)

    lengths = torch.tensor([w.size(0) for w in speech], dtype=torch.long)
    batch = {"speech": pad_list(speech, 0.0), "speech_lengths": lengths}
    batch = to_device(batch, device=device)

    with torch.no_grad():
        enc, enc_olens = asr_model.encode(**batch)

    enc = hugging_face_linear_in(enc[0])
    encoded_speech = {f: enc[j, : enc_olens[j]] for j, f in enumerate(text_fields)}

    inputs_list = []

    for text in task2list["mnli_matched"]:
        if text.startswith("%"):
            inputs_list.append(encoded_speech[text[1:]].detach())
        else:
            inputs_list.append(
                word_embeddings(hf_tokenizer(text, return_tensors="pt")["input_ids"])
                .squeeze(0)
                .detach()
            )

    inputs_embeds = torch.cat(inputs_list).unsqueeze(0)
    if load_in_8bit:
        inputs_embeds = inputs_embeds.half()

    input_ids = torch.ones(
        [1, inputs_embeds.shape[1]],
        dtype=int,
        device=enc.device,
    )

    with torch.no_grad():
        outputs = hugging_face_model.generate(
            input_ids,
            inputs_embeds=inputs_embeds,
            num_beams=len(answer_options),
            force_words_ids=force_words_ids,
            max_new_tokens=max_answer_length,
            renormalize_logits=True,
        )

    hyp = [
        x.strip()
        for x in hf_tokenizer.batch_decode(
            outputs[:, input_ids.shape[1] :], skip_special_tokens=True
        )
    ]

    hyps.extend(hyp)

json.dump({"refs": refs, "hyps": hyps}, open(f"{odir}/xnli_{lang}.json", "w"))

print("Done")
