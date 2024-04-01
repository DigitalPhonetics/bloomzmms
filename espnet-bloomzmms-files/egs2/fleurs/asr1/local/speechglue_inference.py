#!/usr/bin/env python3

import csv
import torch
import soundfile
import json
import sys

import numpy as np
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

glue_task = sys.argv[1]

task2text = {
    "cola": ["sentence"],
    "mnli_matched": ["premise", "hypothesis"],
    "mnli_mismatched": ["premise", "hypothesis"],
    "mrpc": ["sentence1", "sentence2"],
    "qnli": ["question", "sentence"],
    "qqp": ["question1", "question2"],
    "rte": ["sentence1", "sentence2"],
    "sst2": ["sentence"],
    "stsb": ["sentence1", "sentence2"],
    "wnli": ["sentence1", "sentence2"],
}

if glue_task == "mnli_matched":
    dataset = load_dataset("glue", "mnli", split="validation_matched")
elif glue_task == "mnli_mismatched":
    dataset = load_dataset("glue", "mnli", split="validation_mismatched")
else:
    dataset = load_dataset("glue", glue_task, split="validation")

task2template = {
    "mnli_matched": "f3ebe1ac-194b-41e7-b008-36eafdbfbe25",
    "mnli_mismatched": "770aa883-efec-4258-9e1e-1a96d0c20ed5",
    "cola": "39a701ff-bb4b-48ac-8c0a-8c61bf0d4b8d",
    "sst2": "63c6b2be-8ecd-42ad-88c7-0d1dc1a8323a",
    "mrpc": "adf659af-4e2d-4e7e-ab89-b33cfc0b5a50",
    "qqp": "8e711799-a57c-4941-833b-466bedfb80ad",
    "stsb": "ca75788d-4974-440a-a7b7-c42bae814d59",
    "qnli": "c626350d-6c0e-47be-b09e-c9ba1446b027",
    "rte": "4ee6ff27-de63-4e7b-a9d4-82a17eba407a",
    "wnli": "10c354ee-6f4e-4b04-91e1-29e999a8f3e7",
}

asr_expdir = sys.argv[2]
asr_train_config = f"{asr_expdir}/config.yaml"
asr_model_file = f"{asr_expdir}/0epoch.pth"
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
data = open(
    f"/mount/arbeitsdaten/asr-3/denisopl/speechGLUE/dump/{glue_task}/validation/data.csv"
)

for l in csv.DictReader(data):
    speechglue_data[l["idx"]] = l

data.close()

prompts = DatasetTemplates(f"glue/{glue_task}")
template = prompts.templates[task2template[glue_task]]

answer_options = set()

for e in dataset:
    example = template.apply(e)
    speechglue_data[str(e["idx"])]["label"] = example[1]
    answer_options.add(example[1])

print(f"Answer options: {answer_options}")
hf_tokenizer = AutoTokenizer.from_pretrained(bpemodel)
force_words_ids = hf_tokenizer.batch_encode_plus(
    list(answer_options), return_attention_mask=False
)["input_ids"]
max_answer_length = max([len(x) for x in force_words_ids])

refs = []
hyps = []

for i, l in enumerate(speechglue_data.values()):
    if i % 10 == 0:
        print(f"Processing {glue_task} sample {i}")

    refs.append(l["label"])

    speech = []
    text_fields = task2text[glue_task]

    for text_field in text_fields:
        wav_file = (
            l[f"file_{text_field}"]
            .replace(".wav", ".flac")
            .replace("arbeitsdaten45/projekte/asr-4", "arbeitsdaten/asr-3")
        )

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

    for text in task2list[glue_task]:
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

json.dump({"refs": refs, "hyps": hyps}, open(f"outputs_speechglue/multiprompt-synth/speechglue_{glue_task}.json", "w"))

print("Done")
