#!/usr/bin/env python3

import glob
import os
import shutil
import subprocess
import sys

from datasets import load_dataset

src_lang = sys.argv[1]
tgt_lang = sys.argv[2]

idir = "dump/raw/test_fleurs"
odir = f"dump/raw/test_fleurs_{src_lang}-{tgt_lang}"
os.makedirs(odir, exist_ok=True)

src_lang = src_lang.replace("-", "_")
tgt_lang = tgt_lang.replace("-", "_")

wavid2wav = {}

with open(f"{idir}/wav.scp") as wfile:
    for l in wfile:
        utt, wav = l.strip().split()
        lang, _, _, wavid = utt.split("-")
        if lang == src_lang:
            wavid2wav[wavid] = wav

shutil.copy2(f"{idir}/feats_type", f"{odir}/feats_type")

src_dset = load_dataset("google/fleurs", src_lang, split="test", trust_remote_code=True)
id2wavid = {}

for example in src_dset:
    id2wavid[str(example["id"])] = example["path"].split("/")[-1][:-4]

tgt_dset = load_dataset("google/fleurs", tgt_lang, split="test", trust_remote_code=True)
id2text = {}

for example in tgt_dset:
    id2text[str(example["id"])] = example["raw_transcription"]

with open(f"{odir}/text", "w", encoding="utf-8") as tfile, open(
    f"{odir}/utt2spk", "w"
) as sfile, open(f"{odir}/wav.scp", "w") as wfile:
    for textid in id2text.keys():
        if textid in id2wavid:
            wavid = id2wavid[textid]
            utt = f"{src_lang}-audio-test-{wavid}"
            tfile.write(f"{utt} {id2text[textid]}\n")
            sfile.write(f"{utt} spk\n")
            wfile.write(f"{utt} {wavid2wav[wavid]}\n")

subprocess.call(f"utils/fix_data_dir.sh {odir}", shell=True)
