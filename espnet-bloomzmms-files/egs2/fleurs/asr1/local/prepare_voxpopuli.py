#!/usr/bin/env python3

import csv
import os
import subprocess
import sys

idir = sys.argv[1]
odir = sys.argv[2]

os.makedirs(odir, exist_ok=True)

with open(f"{idir}/asr_test.tsv") as f, open(f"{odir}/text", "w") as text, open(
    f"{odir}/wav.scp", "w"
) as wavscp, open(f"{odir}/utt2spk", "w") as utt2spk:
    tsv_file = csv.DictReader(f, delimiter="\t")

    for l in tsv_file:
        utt = l["speaker_id"] + "-" + l["id"]
        year = l["id"][:4]

        if l["raw_text"] == "":
            transcription = l["normalized_text"]
        else:
            transcription = l["raw_text"]

        if transcription == "":
            continue

        wavscp.write(
            f"{utt} ffmpeg -i \"{idir}/{year}/{l['id']}.ogg\" "
            + "-f wav -ar 16000 -ab 16 -ac 1 - |\n"
        )
        text.write(f"{utt} {transcription}\n")
        utt2spk.write(f"{utt} {l['speaker_id']}\n")

subprocess.call(f"utils/fix_data_dir.sh {odir}", shell=True)
