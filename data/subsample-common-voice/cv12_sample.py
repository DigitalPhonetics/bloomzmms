#!/usr/bin/python3

import random

random.seed(1)

lang2dur = {}
wav2dur = {}

# Download cv12_durations.tsv from https://zenodo.org/records/10900287/files/cv12_durations.tsv.gz?download=1

with open("cv12_durations.tsv") as f:
    for l in f:
        wav, dur = l.strip().split("\t")
        dur = float(dur)

        wav2dur[wav] = dur

        lang = wav.split("_")[2]

        if lang in lang2dur:
            lang2dur[lang] += dur
        else:
            lang2dur[lang] = dur

max_secs = 25 * 3600.0

idir = "/path/to/cv-corpus-12.0-2022-12-07"

for lang in lang2dur:
    text2utt = {}

    with open(f"{idir}/{lang}/train.tsv", encoding="utf-8") as meta:
        for line in meta:
            if line[:6] == "client":
                continue
            parts = line.strip().split("\t")
            utt = {"spk": parts[0], "wav": parts[1], "text": parts[2]}

            if utt["text"] in text2utt:
                text2utt[utt["text"]].append(utt)
            else:
                text2utt[utt["text"]] = [utt]

    texts = [x for x in text2utt.keys()]
    random.shuffle(texts)

    for text in texts:
        random.shuffle(text2utt[text])

    total_secs = 0.0

    while True:
        for text in texts:
            if len(text2utt[text]) > 0:
                utt = text2utt[text].pop(0)
                print(utt["wav"][:-4])
                total_secs += wav2dur[utt["wav"]]

                if total_secs > max_secs:
                    break

        if total_secs > max_secs or sum([len(x) for x in text2utt.values()]) == 0:
            break
