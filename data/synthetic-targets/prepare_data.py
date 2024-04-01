import json
from transformers import AutoTokenizer

idir = "/mount/arbeitsdaten/asr-3/denisopl/fleurs/dump/raw/train_fl_cv_raw_sp"
odir = "/mount/arbeitsdaten/asr-3/denisopl/fleurs/dump/raw/train_fl_cv_raw_prompts_sp"

parts = [0]

parts = [x for x in range(62)]

checkpoint = "bigscience/bloomz-7b1"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

text2meta = {}

print("Reading outputs")

for d in parts:
    with open(f"text{d:02d}.txt") as ft, open(f"prefix{d:02d}.txt") as fp, open(
        f"postfix{d:02d}.txt"
    ) as fs, open(f"output{d:02d}.txt") as fo:
        for lt, lp, ls, lo in zip(ft, fp, fs, fo):
            text = lt.strip()

            if text not in text2meta:
                text2meta[text] = []

            meta = {
                "prefix": " ".join(
                    [str(x) for x in tokenizer(json.loads(lp.strip()))["input_ids"]]
                ),
                "postfix": " ".join(
                    [str(x) for x in tokenizer(json.loads(ls.strip()))["input_ids"]]
                ),
                "output": " ".join(
                    [str(x) for x in tokenizer(json.loads(lo.strip()))["input_ids"]]
                ),
            }

            if len(meta["output"]) > 0 and (
                len(meta["prefix"]) > 0 or len(meta["postfix"]) > 0
            ):
                text2meta[text].append(meta)

utt2wav = {}
utt2text = {}
utt2spk = {}
utt2count = {}

print("Reading data")

with open(f"{idir}/wav.scp") as fw, open(f"{idir}/text") as ft, open(
    f"{idir}/utt2spk"
) as fs:
    for lw, lt, ls in zip(fw, ft, fs):
        utt, wav = lw.strip().split(" ", maxsplit=1)
        utt2wav[utt] = wav

        utt, text = lt.strip().split(" ", maxsplit=1)
        utt2text[utt] = text.strip()

        utt, spk = ls.strip().split(" ", maxsplit=1)
        utt2spk[utt] = spk

        utt2count[utt] = 0

print("Writing data")

with open(f"{odir}/wav.scp", "w") as fw, open(f"{odir}/text", "w") as ft, open(
    f"{odir}/utt2spk", "w"
) as fs, open(f"{odir}/decoder_prefix", "w") as fpr, open(
    f"{odir}/decoder_postfix", "w"
) as fpo:
    while len(text2meta) > 0:
        print(len(text2meta))
        for utt in utt2wav:
            text = utt2text[utt]

            if text in text2meta:
                meta = text2meta[text].pop()

                if len(text2meta[text]) == 0:
                    del text2meta[text]

                utt2count[utt] += 1
                new_utt = f"{utt}-{utt2count[utt]}"

                fw.write(f"{new_utt} {utt2wav[utt]}\n")
                ft.write(f"{new_utt} {meta['output']}\n")
                fs.write(f"{new_utt} {utt2spk[utt]}\n")
                fpr.write(f"{new_utt} {meta['prefix']}\n")
                fpo.write(f"{new_utt} {meta['postfix']}\n")

print("Done")
