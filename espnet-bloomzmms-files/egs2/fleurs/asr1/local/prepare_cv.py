import glob
import os
import subprocess
import sys

utts = set()

with open("downloads/cv12_train_utts.lst") as f:
    utts = set([x.strip() for x in f.readlines()])

idir = sys.argv[1]
odir = "data/train_cv"
os.makedirs(odir, exist_ok=True)

with open(f"{odir}/wav.scp", "w") as wav_scp, open(
    f"{odir}/text", "w", encoding="utf-8"
) as text_scp, open(f"{odir}/utt2spk", "w") as utt2spk:

    for langdir in glob.glob(f"{idir}/*"):
        with open(f"{langdir}/train.tsv", encoding="utf-8") as meta:
            for line in meta:
                if line[:6] == "client":
                    continue
                parts = line.strip().split("\t")
                utt = {"wav": parts[1], "text": parts[2]}
                uttid = utt["wav"][:-4]

                if uttid in utts:
                    wav_scp.write(
                        f"{uttid} ffmpeg -i {langdir}/clips/{utt['wav']} -f wav -ar 16000 -ab 16 -ac 1 - |\n"
                    )
                    text_scp.write(f"{uttid} {utt['text']}\n")
                    utt2spk.write(f"{uttid} {uttid}\n")


subprocess.call(f"utils/fix_data_dir.sh {odir}", shell=True)
