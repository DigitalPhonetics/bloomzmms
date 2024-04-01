import os
import subprocess

try:
    from datasets import load_dataset
except Exception:
    print("Error importing datasets library")
    print("datasets can be installed via espnet/tools/installers/install_datasets")
    exit()

common_voice_split_map = {"train": "train", "validation": "dev", "test": "test"}

"""
Use the fleurs portion of "google/xtreme_s" instead of "google/fleurs".
google/fleurs data does not include the path to the downloaded audio clips
"""
fleurs_asr = load_dataset(
    "google/xtreme_s",
    "fleurs.all",
)
lang_iso_map = fleurs_asr["train"].features["lang_id"].names


def create_data(split):
    paths = fleurs_asr[split]["path"]
    raw_transcriptions = fleurs_asr[split]["raw_transcription"]
    ids = fleurs_asr[split]["id"]
    lang_ids = fleurs_asr[split]["lang_id"]

    subset = common_voice_split_map[split]

    odir = f"data/{subset}_fleurs"
    os.makedirs(odir, exist_ok=True)

    with open(odir + "/text", "w", encoding="utf-8") as text, open(
        odir + "/wav.scp", "w"
    ) as wavscp, open(odir + "/utt2spk", "w") as utt2spk:

        for i, _ in enumerate(paths):
            if ids[i] == 10:
                continue
            wav = paths[i]
            words = raw_transcriptions[i]
            lang_iso = lang_iso_map[lang_ids[i]]
            uttid = lang_iso + "-" + wav[:-4].split("/")[-1]

            text.write(f"{uttid} {words}\n")
            utt2spk.write(f"{uttid} {uttid}\n")
            wavscp.write(
                f"{uttid} sox --norm=-1 {wav} -r 16k -t wav -c 1 -b 16 -e signed - |\n"
            )

    subprocess.call(f"utils/fix_data_dir.sh {odir}", shell=True)


create_data("train")
create_data("validation")
create_data("test")
