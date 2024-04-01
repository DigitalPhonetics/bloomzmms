#!/usr/bin/env python3

from datasets import load_dataset

langs = [
    "ar",
    "bg",
    "de",
    "el",
    "en",
    "es",
    "fr",
    "hi",
    "ru",
    "sw",
    "th",
    "tr",
    "ur",
    "vi",
    "zh",
]

for lang in langs:
    dataset = load_dataset("xnli", lang, split="validation")
    sentences = set(sum([[x["premise"], x["hypothesis"]] for x in dataset], []))

    with open(f"sentences_{lang}.txt", "w", encoding="utf-8") as f:
        f.writelines([f"{l}\n" for l in list(sentences)])
