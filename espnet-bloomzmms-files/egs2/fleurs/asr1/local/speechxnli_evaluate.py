#!/usr/bin/env python3

import json
import evaluate
import sys

from promptsource.templates import DatasetTemplates

idir = sys.argv[1]

prompts = DatasetTemplates("xnli/en")

choises = [
    x.lower()
    for x in prompts.templates[
        "172b73dc-d045-491c-9dc2-76bf6566c8ee"
    ].get_fixed_answer_choices_list()
]

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
    results = json.load(open(f"{idir}/xnli_{lang}.json"))

    refs = [x for x in results["refs"]]
    hyps = [
        choises.index(x.lower()) if x.lower() in choises else 0 for x in results["hyps"]
    ]

    metric = evaluate.load("xnli")
    results = metric.compute(predictions=hyps, references=refs)
    print(f"{lang} {results['accuracy']*100.0:.02f}")
