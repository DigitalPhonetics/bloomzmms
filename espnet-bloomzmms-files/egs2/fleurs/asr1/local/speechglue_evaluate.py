#!/usr/bin/env python3

import json
import evaluate
import sys

from promptsource.templates import DatasetTemplates

idir = sys.argv[1]

task2template = {
    "cola": "39a701ff-bb4b-48ac-8c0a-8c61bf0d4b8d",
    "sst2": "63c6b2be-8ecd-42ad-88c7-0d1dc1a8323a",
    "mrpc": "adf659af-4e2d-4e7e-ab89-b33cfc0b5a50",
    "qqp": "8e711799-a57c-4941-833b-466bedfb80ad",
    "mnli_matched": "f3ebe1ac-194b-41e7-b008-36eafdbfbe25",
    "mnli_mismatched": "770aa883-efec-4258-9e1e-1a96d0c20ed5",
    "qnli": "c626350d-6c0e-47be-b09e-c9ba1446b027",
    "rte": "4ee6ff27-de63-4e7b-a9d4-82a17eba407a",
    "wnli": "10c354ee-6f4e-4b04-91e1-29e999a8f3e7",
}

task2metrics = {
    "mnli_matched": ("accuracy"),
    "mnli_mismatched": ("accuracy"),
    "cola": ("matthews_correlation"),
    "mrpc": ("accuracy"),
    "qnli": ("accuracy"),
    "qqp": ("accuracy"),
    "rte": ("accuracy"),
    "sst2": ("accuracy"),
    "wnli": ("accuracy"),
}

all_results = []

for glue_task in task2template.keys():
    results = json.load(open(f"{idir}/speechglue_{glue_task}.json"))

    if glue_task == "stsb":
        refs = [float(x) for x in results["refs"]]
        hyps = [float(x) for x in results["hyps"]]
    else:
        prompts = DatasetTemplates(f"glue/{glue_task}")
        choises = [
            x.lower()
            for x in prompts.templates[
                task2template[glue_task]
            ].get_fixed_answer_choices_list()
        ]
        refs = [choises.index(x.lower()) for x in results["refs"]]
        hyps = [
            choises.index(x.lower()) if x.lower() in choises else 0
            for x in results["hyps"]
        ]

    metric = evaluate.load("glue", glue_task)

    results = metric.compute(predictions=hyps, references=refs)

    k = task2metrics[glue_task]
    print(f"{glue_task}: {k}={results[k]*100.0:.01f}")
    if k != "wnli":
        all_results.append(results[k])

print(f"Avg.: {sum(all_results)/len(all_results)*100.0:.01f}")
