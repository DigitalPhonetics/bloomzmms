import json
import sys

print("Loading")

inputs = {}

with open(f"inputs.txt") as f:
    for l in f:
        text = l.strip()
        if text in inputs:
            inputs[text] += 1
        else:
            inputs[text] = 1

prompts = []

print("Preparing")

d = 61

with open(f"text{d:02d}.txt", "w") as t, open(f"prefix{d:02d}.txt", "w") as p, open(
    f"postfix{d:02d}.txt", "w"
) as s, open(f"output{d:02d}.txt", "w") as o:
    for text in inputs:
        k = inputs[text]
        text_templates = [("Repeat the sentence: ", ". ") for _ in range(k)]
        t.writelines([text + "\n" for _ in range(k)])
        p.writelines([json.dumps(x[0]) + "\n" for x in text_templates])
        s.writelines([json.dumps(x[1]) + "\n" for x in text_templates])
        o.writelines([json.dumps(text) + "\n" for _ in range(k)])

print("DONE")
