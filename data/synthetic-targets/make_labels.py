import json
import random
import sys
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

d = int(sys.argv[1])

print("Loading")

checkpoint = "bigscience/bloomz-7b1"
model = AutoModelForCausalLM.from_pretrained(
    checkpoint, device_map="auto", load_in_8bit=True
)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

templates = []

with open("templates.json") as f:
    templates = json.load(f)

inputs = {}

with open(f"inputs{d:02d}.txt") as f:
    for l in f:
        text = l.strip()
        if text in inputs:
            inputs[text] += 1
        else:
            inputs[text] = 1

prompts = []
u = 2

print("Preparing")

with open(f"text{d:02d}.txt", "w") as t, open(f"prefix{d:02d}.txt", "w") as p, open(
    f"postfix{d:02d}.txt", "w"
) as s:
    for text in inputs:
        k = inputs[text] * u
        text_templates = random.choices(templates, k=k)
        prompts.extend([t[0] + text + t[1] for t in text_templates])
        t.writelines([text + "\n" for _ in range(k)])
        p.writelines([json.dumps(x[0]) + "\n" for x in text_templates])
        s.writelines([json.dumps(x[1]) + "\n" for x in text_templates])

batch = 16

print("Generating")

with open(f"output{d:02d}.txt", "w") as o:
    for i in tqdm(range(0, len(prompts), batch)):
        prompt_inputs = tokenizer(
            prompts[i : i + batch], return_tensors="pt", padding=True
        ).to("cuda")

        with torch.no_grad():
            prompt_outputs = model.generate(
                **prompt_inputs,
                num_beams=1,
                max_new_tokens=128,
            )

        text_outputs = [
            x
            for x in tokenizer.batch_decode(
                prompt_outputs[:, prompt_inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )
        ]

        o.writelines([json.dumps(output) + "\n" for output in text_outputs])

print("DONE")
