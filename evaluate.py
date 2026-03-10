import torch
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
)
from peft import PeftModel
from sklearn.metrics import accuracy_score
import argparse
import random
import json
from tqdm import tqdm
import os

DEFAULT_PROMPT = "Is this image sensational? A sensational image evokes strong emotions (e.g. fear, anger, anxiety, disgust, shock). Answer with a single yes or no."

def parse_response(resp):
    if "yes" in resp.lower():
        return True
    elif "no" in resp.lower():
        return False
    else:
        print("invalid llm response!")
        return random.random() < 0.5

def evaluate_row(model, processor, prompt, format, image):
    if format == "pre":
        content = [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt}
        ]
    else:
        content = [
            {"type": "text", "text": prompt},
            {"type": "image", "image": image}
        ]
        
    msgs = [
        {
            "role": "user",
            "content": content,
        }
    ]

    batch = processor.apply_chat_template(
        msgs,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda")
    
    out_ids = model.generate(
        **batch,
        max_new_tokens=16,
        do_sample=False,
    )[0]

    out_ids = out_ids[len(batch.input_ids[0]):]
    text = processor.decode(out_ids, skip_special_tokens=True)

    return parse_response(text)

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--dataset", type=str, default="full_test.json")
parser.add_argument("--peft_ckp", type=str)
parser.add_argument("--model_id", type=str, required=True)
parser.add_argument("--prompt", type=str)
parser.add_argument("--format", type=str, choices=["pre", "post"], default="pre")
parser.add_argument("--trust_remote_code", action="store_true")
args = parser.parse_args()

model = AutoModelForImageTextToText.from_pretrained(
    args.model_id,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    trust_remote_code=args.trust_remote_code
)

if args.peft_ckp is not None:
    model = PeftModel.from_pretrained(model, args.peft_ckp)
    print("peft enabled")

model.cuda()

processor = AutoProcessor.from_pretrained(args.model_id)

prompt = args.prompt
if prompt is None:
    prompt = DEFAULT_PROMPT

with open(args.dataset, "r") as f:
    dataset = json.load(f)

y_true = []
y_pred = []
for ex in tqdm(dataset):
    gt = ex["gt"]
    image = os.path.join(args.data_dir, ex["image"])

    pred = evaluate_row(model, processor, prompt, args.format, image)

    y_true.append(gt)
    y_pred.append(pred)

top1 = 100 * accuracy_score(y_true=y_true, y_pred=y_pred)
print("Accuracy = {}".format(round(top1, 1)))
