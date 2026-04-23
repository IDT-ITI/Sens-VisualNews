# Andreas Goulas <agoulas@iti.gr>

import json
import os
import torch
from transformers import (
    TrainingArguments,
    Trainer,
    HfArgumentParser,
    AutoModelForImageTextToText,
    AutoProcessor,
)
from peft import get_peft_model, LoraConfig

PROMPT = "Is this image sensational? A sensational image evokes strong emotions (e.g. fear, anger, anxiety, disgust, shock). Answer with a single yes or no."

class SensVisualNews(torch.utils.data.Dataset):
    def __init__(self, path, prompt, data_dir):
        self.prompt = prompt
        self.data_dir = data_dir

        with open(path, "r") as f:
            self.rows = json.load(f)

    def __len__(self):
        return len(self.rows)
    
    def __getitem__(self, idx):
        row = self.rows[idx]
        label = "Yes" if row["gt"] else "No"
        path = os.path.join(self.data_dir, row["image"])

        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "path": path},
                    {"type": "text", "text": self.prompt},
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": label}
                ]
            }
        ]

        return {
            "msgs": msgs
        }
    
class MyCollate:
    def __init__(self, processor):
        self.processor = processor

        self.assistant_start_ids = torch.tensor(self.processor.tokenizer.encode(
            "<|im_start|>assistant\n",
            add_special_tokens=False
        ))

        self.assistant_start_len = len(self.assistant_start_ids)

    def __call__(self, inputs):
        batch = self.processor.apply_chat_template(
            [x["msgs"] for x in inputs],
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            padding=True
        )

        labels = batch.input_ids.clone()

        for i in range(labels.size(0)):
            idx = 0
            for t in range(labels.size(1) - self.assistant_start_len + 1):
                if torch.equal(
                    batch.input_ids[i, t:t+self.assistant_start_len],
                    self.assistant_start_ids
                ):
                    idx = t + self.assistant_start_len

            labels[i, :idx] = -100

        batch["labels"] = labels

        return batch

parser = HfArgumentParser((TrainingArguments,))
parser.add_argument("--train_dataset", type=str, default="dataset/full_val.json")
parser.add_argument("--data_dir", type=str)
parser.add_argument("--pretrained", type=str)
parser.add_argument("--lora_r", type=int, default=16)
parser.add_argument("--lora_alpha", type=int, default=32)
train_args, args = parser.parse_args_into_dataclasses()

model = AutoModelForImageTextToText.from_pretrained(
    args.pretrained,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

processor = AutoProcessor.from_pretrained(
    args.pretrained,
    padding_side="left"
)

peft_config = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "down_proj",
        "gate_proj"
    ]
)

model = get_peft_model(model, peft_config)

train_dataset = SensVisualNews(args.train_dataset, PROMPT, args.data_dir)

trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    data_collator=MyCollate(processor),
    processing_class=None,
    args=train_args
)

trainer.train()
