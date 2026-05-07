# Sens-VisualNews

## Introduction

This repository contains the dataset annotations and evaluation code for our sensational image detection dataset, called Sens-VisualNews. In this repository, we release the JSON dataset annotation files. The source images can be retrieved from the official VisualNews repository, as explained in the sections below.

## Paper Abstract

The detection of sensational content in media items can be a critical filtering mechanism for identifying check-worthy content and flagging potential disinformation, since such content triggers physiological arousal that often bypasses critical evaluation and accelerates viral sharing. In this paper we introduce the task of sensational image detection, which aims to determine whether an image contains shocking, provocative, or emotionally charged features to grab attention and trigger strong emotional responses. To support research on this task, we create a new benchmark dataset (called Sens-VisualNews) that contains 9,576 images from news items, annotated based on the (in-)existence of various sensational concepts and events in their visual content. Finally, using Sens-VisualNews, we study the prompt sensitivity, performance and robustness of a wide range of open SotA Multimodal LLMs, across both zero-shot and fine-tuned settings.

## Instructions

Please download the original source images from the official VisualNews website: https://www.cs.rice.edu/~vo9/visualnews/

Folder structure of Sens-VisualNews:
```
dataset/
- full_test.json    # test annotations
- full_dev.json     # development annotations
- strict_test.json  # test annotations for the strict subset
- strict_dev.json   # development annotations for the strict subset
```

Each sample contains the following fields:
```
{
    "image": path to the image in the VisualNews dataset,
    "gt": true if the image is sensational, otherwise false
}
```

## Requirements

- PyTorch (torch)
- FlashAttention2 (flash-attn)
- transformers
- sklearn
- peft

## Evaluation

The following script can be used to evaluate a Multimodal LLM on the Sens-VisualNews dataset. For example, to reproduce the evaluation results for Qwen3-VL 2B, run the following command:
```
python evaluate.py \
    --data_dir PATH_TO_VISUAL_NEWS \
    --dataset dataset/full_test.json \
    --model_id Qwen/Qwen3-VL-2B-Instruct \
    --format pre \
    --prompt "Does this image trigger strong emotional responses (e.g. fear, anger, anxiety, disgust, shock)? Answer with a single yes or no."
```

Fine-tuned models with PEFT can be evaluted by specifying the `--peft_ckp` argument. Note that, depending on the model and fine-tuning setup, the optimal model prompt may differ. A custom prompt can be specified via the `--prompt` argument. `--format` controls whether the visual tokens are appended or pre-prended in the prompt (either "pre" or "post"). Please consult our paper for more details on the experimental setup.

## Fine-tuning with LoRA

Run the following command to fine-tune Qwen3-VL 2B on the Sens-VisualNews development set using LoRA:
```
torchrun sft.py \
    --data_dir PATH_TO_VISUAL_NEWS \
    --train_dataset dataset/full_dev.json \
    --pretrained Qwen/Qwen3-VL-2B-Instruct \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --num_train_epochs 30 \
    --gradient_checkpointing true \
    --learning_rate 5e-4 \
    --lr_scheduler_type cosine \
    --warmup_steps 0.1 \
    --logging_steps 1 \
    --bf16 \
    --remove_unused_columns false \
    --dataloader_num_workers 4
```

To evaluate the final PEFT checkpoint, run the following command:
```
python evaluate.py \
    --data_dir PATH_TO_VISUAL_NEWS \
    --dataset dataset/full_test.json \
    --model_id Qwen/Qwen3-VL-2B-Instruct \
    --peft_ckp trainer_output/checkpoint-450
```

## Citation

If you find our work or code useful, please cite our publication:

A. Goulas, D. Galanopoulos, E. Apostolidis, V. Mezaris, "Sens-VisualNews: A Benchmark Dataset for Sensational Image Detection", IEEE Int. Conf. on Image Processing (ICIP 2026), Tampere, Finland, Sept. 2026.

```
@inproceedings{goulas2026SensVisualNews,
  title={Sens-VisualNews: A Benchmark Dataset for Sensational Image Detection},
  author={Goulas, Andreas and Galanopoulos, Damianos and Apostolidis, Evlampios and Mezaris, Vasileios},
  booktitle={IEEE Int. Conf. on Image Processing (ICIP 2026)},
  year={2026},
  organization={IEEE}
}
```

Link to preprint: coming soon

## License

This code is provided for academic, non-commercial use only. Please also check for any restrictions applied in the code parts and datasets used here from other sources. For the materials not covered by any such restrictions, redistribution and use in source and binary forms, with or without modification, are permitted for academic non-commercial use provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation provided with the distribution.

This software is provided by the authors "as is" and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed. In no event shall the authors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this software, even if advised of the possibility of such damage.

## Acknowledgement

This work was supported by the EU’s Horizon Europe programme under grant agreement 101070190 AI4Trust.


