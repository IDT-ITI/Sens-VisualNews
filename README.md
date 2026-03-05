# Sens-VisualNews

## Introduction

This repository contains the dataset annotations for our sensational image detection dataset, called Sens-VisualNews. In this repository, we only release the annotation files. Please download the source images from the VisualNews repository, as explained in the Instructions section below.

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

## Citation

If you find our work useful, please cite:

```
TODO
```

## License

## Acknowledgement

This work was supported by the EU’s Horizon Europe programme under grant agreement 101070190 AI4Trust.


