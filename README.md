# Multimodal Sentiment Analysis on Telugu Memes 

This repository contains a **multimodal sentiment analysis** project built on **Telugu memes**, where sentiment is predicted using **both meme text + meme image**.  
The core idea is simple: extract features from the **image**, extract features from the **text**, train models, and then combine (fuse) their predictions to get a final sentiment.

---

## What’s inside this repo?

### 1) `Memes (1).xlsx`
A dataset file with **2000 meme samples** and 3 columns:
- `Meme_ID` → image id (example: `1`, `2`, `3`…)
- `Meme_Text` → Telugu meme text/dialogue
- `Label` → sentiment class (**Positive / Negative / Neutral**)  
> Note: Some labels may have small inconsistencies like spacing/case/typos (e.g., `neutral`, `Neutral `, `Nuetral`). Normalizing labels before training is recommended.

### 2) `Multimodal_Memes (2).ipynb`
A notebook (Colab-style) that:
- Loads the Excel dataset
- Loads meme images using `Meme_ID` (supports `.jpg` and `.jpeg`)
- Builds:
  - **Image model** using **EfficientNetB0 (ImageNet pretrained)**
  - **Text model** using tokenization + padding (simple Dense baseline)
- Trains both models separately
- Applies **late fusion** (averaging predictions from image + text models)
- Prints a **classification report**

---

## Model Approach (High Level)

### ✅ Image Branch
- Uses **EfficientNetB0** (pretrained on ImageNet)
- Removes top layer (`include_top=False`)
- Adds `Flatten + Dense + Softmax` for classification

### ✅ Text Branch
- Uses Keras `Tokenizer` + `pad_sequences`
- A lightweight Dense-based classifier (baseline)

### ✅ Late Fusion
Final prediction = average of predictions:
```python
final_preds = (image_preds + text_preds) / 2
