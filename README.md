# 🔥 Hateful Meme Classification

Hateful memes are a subtle yet harmful form of online hate speech. Unlike traditional text or image content, memes often convey hateful meaning through the **combination of text and visuals**, making them especially difficult for traditional moderation systems to detect.

This project aims to build robust models for **hateful meme classification**, leveraging **image-based**, **text-based**, and **multimodal CLIP-based** approaches. Our goal is to **maximize hateful content detection** while keeping **false positives low** on non-hateful memes.


> This project explores the classification of hateful memes using three different approaches:
>
> - **Image-based classifier (CA_CNN)**  
> - **Text-based classifier (HateBERT)**  
> - **Multimodal CLIP-based classifiers** with three fusion strategies:  
>   - Concatenation  
>   - Ensemble  
>   - Self-attention  
>
> A deep neural network classifier is trained on each fusion output. The primary objective is to **maximize the detection of hateful memes** while **minimizing false positives** on benign content.

---

## 📊 Dataset

We use the dataset from **Phase I of the Facebook Hateful Memes Challenge**, structured as follows:

| Split          | Total Samples | Hateful | Non-Hateful |
|----------------|----------------|---------|-------------|
| Training Set   | 8,500          | 35%     | 65%         |
| Validation Set | 500            | 50%     | 50%         |
| Test Set       | 1,000          | 50%     | 50%         |

Each meme consists of an image and a caption, labeled as **hateful** or **non-hateful**.

---

## 📈 Results

Here’s a performance comparison across our different models:

| Model           | Accuracy | AUC-ROC | Precision | Recall | F1 Score |
|-----------------|----------|---------|-----------|--------|----------|
| **HateBERT**        | 0.621    | 0.675   | 0.599     | 0.686  | 0.639    |
| **CA_CNN**          | 0.517    | 0.531   | 0.506     | 0.567  | 0.535    |
| **ConcatCLIP**      | **0.726**| **0.809**| 0.690     | 0.800  | **0.741**|
| **EnsembleCLIP**    | 0.707    | 0.766   | **0.700** | 0.700  | 0.700    |
| **AttentionCLIP**   | 0.646    | 0.737   | 0.589     | **0.916** | 0.717 |


---


