Get the data from the google drive. You need to update this based on where your data is in the google drive


```python
import os
DATA_DRIVE = 'data'
DRIVE_PATH = os.path.join('/storage/ice1/6/1/ssubedi33', DATA_DRIVE)
print(os.listdir(DRIVE_PATH))
```

    ['.idea', 'LICENSE.txt', 'test_seen.jsonl', 'img', 'test.jsonl', 'README.md', 'dev_seen.jsonl', 'dev.jsonl', '.DS_Store', 'train.jsonl']


Import torch and reload external modules


```python
# Just run this block. Please do not modify the following code.
import pandas as pd
import torch

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
%load_ext autoreload
%autoreload 2
```

Now lets check your GPU availability and load some sanity checkers. By default you should be using your gpu for this assignment if you have one available.


```python
# Check device availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("You are using device: %s" % device)
```

    You are using device: cuda


Load data here. train.jsonl and dev.jsonl file contains id, image location, label and caption in the image.


```python
path = DRIVE_PATH + '/train.jsonl'
train_data=pd.read_json(path,lines=True)
print(f"Train set size: {len(train_data)}")
```

    Train set size: 8500



```python
path = DRIVE_PATH + '/dev_seen.jsonl'
val_data=pd.read_json(path,lines=True)
print(f"Val set size: {len(val_data)}")
```

    Val set size: 500



```python
path = DRIVE_PATH + '/test_seen.jsonl'
test_data=pd.read_json(path,lines=True)
print(f"Val set size: {len(test_data)}")
```

    Val set size: 1000


Split train data in to train, validation set. dev data will be test set.


```python
from sklearn.model_selection import train_test_split


print(f"Train set size: {len(train_data)}")
print(f"Validation set size: {len(val_data)}")
print(f"Test set size: {len(test_data)}")
print(test_data.head())

```

    Train set size: 8500
    Validation set size: 500
    Test set size: 1000
          id            img  label                                         text
    0  16395  img/16395.png      1                     handjobs sold seperately
    1  37405  img/37405.png      1         introducing fidget spinner for women
    2  94180  img/94180.png      1  happy pride month let's go beat up lesbians
    3  54321  img/54321.png      1       laughs in [majority of u.s crime rate]
    4  97015  img/97015.png      1       finds out those 72 virgins.. are goats


Fine Tuning CLIP model.
This part of the code imports the pretrained model and processor from openAI.
model contains weights and actual architecture.
processor contains tokenizer for words and feature extractor for image. These are used to convert image and caption into numbers that computer will understand.



```python
from transformers import CLIPModel, CLIPProcessor

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
```

    Using a slow image processor as `use_fast` is unset and a slow processor was saved with this model. `use_fast=True` will be the default behavior in v4.52, even if the model was saved with a slow processor. This will result in minor differences in outputs. You'll still be able to use a slow processor with `use_fast=False`.


This is a class to tokenize dataset. It takes an input and output tokens for text and image. It uses the same tokenizer that CLIP uses.


```python
from torch.utils.data import Dataset
from PIL import Image
import torch

class MemeDataset(Dataset):
    def __init__(self, dataframe, processor, image_root_dir, max_length = 70):
        self.df = dataframe.reset_index(drop=True)
        self.processor = processor
        #directory where all images are
        self.image_root = image_root_dir
        #to make sure all captions are of same length
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        #get the data at idx
        row = self.df.loc[idx]
        #extract path to image
        img_path = f"{self.image_root}/{row['img'].split('/')[-1]}"
        #load image in RGB
        image = Image.open(img_path).convert("RGB")
        #load text
        text = row['text']
        #load label
        label = torch.tensor(row['label'], dtype=torch.float)

        # Convert text and image to tokens
        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding='max_length',
            max_length=self.max_length,
            truncation=True)

        # Remove batch dimension (1) from processor outputs
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        #add label
        inputs["labels"] = label
        return inputs

```

Implement MemeDataset class to the train data, validation data, and test data.


```python
from torch.utils.data import DataLoader
import torch

BATCH_SIZE = 32
train_dataset = MemeDataset(train_data, processor, image_root_dir = DRIVE_PATH + "/img")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataset = MemeDataset(val_data, processor, image_root_dir = DRIVE_PATH + "/img")
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_dataset = MemeDataset(test_data, processor, image_root_dir = DRIVE_PATH + "/img")
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

```

Binary classification head at the end of CLIP model. It uses CLIP embeddings from both text and image inputs, and combines them to make a prediction.



```python
import torch.nn as nn

class CLIPBinaryClassifier(nn.Module):
    def __init__(self, clip_model, dropout_prob):
        super().__init__()
        #parent model - pretrained openAI CLIP model
        self.clip = clip_model
        #get image and text embeddings
        dim = self.clip.config.projection_dim
        #Linear classification head with two outputs - Hateful or Not Hateful
        self.classifier = nn.Sequential(
            nn.Linear(dim * 2, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.LayerNorm(dim * 4),
            nn.Linear(dim * 4, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.LayerNorm(dim*2),
            nn.GELU(),
            nn.Linear(dim * 2, 1))
        #self.classifier = nn.Sequential(nn.Linear(dim*2, 2))
        
        self._init_weights()
    #Initialize weights 
    def _init_weights(self):
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask, pixel_values):
        #forward pass on tokenized inputs, outputs image and text embeddings
        outputs = self.clip(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
        combined = torch.cat([outputs.image_embeds, outputs.text_embeds], dim=1)
        logits = self.classifier(combined)
        return logits.squeeze(-1)
```

Training Loop


```python
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score 
from tqdm import tqdm  # For progress bars

# Initialize classifier, takes the CLIP model and dropout probability as input
classifier = CLIPBinaryClassifier(model, dropout_prob=0.2).to(device)

# Freeze CLIP backbone, to make the model faster
for param in classifier.clip.parameters():
    param.requires_grad = False

# Optimizer - now includes all trainable parameters (classifiers + alpha)
learning_rate = 0.0001
#reduce the learning rate for alpha
optimizer = torch.optim.AdamW([{'params': classifier.classifier.parameters()}], lr=learning_rate)

# Cross entropy Loss and epochs
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([65/35], device=device))
epochs = 15

# Track losses, auc scores and alpha
train_losses = []
val_losses = []
ensemble_weights = [] 
train_auc_scores = []
val_auc_scores = []

# Training loop
for epoch in range(epochs):
    #Train classifier
    classifier.train()
    #Initialize loss
    total_train_loss = 0
    #Initialize logits for AUCROC
    train_logits = []
    train_labels = []
    #shows the progress bar 
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
    
    for batch in progress_bar:
        # Move data to cuda for training
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        optimizer.zero_grad()
        logits = classifier(input_ids, attention_mask, pixel_values)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track stats
        total_train_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
                               
        # Collect logits and labels, first stack into 1 tensor
        train_logits.append(logits.detach().cpu())
        train_labels.append(labels.detach().cpu())
     
    # Compute train AUC-ROC
    train_preds = torch.cat(train_logits).numpy()
    train_targets = torch.cat(train_labels).numpy()
    train_auc = roc_auc_score(train_targets, train_preds)
    train_auc_scores.append(train_auc)
    
    # Validation
    classifier.eval()
    total_val_loss = 0
    val_logits = []
    val_labels = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            #Run the model
            logits = classifier(input_ids, attention_mask, pixel_values)
            loss = criterion(logits, labels)
            total_val_loss += loss.item()
    
            #logits and labels to cpu
            val_logits.append(logits.cpu())
            val_labels.append(labels.cpu())
    
    #validation AUC-ROC, first stack into 1 tensor
    val_preds = torch.cat(val_logits).numpy()
    val_targets = torch.cat(val_labels).numpy()
    val_auc = roc_auc_score(val_targets, val_preds)
    val_auc_scores.append(val_auc)
                               
    # Save epoch metrics
    avg_train_loss = total_train_loss / len(train_loader)
    avg_val_loss = total_val_loss / len(val_loader)
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    
    #Print Epoch metrics
    print(f"Epoch {epoch+1} | "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | "
          f"Train AUC: {train_auc:.4f} | "
          f"Val AUC: {val_auc:.4f}")

# Plotting
plt.figure(figsize=(12, 5))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, epochs+1), val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Curves')
plt.legend()
plt.grid(True)

#AUC-ROC plot
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs+1), train_auc_scores, label='Train AUC-ROC')
plt.plot(range(1, epochs+1), val_auc_scores, label='Validation AUC-ROC')
plt.xlabel('Epoch')
plt.ylabel('AUC-ROC')
plt.title('Training Curves')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

    Epoch 1/15: 100%|██████████| 266/266 [09:37<00:00,  2.17s/it, loss=0.705]


    Epoch 1 | Train Loss: 0.7679 | Val Loss: 0.9246 | Train AUC: 0.7609 | Val AUC: 0.7124


    Epoch 2/15: 100%|██████████| 266/266 [09:37<00:00,  2.17s/it, loss=1.09] 


    Epoch 2 | Train Loss: 0.6564 | Val Loss: 0.9491 | Train AUC: 0.8333 | Val AUC: 0.7172


    Epoch 3/15: 100%|██████████| 266/266 [09:36<00:00,  2.17s/it, loss=0.694]


    Epoch 3 | Train Loss: 0.5754 | Val Loss: 0.9614 | Train AUC: 0.8768 | Val AUC: 0.7389


    Epoch 4/15: 100%|██████████| 266/266 [09:43<00:00,  2.19s/it, loss=0.688]


    Epoch 4 | Train Loss: 0.4712 | Val Loss: 1.2963 | Train AUC: 0.9184 | Val AUC: 0.7385


    Epoch 5/15: 100%|██████████| 266/266 [09:41<00:00,  2.19s/it, loss=0.168]


    Epoch 5 | Train Loss: 0.3454 | Val Loss: 1.4686 | Train AUC: 0.9567 | Val AUC: 0.7451


    Epoch 6/15: 100%|██████████| 266/266 [09:39<00:00,  2.18s/it, loss=0.344] 


    Epoch 6 | Train Loss: 0.2534 | Val Loss: 1.5524 | Train AUC: 0.9769 | Val AUC: 0.7434


    Epoch 7/15: 100%|██████████| 266/266 [09:46<00:00,  2.21s/it, loss=0.351] 


    Epoch 7 | Train Loss: 0.1727 | Val Loss: 2.0465 | Train AUC: 0.9890 | Val AUC: 0.7641


    Epoch 8/15: 100%|██████████| 266/266 [08:47<00:00,  1.98s/it, loss=0.26]   


    Epoch 8 | Train Loss: 0.1301 | Val Loss: 1.9383 | Train AUC: 0.9934 | Val AUC: 0.7642


    Epoch 9/15: 100%|██████████| 266/266 [06:24<00:00,  1.44s/it, loss=0.111] 


    Epoch 9 | Train Loss: 0.1085 | Val Loss: 2.4518 | Train AUC: 0.9955 | Val AUC: 0.7593


    Epoch 10/15: 100%|██████████| 266/266 [06:27<00:00,  1.46s/it, loss=0.0467] 


    Epoch 10 | Train Loss: 0.0855 | Val Loss: 2.2976 | Train AUC: 0.9972 | Val AUC: 0.7558


    Epoch 11/15: 100%|██████████| 266/266 [05:36<00:00,  1.27s/it, loss=0.0488] 


    Epoch 11 | Train Loss: 0.0795 | Val Loss: 2.2229 | Train AUC: 0.9972 | Val AUC: 0.7566


    Epoch 12/15: 100%|██████████| 266/266 [05:09<00:00,  1.16s/it, loss=0.0373] 


    Epoch 12 | Train Loss: 0.0628 | Val Loss: 2.5812 | Train AUC: 0.9982 | Val AUC: 0.7543


    Epoch 13/15: 100%|██████████| 266/266 [05:08<00:00,  1.16s/it, loss=0.0215] 


    Epoch 13 | Train Loss: 0.0619 | Val Loss: 2.2684 | Train AUC: 0.9983 | Val AUC: 0.7657


    Epoch 14/15: 100%|██████████| 266/266 [05:07<00:00,  1.15s/it, loss=0.0388] 


    Epoch 14 | Train Loss: 0.0556 | Val Loss: 2.6161 | Train AUC: 0.9983 | Val AUC: 0.7612


    Epoch 15/15: 100%|██████████| 266/266 [05:08<00:00,  1.16s/it, loss=0.0629] 


    Epoch 15 | Train Loss: 0.0549 | Val Loss: 2.6496 | Train AUC: 0.9984 | Val AUC: 0.7571



    
![png](output_21_30.png)
    


Accuracy


```python
from sklearn.metrics import accuracy_score, roc_auc_score,precision_score,recall_score,f1_score
import numpy as np
import torch

classifier.eval()

# Store predictions and labels
logits = []
test_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        outputs = classifier(input_ids, attention_mask, pixel_values)
        
        #model output
        logits.append(outputs.cpu())
        #Actual label
        test_labels.append(labels.cpu())

# Stack each batch output into one large tensor
logits = torch.cat(logits).numpy()  
test_labels = torch.cat(test_labels).numpy()  

# Convert logits to probabilities using sigmoid
probs = torch.sigmoid(torch.tensor(logits)).numpy()

# Convert probabilities to class predictions using threshold 0.5, could change this threshold
preds = (probs >= 0.5).astype(int)


# Compute basic metrics
acc = accuracy_score(test_labels, preds)
auc = roc_auc_score(test_labels, probs)
precision = precision_score(test_labels, preds)
recall = recall_score(test_labels, preds)
f1 = f1_score(test_labels, preds)
print("\nEvaluation Results:")
print(f"  Accuracy: {acc:.4f}")
print(f"  ROC AUC: {auc:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall: {recall:.4f}")
print(f"  F1 score: {f1:.4f}")

```

    
    Evaluation Results:
      Accuracy: 0.7190
      ROC AUC: 0.8100
      Precision: 0.8083
      Recall: 0.5592
      F1 score: 0.6610


# Use Validation Data to fine tune the threshold > optimize for f1 score


```python
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import numpy as np
import torch

classifier.eval()

# Store predictions and labels
logits = []
val_targets = []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        outputs = classifier(input_ids, attention_mask, pixel_values)
        
        #model output
        logits.append(outputs.cpu())
        #Actual label
        val_targets.append(labels.cpu())

# Stack each batch output into one large tensor
logits = torch.cat(logits).numpy()  
val_targets = torch.cat(val_targets).numpy()  

# Convert logits to probabilities using sigmoid
val_probs = torch.sigmoid(torch.tensor(logits)).numpy()

```


```python
from sklearn.metrics import precision_recall_fscore_support, f1_score
import numpy as np


thresholds = np.arange(0.0, 1.01, 0.01)
best_f1 = 0
best_thresh = 0

for t in thresholds:
    y_pred = (val_probs >= t).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
    val_targets, y_pred, average='binary', zero_division=1)

    
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print(f"Best threshold: {best_thresh:.2f}, Best F1: {best_f1:.4f}")

```

    Best threshold: 0.01, Best F1: 0.6965


# Model performance with the optimized threshold


```python
# Convert probabilities to class predictions using threshold 0.5, could change this threshold
preds = (probs >= 0.01).astype(int)

# Compute basic metrics
acc = accuracy_score(test_labels, preds)
auc = roc_auc_score(test_labels, probs)
precision = precision_score(test_labels, preds)
recall = recall_score(test_labels, preds)
f1 = f1_score(test_labels, preds)
print("\nEvaluation Results:")
print(f"  Accuracy: {acc:.4f}")
print(f"  ROC AUC: {auc:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall: {recall:.4f}")
print(f"  F1 score: {f1:.4f}")
```

    
    Evaluation Results:
      Accuracy: 0.7240
      ROC AUC: 0.8100
      Precision: 0.6845
      Recall: 0.8102
      F1 score: 0.7421



```python
from sklearn.metrics import roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(test_labels, preds)
tn, fp, fn, tp = cm.ravel()
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Non-hateful', 'Hateful'],
            yticklabels=['Non-hateful', 'Hateful'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix: TP, TN, FP, FN')
plt.show()
```


    
![png](output_29_0.png)
    



```python

```
