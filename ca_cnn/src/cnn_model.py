import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from collections import defaultdict
import os
from tqdm import tqdm

class HatefulMemesCNN(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super(HatefulMemesCNN, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def calculate_class_weights(train_loader, device):
    label_counts = defaultdict(int)
    for _, labels in train_loader:
        for label in labels:
            label_counts[label.item()] += 1
    
    total_samples = sum(label_counts.values())
    weights = {}
    for class_id, count in label_counts.items():
        weights[class_id] = total_samples / (2 * count)
    
    weight_tensor = torch.tensor([weights[0], weights[1]], dtype=torch.float32)
    print(f"Class distribution: {dict(label_counts)}")
    return weight_tensor.to(device)

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    
    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = total_loss / len(train_loader)
    epoch_acc = accuracy_score(all_labels, all_predictions)
    return epoch_loss, epoch_acc

def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_probabilities = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = total_loss / len(val_loader)
    epoch_acc = accuracy_score(all_labels, all_predictions)
    epoch_auc = roc_auc_score(all_labels, all_probabilities)
    epoch_f1 = f1_score(all_labels, all_predictions)
    
    return epoch_loss, epoch_acc, epoch_auc, epoch_f1

def evaluate_test_set(model, test_loader, device):
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_acc = accuracy_score(all_labels, all_predictions)
    test_auc = roc_auc_score(all_labels, all_probabilities)
    test_f1 = f1_score(all_labels, all_predictions)
    cm = confusion_matrix(all_labels, all_predictions)
    
    return {
        'accuracy': test_acc,
        'auc_roc': test_auc,
        'f1_score': test_f1,
        'confusion_matrix': cm
    }

def plot_confusion_matrix(cm, save_path="./results/confusion_matrix.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Non-hateful', 'Hateful'],
               yticklabels=['Non-hateful', 'Hateful'])
    plt.title('Confusion Matrix - Test Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to: {save_path}")
    plt.show()

def main_experiment(train_loader, val_loader, test_loader, lr=1e-4, epochs=20, dropout_rate=0.5):
    print("🚀 HATEFUL MEMES CNN EXPERIMENT")
    print("=" * 50)
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./results", exist_ok=True)
    
    # Initialize model
    model = HatefulMemesCNN(num_classes=2, dropout_rate=dropout_rate)
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup training
    class_weights = calculate_class_weights(train_loader, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Training loop
    history = defaultdict(list)
    best_val_auc = 0.0
    
    print(f"\nStarting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, val_acc, val_auc, val_f1 = validate_epoch(model, val_loader, criterion, device)
        
        # Record metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        history['val_f1'].append(val_f1)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}")
        
        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), './models/best_model.pth')
            print(f"New best model saved! (AUC: {val_auc:.4f})")
    
    # Load best model and evaluate
    model.load_state_dict(torch.load('./models/best_model.pth', map_location=device))
    test_results = evaluate_test_set(model, test_loader, device)
    
    # Plot confusion matrix
    plot_confusion_matrix(test_results['confusion_matrix'])
    
    print(f"\nExperiment completed!")
    print(f"Best Validation AUC: {best_val_auc:.4f}")
    print(f"Test Accuracy: {test_results['accuracy']:.4f}")
    print(f"Test AUC-ROC: {test_results['auc_roc']:.4f}")
    print(f"Test F1-Score: {test_results['f1_score']:.4f}")
    
    return {
        'training_history': dict(history),
        'test_results': test_results,
        'best_val_auc': best_val_auc
    }

if __name__ == "__main__":
    print("CNN Model module loaded successfully!")
    print("Available functions: main_experiment")

