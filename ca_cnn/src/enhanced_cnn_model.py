"""
UPDATED enhanced_cnn_model.py
 : Use research parameters as STARTING POINTS for novel improvements
Focus: Deeper analysis through confidence-based classification + adaptive tuning
ENHANCEMENT: Added Precision and Recall to comprehensive evaluation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm
import itertools
import random
import numpy as np

class EnhancedHatefulMemesCNN(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.5, backbone='resnet50', pretrained=True):
        super(EnhancedHatefulMemesCNN, self).__init__()
        self.backbone_name = backbone
        
        # Keep 4 CNN architectures
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        elif backbone == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        elif backbone == 'efficientnet_b0':
            self.backbone = timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=0)
            feature_dim = self.backbone.num_features
            
        elif backbone == 'densenet121':
            self.backbone = models.densenet121(pretrained=pretrained)
            feature_dim = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
            
        else:
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        
        #   1: Enhanced attention with confidence-aware weighting
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(feature_dim // 4, feature_dim),
            nn.Sigmoid()
        )
        
        #   2: Dual-head classifier with uncertainty quantification
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, num_classes)
        )
        
        #   3: Multi-layer confidence head for deeper analysis
        self.confidence_head = nn.Sequential(
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate * 0.25),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        #   4: Confidence-aware feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.3)
        )
    
    def forward(self, x, return_confidence=False, return_attention=False):
        # Extract backbone features
        features = self.backbone(x)
        
        # Apply attention mechanism
        attention_weights = self.attention(features)
        attended_features = features * attention_weights
        
        #  : Confidence-aware feature fusion
        fused_features = self.feature_fusion(attended_features) + features
        
        # Classification
        logits = self.classifier(fused_features)
        
        if return_confidence or return_attention:
            outputs = [logits]
            
            if return_confidence:
                confidence = self.confidence_head(fused_features)
                outputs.append(confidence.squeeze(1))
            
            if return_attention:
                outputs.append(attention_weights)
            
            return tuple(outputs) if len(outputs) > 2 else (outputs[0], outputs[1])
        
        return logits

def get_cnn_architectures():
    """Get the 4 CNN architectures to test"""
    return ['resnet50', 'resnet101', 'efficientnet_b0', 'densenet121']

def get_adaptive_hyperparameter_space():
    """
     : Adaptive hyperparameter exploration around research starting points
    Uses proven baselines but explores improvements for hateful memes specifically
    """
    
    # Research starting points from successful papers
    research_baselines = {
        'conservative': {'lr': 1e-4, 'dropout': 0.5, 'weight_decay': 1e-4},
        'aggressive': {'lr': 3e-4, 'dropout': 0.4, 'weight_decay': 5e-4},
        'stable': {'lr': 5e-5, 'dropout': 0.6, 'weight_decay': 1e-3}
    }
    
    #  : Adaptive variations around research baselines
    adaptive_configs = []
    
    for i, (baseline_name, baseline) in enumerate(research_baselines.items(), 1):
        # Base configuration from research
        base_config = {
            'trial_id': i,
            'baseline_source': baseline_name,
            'lr': baseline['lr'],
            'batch_size': 32,
            'dropout_rate': baseline['dropout'],
            'weight_decay': baseline['weight_decay'],
            'epochs': 25,
            'optimizer': 'adamw',
            'scheduler': 'cosine',
            'confidence_weight': 0.1,  #  : Confidence loss weighting
            'attention_dropout': baseline['dropout'] * 0.5  #  : Separate attention dropout
        }
        adaptive_configs.append(base_config)
        
        #  : Add adaptive variations for deeper analysis
        # Variation 1: Higher confidence weighting
        confidence_config = base_config.copy()
        confidence_config.update({
            'trial_id': i + 10,
            'baseline_source': f'{baseline_name}_high_confidence',
            'confidence_weight': 0.3,  # Higher confidence emphasis
            'dropout_rate': baseline['dropout'] * 1.2,  # More regularization
        })
        adaptive_configs.append(confidence_config)
        
        # Variation 2: Attention-focused training
        attention_config = base_config.copy()
        attention_config.update({
            'trial_id': i + 20,
            'baseline_source': f'{baseline_name}_attention_focus',
            'attention_dropout': baseline['dropout'] * 0.3,  # Less attention dropout
            'lr': baseline['lr'] * 0.7,  # Slower learning for attention
        })
        adaptive_configs.append(attention_config)
    
    return adaptive_configs

def random_search_hyperparameters(n_trials=3):
    """
    UPDATED: Now returns adaptive configurations around research baselines
    Enables deeper analysis through systematic variations
    """
    print("🔬 Using adaptive hyperparameters around research baselines")
    print("💡  : Systematic exploration for deeper analysis")
    
    adaptive_configs = get_adaptive_hyperparameter_space()
    
    # Return the requested number of trials (prioritize base configs first)
    if n_trials <= 3:
        # Return base research configurations
        return adaptive_configs[:n_trials]
    else:
        # Include adaptive variations for deeper analysis
        return adaptive_configs[:n_trials]

def train_cnn_with_hyperparameters(train_loader, val_loader, test_loader, 
                                  architecture, hyperparams, device):
    """
     : Enhanced training with confidence-aware loss and deeper analysis
    UPDATED: Now includes ALL 5 STANDARD CLASSIFICATION METRICS
    """
    import torch.optim as optim
    from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np
    import os
    import time
    
    print(f"🚀 Training {architecture.upper()} - {hyperparams.get('baseline_source', 'Unknown')} Config")
    print(f"📋  : LR={hyperparams['lr']}, Conf_Weight={hyperparams.get('confidence_weight', 0.1)}")
    
    training_start = time.time()
    
    # Calculate class weights for balanced training
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.numpy())
    
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(all_labels),
        y=all_labels
    )
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    # Create enhanced model
    model = EnhancedHatefulMemesCNN(
        num_classes=2,
        dropout_rate=hyperparams['dropout_rate'],
        backbone=architecture,
        pretrained=True
    ).to(device)
    
    #  : Enhanced loss functions for deeper analysis
    classification_criterion = nn.CrossEntropyLoss(weight=class_weights)
    confidence_criterion = nn.MSELoss()
    
    #  : Confidence-aware loss weighting
    confidence_weight = hyperparams.get('confidence_weight', 0.1)
    
    # Optimizer with adaptive parameters
    if hyperparams['optimizer'] == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=hyperparams['lr'],
            weight_decay=hyperparams['weight_decay']
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=hyperparams['lr'],
            weight_decay=hyperparams['weight_decay'],
            momentum=0.9,
            nesterov=True
        )
    
    # Scheduler
    if hyperparams['scheduler'] == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=hyperparams['epochs'])
    elif hyperparams['scheduler'] == 'step':
        scheduler = StepLR(optimizer, step_size=hyperparams['epochs']//3, gamma=0.1)
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    #  : Track confidence evolution for deeper analysis
    confidence_evolution = []
    attention_analysis = []
    
    # Enhanced training loop
    best_val_auc = 0
    best_model_state = None
    patience_counter = 0
    patience_limit = 6
    
    print(f"🏃 Training with confidence weight: {confidence_weight}")
    
    for epoch in range(hyperparams['epochs']):
        # Training phase with confidence tracking
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        epoch_confidences = []
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            #  : Forward pass with confidence and attention
            outputs, confidence = model(images, return_confidence=True)
            
            #  : Enhanced loss computation
            classification_loss = classification_criterion(outputs, labels)
            
            # Confidence target based on prediction correctness
            with torch.no_grad():
                predictions = torch.argmax(outputs, dim=1)
                confidence_targets = (predictions == labels).float()
            
            confidence_loss = confidence_criterion(confidence, confidence_targets)
            
            #  : Adaptive confidence weighting
            total_loss = classification_loss + confidence_weight * confidence_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Track metrics
            train_loss += total_loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            epoch_confidences.extend(confidence.detach().cpu().numpy())
        
        train_accuracy = train_correct / train_total
        
        #  : Track confidence evolution for analysis
        confidence_evolution.append({
            'epoch': epoch + 1,
            'mean_confidence': np.mean(epoch_confidences),
            'confidence_std': np.std(epoch_confidences),
            'train_accuracy': train_accuracy
        })
        
        # UPDATED: Validation with ALL 5 METRICS
        val_accuracy, val_auc, val_precision, val_recall, val_f1 = evaluate_model_enhanced(model, val_loader, device)
        
        # Scheduler step
        if hyperparams['scheduler'] == 'plateau':
            scheduler.step(val_auc)
        else:
            scheduler.step()
        
        # Enhanced progress reporting
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == hyperparams['epochs'] - 1:
            conf_mean = np.mean(epoch_confidences)
            print(f"   Epoch {epoch+1:2d}: Train Acc={train_accuracy:.3f}, "
                  f"Val AUC={val_auc:.3f}, Val F1={val_f1:.3f}, Confidence={conf_mean:.3f}")
        
        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience_limit:
            print(f"   ⏸️  Early stopping at epoch {epoch+1}")
            break
    
    # Load best model and comprehensive evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # UPDATED: Comprehensive evaluation with ALL 5 METRICS
    test_accuracy, test_auc, test_precision, test_recall, test_f1 = evaluate_model_enhanced(model, test_loader, device)
    confidence_analysis = get_enhanced_confidence_analysis(model, test_loader, device)
    
    training_time = (time.time() - training_start) / 60
    
    # Save model with enhanced naming
    os.makedirs('./models', exist_ok=True)
    baseline_name = hyperparams.get('baseline_source', 'unknown')
    model_path = f'./models/enhanced_{architecture}_{baseline_name}_trial_{hyperparams["trial_id"]}.pth'
    torch.save(model.state_dict(), model_path)
    
    # UPDATED: Return comprehensive results with ALL 5 METRICS
    results = {
        'architecture': architecture,
        'hyperparameters': hyperparams,
        'baseline_source': hyperparams.get('baseline_source', 'unknown'),
        'best_val_auc': best_val_auc,
        'training_time_minutes': training_time,
        'test_results': {
            'accuracy': test_accuracy,
            'auc_roc': test_auc,
            'precision': test_precision,    # NEW
            'recall': test_recall,          # NEW  
            'f1_score': test_f1
        },
        'confidence_analysis': confidence_analysis,
        'confidence_evolution': confidence_evolution,  # NEW: Training dynamics
        ' _metrics': {
            'confidence_weight_used': confidence_weight,
            'attention_dropout': hyperparams.get('attention_dropout', 0.25),
            'adaptive_variations': len([k for k in hyperparams.keys() if ' ' in str(k)])
        },
        'model_path': model_path
    }
    
    print(f"✅ {architecture.upper()} ({baseline_name}) completed:")
    print(f"   📊 Test Metrics: AUC={test_auc:.4f}, Acc={test_accuracy:.4f}, Prec={test_precision:.4f}, Rec={test_recall:.4f}, F1={test_f1:.4f}")
    print(f"   🎯 Confidence Analysis: {confidence_analysis['confidence_accuracy_correlation']:.3f}")
    print(f"   ⏱️  Training time: {training_time:.1f} minutes")
    
    return results

def evaluate_model_enhanced(model, data_loader, device):
    """
    UPDATED: Enhanced evaluation with ALL 5 STANDARD CLASSIFICATION METRICS
    Returns: accuracy, auc_roc, precision, recall, f1_score
    """
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
    
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate ALL 5 standard classification metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    auc_roc = roc_auc_score(all_labels, all_probabilities)
    precision = precision_score(all_labels, all_predictions, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='binary', zero_division=0)
    
    return accuracy, auc_roc, precision, recall, f1

def get_enhanced_confidence_analysis(model, data_loader, device):
    """
     : Enhanced confidence analysis for deeper insights
    """
    model.eval()
    all_confidences = []
    all_correct = []
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs, confidence = model(images, return_confidence=True)
            predictions = torch.argmax(outputs, dim=1)
            correct = (predictions == labels)
            
            all_confidences.extend(confidence.cpu().numpy())
            all_correct.extend(correct.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Enhanced confidence analysis
    confidences = np.array(all_confidences)
    correct = np.array(all_correct)
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)
    
    #  : Multiple confidence thresholds for deeper analysis
    thresholds = [0.6, 0.7, 0.8, 0.9]
    threshold_analysis = {}
    
    for threshold in thresholds:
        high_conf_mask = confidences >= threshold
        if high_conf_mask.sum() > 0:
            threshold_analysis[f'threshold_{threshold}'] = {
                'accuracy': correct[high_conf_mask].mean(),
                'coverage': high_conf_mask.mean(),
                'precision': correct[high_conf_mask].sum() / high_conf_mask.sum()
            }
    
    #  : Class-specific confidence analysis
    class_0_mask = labels == 0
    class_1_mask = labels == 1
    
    return {
        'mean_confidence': confidences.mean(),
        'confidence_std': confidences.std(),
        'high_conf_accuracy': correct[confidences > 0.8].mean() if (confidences > 0.8).sum() > 0 else 0,
        'high_conf_coverage': (confidences > 0.8).mean(),
        'confidence_accuracy_correlation': np.corrcoef(confidences, correct)[0, 1] if len(confidences) > 1 else 0,
        'threshold_analysis': threshold_analysis,  # NEW: Multi-threshold analysis
        'class_specific_confidence': {  # NEW: Class-specific insights
            'class_0_mean_conf': confidences[class_0_mask].mean() if class_0_mask.sum() > 0 else 0,
            'class_1_mean_conf': confidences[class_1_mask].mean() if class_1_mask.sum() > 0 else 0,
        },
        'confidence_distribution': {  # NEW: Distribution analysis
            'q25': np.percentile(confidences, 25),
            'q50': np.percentile(confidences, 50),
            'q75': np.percentile(confidences, 75),
            'q95': np.percentile(confidences, 95)
        }
    }