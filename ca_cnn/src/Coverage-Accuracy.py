

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_curve, roc_curve, auc
import pandas as pd
import os
import sys
from tqdm import tqdm

# Add src directory to Python path for Colab
sys.path.append('/content/drive/MyDrive/hateful_memes_cnn/src')

class CoverageAccuracyAnalyzer:

    
    def __init__(self, model, device=None):

        self.model = model
        self.model.eval()
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else 
                                 "mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device
        
        # Check if model has confidence estimation
        self.has_confidence_head = self._check_confidence_capability()
        
        print(f"✅ Coverage-Accuracy analyzer initialized")
        print(f"🔧 Device: {self.device}")
        print(f"🎯 Confidence estimation: {'Available' if self.has_confidence_head else 'Using softmax (Basic Model)'}")
    
    def _check_confidence_capability(self):
        """
        Check if model has built-in confidence estimation
        
        Returns:
            bool: True if model has confidence head
        """
        # Test if model can return confidence scores
        try:
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            with torch.no_grad():
                output = self.model(dummy_input, return_confidence=True)
                return isinstance(output, tuple) and len(output) >= 2
        except:
            return False
    
    def extract_predictions_and_confidence(self, data_loader, max_samples=None):
        """
        Extract predictions, true labels, and confidence scores from data loader
        
        Args:
            data_loader: PyTorch data loader
            max_samples: Maximum samples to process (None for all)
            
        Returns:
            dict: Contains predictions, labels, confidences, and probabilities
        """
        print("🔍 Extracting predictions and confidence scores...")
        
        all_predictions = []
        all_labels = []
        all_confidences = []
        all_probabilities = []
        all_correct = []
        
        samples_processed = 0
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(tqdm(data_loader, desc="Processing batches")):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Get model outputs
                if self.has_confidence_head:
                    # Enhanced model with confidence head
                    outputs, confidence_scores = self.model(images, return_confidence=True)
                    confidences = confidence_scores.cpu().numpy()
                else:
                    # Basic model - use softmax probabilities as confidence
                    outputs = self.model(images)
                    probabilities = F.softmax(outputs, dim=1)
                    confidences = torch.max(probabilities, dim=1)[0].cpu().numpy()
                
                # Get predictions and probabilities
                probabilities = F.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_confidences.extend(confidences)
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # Prob of hateful class
                
                # Track correctness
                correct = (predictions == labels).cpu().numpy()
                all_correct.extend(correct)
                
                samples_processed += len(images)
                if max_samples and samples_processed >= max_samples:
                    break
        
        results = {
            'predictions': np.array(all_predictions),
            'labels': np.array(all_labels),
            'confidences': np.array(all_confidences),
            'probabilities': np.array(all_probabilities),
            'correct': np.array(all_correct),
            'total_samples': len(all_predictions)
        }
        
        print(f"✅ Processed {results['total_samples']} samples")
        print(f"📊 Overall accuracy: {results['correct'].mean():.4f}")
        
        return results
    
    def compute_coverage_accuracy_curve(self, results, confidence_thresholds=None):
        """
        Compute coverage-accuracy curve at different confidence thresholds
        
        Args:
            results: Output from extract_predictions_and_confidence
            confidence_thresholds: List of thresholds to evaluate
            
        Returns:
            dict: Coverage-accuracy curve data
        """
        if confidence_thresholds is None:
            # Create 50 evenly spaced thresholds
            min_conf = results['confidences'].min()
            max_conf = results['confidences'].max()
            confidence_thresholds = np.linspace(min_conf, max_conf, 50)
        
        print(f"📈 Computing coverage-accuracy curve with {len(confidence_thresholds)} thresholds...")
        
        coverages = []
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        sample_counts = []
        
        for threshold in confidence_thresholds:
            # Select samples above threshold
            high_conf_mask = results['confidences'] >= threshold
            
            if high_conf_mask.sum() == 0:
                # No samples above threshold
                coverages.append(0.0)
                accuracies.append(0.0)
                precisions.append(0.0)
                recalls.append(0.0)
                f1_scores.append(0.0)
                sample_counts.append(0)
                continue
            
            # Calculate metrics for high-confidence samples
            high_conf_correct = results['correct'][high_conf_mask]
            high_conf_predictions = results['predictions'][high_conf_mask]
            high_conf_labels = results['labels'][high_conf_mask]
            
            coverage = high_conf_mask.mean()
            accuracy = high_conf_correct.mean()
            
            # Calculate precision, recall, F1 for hateful class
            if len(np.unique(high_conf_labels)) > 1:  # Both classes present
                from sklearn.metrics import precision_score, recall_score, f1_score
                precision = precision_score(high_conf_labels, high_conf_predictions, pos_label=1, zero_division=0)
                recall = recall_score(high_conf_labels, high_conf_predictions, pos_label=1, zero_division=0)
                f1 = f1_score(high_conf_labels, high_conf_predictions, pos_label=1, zero_division=0)
            else:
                precision = 0.0
                recall = 0.0
                f1 = 0.0
            
            coverages.append(coverage)
            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
            sample_counts.append(high_conf_mask.sum())
        
        curve_data = {
            'thresholds': confidence_thresholds,
            'coverages': np.array(coverages),
            'accuracies': np.array(accuracies),
            'precisions': np.array(precisions),
            'recalls': np.array(recalls),
            'f1_scores': np.array(f1_scores),
            'sample_counts': np.array(sample_counts)
        }
        
        print(f"✅ Coverage-accuracy curve computed")
        
        return curve_data
    
    def find_optimal_thresholds(self, curve_data, target_accuracy=0.8, target_coverage=0.8):
        """
        Find optimal confidence thresholds for different criteria
        
        Args:
            curve_data: Output from compute_coverage_accuracy_curve
            target_accuracy: Minimum desired accuracy
            target_coverage: Minimum desired coverage
            
        Returns:
            dict: Optimal thresholds and their characteristics
        """
        print(f"🎯 Finding optimal thresholds...")
        
        thresholds = curve_data['thresholds']
        coverages = curve_data['coverages']
        accuracies = curve_data['accuracies']
        
        optimal_thresholds = {}
        
        # 1. Threshold for target accuracy
        accuracy_mask = accuracies >= target_accuracy
        if accuracy_mask.any():
            # Find highest coverage that meets accuracy requirement
            valid_indices = np.where(accuracy_mask)[0]
            best_idx = valid_indices[np.argmax(coverages[valid_indices])]
            
            optimal_thresholds['target_accuracy'] = {
                'threshold': thresholds[best_idx],
                'accuracy': accuracies[best_idx],
                'coverage': coverages[best_idx],
                'samples': curve_data['sample_counts'][best_idx]
            }
        
        # 2. Threshold for target coverage
        coverage_mask = coverages >= target_coverage
        if coverage_mask.any():
            # Find highest accuracy that meets coverage requirement
            valid_indices = np.where(coverage_mask)[0]
            best_idx = valid_indices[np.argmax(accuracies[valid_indices])]
            
            optimal_thresholds['target_coverage'] = {
                'threshold': thresholds[best_idx],
                'accuracy': accuracies[best_idx],
                'coverage': coverages[best_idx],
                'samples': curve_data['sample_counts'][best_idx]
            }
        
        # 3. Balanced threshold (maximize accuracy * coverage)
        balance_scores = accuracies * coverages
        best_balance_idx = np.argmax(balance_scores)
        
        optimal_thresholds['balanced'] = {
            'threshold': thresholds[best_balance_idx],
            'accuracy': accuracies[best_balance_idx],
            'coverage': coverages[best_balance_idx],
            'balance_score': balance_scores[best_balance_idx],
            'samples': curve_data['sample_counts'][best_balance_idx]
        }
        
        # 4. High precision threshold
        f1_scores = curve_data['f1_scores']
        best_f1_idx = np.argmax(f1_scores)
        
        optimal_thresholds['best_f1'] = {
            'threshold': thresholds[best_f1_idx],
            'accuracy': accuracies[best_f1_idx],
            'coverage': coverages[best_f1_idx],
            'f1_score': f1_scores[best_f1_idx],
            'samples': curve_data['sample_counts'][best_f1_idx]
        }
        
        print(f"✅ Found {len(optimal_thresholds)} optimal threshold configurations")
        
        return optimal_thresholds
    
    def create_coverage_accuracy_plots(self, curve_data, optimal_thresholds, save_path="/content/drive/MyDrive/hateful_memes_cnn/results/coverage_accuracy_analysis.png"):
        """
        Create comprehensive coverage-accuracy visualization
        
        Args:
            curve_data: Coverage-accuracy curve data
            optimal_thresholds: Optimal threshold configurations
            save_path: Path to save the visualization
        """
        print(f"📊 Creating coverage-accuracy plots...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Main Coverage-Accuracy Curve
        ax1 = axes[0, 0]
        ax1.plot(curve_data['coverages'], curve_data['accuracies'], 'b-', linewidth=2, label='Coverage-Accuracy Curve')
        ax1.scatter([1.0], [curve_data['accuracies'][0]], color='red', s=100, label='No Rejection (All Samples)', zorder=5)
        
        # Add optimal points
        if 'balanced' in optimal_thresholds:
            opt = optimal_thresholds['balanced']
            ax1.scatter([opt['coverage']], [opt['accuracy']], color='green', s=150, 
                       label=f'Balanced (Acc={opt["accuracy"]:.3f}, Cov={opt["coverage"]:.3f})', zorder=5)
        
        ax1.set_xlabel('Coverage (Fraction of Samples Predicted)', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('Coverage vs Accuracy Tradeoff', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        
        # 2. Confidence Threshold vs Metrics
        ax2 = axes[0, 1]
        ax2.plot(curve_data['thresholds'], curve_data['accuracies'], 'b-', label='Accuracy', linewidth=2)
        ax2.plot(curve_data['thresholds'], curve_data['coverages'], 'r--', label='Coverage', linewidth=2)
        ax2.plot(curve_data['thresholds'], curve_data['f1_scores'], 'g:', label='F1 Score', linewidth=2)
        
        ax2.set_xlabel('Confidence Threshold', fontsize=12)
        ax2.set_ylabel('Score', fontsize=12)
        ax2.set_title('Metrics vs Confidence Threshold', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_ylim(0, 1)
        
        # 3. Sample Count vs Threshold
        ax3 = axes[0, 2]
        ax3.plot(curve_data['thresholds'], curve_data['sample_counts'], 'purple', linewidth=2)
        ax3.fill_between(curve_data['thresholds'], curve_data['sample_counts'], alpha=0.3, color='purple')
        
        ax3.set_xlabel('Confidence Threshold', fontsize=12)
        ax3.set_ylabel('Number of Samples', fontsize=12)
        ax3.set_title('Sample Count vs Threshold', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Risk-Coverage Analysis
        ax4 = axes[1, 0]
        risks = 1 - curve_data['accuracies']  # Risk = 1 - Accuracy
        ax4.plot(curve_data['coverages'], risks, 'red', linewidth=2, label='Risk-Coverage Curve')
        ax4.fill_between(curve_data['coverages'], risks, alpha=0.3, color='red')
        
        ax4.set_xlabel('Coverage', fontsize=12)
        ax4.set_ylabel('Risk (1 - Accuracy)', fontsize=12)
        ax4.set_title('Risk vs Coverage Analysis', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, max(1, risks.max() * 1.1))
        
        # 5. Optimal Thresholds Comparison
        ax5 = axes[1, 1]
        threshold_names = []
        threshold_accuracies = []
        threshold_coverages = []
        
        for name, config in optimal_thresholds.items():
            threshold_names.append(name.replace('_', ' ').title())
            threshold_accuracies.append(config['accuracy'])
            threshold_coverages.append(config['coverage'])
        
        x = np.arange(len(threshold_names))
        width = 0.35
        
        bars1 = ax5.bar(x - width/2, threshold_accuracies, width, label='Accuracy', alpha=0.8)
        bars2 = ax5.bar(x + width/2, threshold_coverages, width, label='Coverage', alpha=0.8)
        
        ax5.set_xlabel('Threshold Strategy', fontsize=12)
        ax5.set_ylabel('Score', fontsize=12)
        ax5.set_title('Optimal Threshold Comparison', fontsize=14, fontweight='bold')
        ax5.set_xticks(x)
        ax5.set_xticklabels(threshold_names, rotation=45)
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
        ax5.set_ylim(0, 1)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        # 6. Confidence Distribution Analysis
        ax6 = axes[1, 2]
        ax6.hist(curve_data['thresholds'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax6.axvline(np.mean(curve_data['thresholds']), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(curve_data["thresholds"]):.3f}')
        ax6.axvline(np.median(curve_data['thresholds']), color='green', linestyle='--',
                   label=f'Median: {np.median(curve_data["thresholds"]):.3f}')
        
        ax6.set_xlabel('Confidence Threshold', fontsize=12)
        ax6.set_ylabel('Frequency', fontsize=12)
        ax6.set_title('Threshold Distribution', fontsize=14, fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle('Coverage-Accuracy Analysis for Selective Prediction', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save the plot
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Coverage-accuracy plots saved to: {save_path}")
        
        # Also save to Colab results folder
        colab_results = '/content/drive/MyDrive/hateful_memes_cnn/results'
        os.makedirs(colab_results, exist_ok=True)
        
        return save_path

def load_model_for_coverage_analysis(model_path, model_type='basic'):

    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else "cpu")
    
    if model_type == 'basic':
        from cnn_model import HatefulMemesCNN
        model = HatefulMemesCNN(num_classes=2, dropout_rate=0.5)
    else:
        from enhanced_cnn_model import EnhancedHatefulMemesCNN
        model = EnhancedHatefulMemesCNN(num_classes=2, dropout_rate=0.5)
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded from: {model_path}")
    print(f"🔧 Using device: {device}")
    
    return model

def generate_coverage_accuracy_report(curve_data, optimal_thresholds, save_path="/content/drive/MyDrive/hateful_memes_cnn/results/coverage_accuracy_report.md"):
    """
    Generate a comprehensive report for coverage-accuracy analysis
    
    Args:
        curve_data: Coverage-accuracy curve data
        optimal_thresholds: Optimal threshold configurations
        save_path: Path to save the report
    """
    report_content = f"""
# Coverage-Accuracy Analysis Report

## Overview
This report presents the coverage-accuracy analysis for selective prediction in hateful meme detection. The analysis shows how prediction quality varies with coverage (the fraction of samples we choose to predict on).

## Key Findings

### Overall Performance
- **Maximum Accuracy**: {curve_data['accuracies'].max():.4f}
- **Full Coverage Accuracy**: {curve_data['accuracies'][0]:.4f}
- **Accuracy Improvement Range**: {curve_data['accuracies'].max() - curve_data['accuracies'][0]:.4f}

### Optimal Thresholds

"""
    
    for name, config in optimal_thresholds.items():
        strategy_name = name.replace('_', ' ').title()
        report_content += f"""
#### {strategy_name} Strategy
- **Confidence Threshold**: {config['threshold']:.4f}
- **Accuracy**: {config['accuracy']:.4f}
- **Coverage**: {config['coverage']:.4f}
- **Samples Retained**: {config['samples']:,}
"""
        
        if 'balance_score' in config:
            report_content += f"- **Balance Score**: {config['balance_score']:.4f}\n"
        if 'f1_score' in config:
            report_content += f"- **F1 Score**: {config['f1_score']:.4f}\n"
    
    report_content += f"""

## Deployment Recommendations

### High-Stakes Applications
For applications where accuracy is critical (e.g., automated content moderation):
- **Recommended Strategy**: Target Accuracy
- **Suggested Threshold**: {optimal_thresholds.get('target_accuracy', {}).get('threshold', 'N/A'):.4f}
- **Expected Performance**: {optimal_thresholds.get('target_accuracy', {}).get('accuracy', 0):.1%} accuracy on {optimal_thresholds.get('target_accuracy', {}).get('coverage', 0):.1%} of samples

### Balanced Applications  
For applications balancing accuracy and coverage:
- **Recommended Strategy**: Balanced
- **Suggested Threshold**: {optimal_thresholds.get('balanced', {}).get('threshold', 'N/A'):.4f}
- **Expected Performance**: {optimal_thresholds.get('balanced', {}).get('accuracy', 0):.1%} accuracy on {optimal_thresholds.get('balanced', {}).get('coverage', 0):.1%} of samples

### High-Volume Applications
For applications requiring broad coverage:
- **Recommended Strategy**: Target Coverage
- **Suggested Threshold**: {optimal_thresholds.get('target_coverage', {}).get('threshold', 'N/A'):.4f}
- **Expected Performance**: {optimal_thresholds.get('target_coverage', {}).get('accuracy', 0):.1%} accuracy on {optimal_thresholds.get('target_coverage', {}).get('coverage', 0):.1%} of samples

## Implementation Guide

### Step 1: Set Confidence Threshold
Choose a confidence threshold based on application requirements from the recommendations above.

### Step 2: Implement Selective Prediction


### Step 3: Handle Rejected Predictions


## Conclusion
The coverage-accuracy analysis demonstrates that selective prediction can significantly improve accuracy by identifying and handling uncertain predictions appropriately. This is particularly valuable for content moderation systems where false positives and negatives carry real consequences.
"""
    
    # Save report - ensure directory exists in Colab
    colab_results = '/content/drive/MyDrive/hateful_memes_cnn/results'
    os.makedirs(colab_results, exist_ok=True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        f.write(report_content)
    
    print(f"✅ Coverage-accuracy report saved to: {save_path}")

def run_coverage_accuracy_analysis():
    """
    Main function to run complete coverage-accuracy analysis
    """
    print("📈 COVERAGE-ACCURACY ANALYSIS")
    print("=" * 50)
    
    # Configuration for Colab - Will use freshly trained enhanced model
    MODEL_PATH = "/content/drive/MyDrive/hateful_memes_cnn/models/enhanced_best_model.pth"
    MODEL_TYPE = "enhanced"  # Use enhanced model from step 2
    DATA_DIR = "/content/drive/MyDrive/hateful_memes_cnn/data/hateful_memes_expanded"
    DATA_DIR = "/content/drive/MyDrive/hateful_memes_cnn/data/hateful_memes_expanded"
    
    # Check if model exists, fallback to basic if needed
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️  Enhanced model not found: {MODEL_PATH}")
        print("🔄 Falling back to basic model...")
        MODEL_PATH = "/content/drive/MyDrive/hateful_memes_cnn/models/best_model.pth"
        MODEL_TYPE = "basic"
        
        if not os.path.exists(MODEL_PATH):
            print(f"❌ Basic model also not found: {MODEL_PATH}")
            print("Please train a model first!")
            return
    
    print(f"📂 Loading model from: {MODEL_PATH}")
    
    # Load model
    try:
        model = load_model_for_coverage_analysis(MODEL_PATH, MODEL_TYPE)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Initialize analyzer
    analyzer = CoverageAccuracyAnalyzer(model)
    
    # Load test data
    print("📦 Loading test data...")
    try:
        from data_preparation import setup_data_loaders
        _, _, test_loader, _ = setup_data_loaders(
            data_dir=DATA_DIR,
            batch_size=32,
            val_size=0.2,
            num_workers=0  # Reduced for compatibility
        )
        
        if test_loader is None:
            print("❌ Failed to load test data")
            return
            
        print(f"✅ Test data loaded successfully")
        
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return
    
    # Extract predictions and confidence scores
    print("🔍 Extracting model predictions and confidence scores...")
    try:
        results = analyzer.extract_predictions_and_confidence(test_loader, max_samples=1000)
    except Exception as e:
        print(f"❌ Failed to extract predictions: {e}")
        return
    
    # Compute coverage-accuracy curve
    print("📈 Computing coverage-accuracy curve...")
    try:
        curve_data = analyzer.compute_coverage_accuracy_curve(results)
    except Exception as e:
        print(f"❌ Failed to compute curve: {e}")
        return
    
    # Find optimal thresholds
    print("🎯 Finding optimal confidence thresholds...")
    try:
        optimal_thresholds = analyzer.find_optimal_thresholds(curve_data)
    except Exception as e:
        print(f"❌ Failed to find optimal thresholds: {e}")
        return
    
    # Create visualizations
    print("📊 Creating coverage-accuracy visualizations...")
    try:
        plot_path = analyzer.create_coverage_accuracy_plots(curve_data, optimal_thresholds)
    except Exception as e:
        print(f"❌ Failed to create plots: {e}")
        return
    
    # Generate report
    print("📄 Generating coverage-accuracy report...")
    try:
        generate_coverage_accuracy_report(curve_data, optimal_thresholds)
    except Exception as e:
        print(f"❌ Failed to generate report: {e}")
    
    # Summary
    print(f"\n📋 COVERAGE-ACCURACY ANALYSIS COMPLETE")
    print("=" * 50)
    print("✅ Selective prediction analysis completed")
    print("📊 Visualizations saved to /content/drive/MyDrive/hateful_memes_cnn/results/coverage_accuracy_analysis.png")
    print("📄 Report saved to /content/drive/MyDrive/hateful_memes_cnn/results/coverage_accuracy_report.md")
    print("🎯 This addresses the 'Selective Prediction' requirement")
    
    print(f"\n💡 Key Insights:")
    best_balanced = optimal_thresholds.get('balanced', {})
    if best_balanced:
        print(f"   - Balanced threshold: {best_balanced.get('threshold', 0):.3f}")
        print(f"   - Achieves {best_balanced.get('accuracy', 0):.1%} accuracy on {best_balanced.get('coverage', 0):.1%} of samples")
        print(f"   - Improvement: {(best_balanced.get('accuracy', 0) - results['correct'].mean()) * 100:.1f}% accuracy gain")
    
    print(f"\n🚀 Next Steps:")
    print("   - Use optimal thresholds for deployment decisions")
    print("   - Implement human-in-the-loop for low-confidence predictions")
    print("   - Consider ensemble methods for rejected samples")
    print("   - Include analysis in academic report")

if __name__ == "__main__":
    run_coverage_accuracy_analysis()