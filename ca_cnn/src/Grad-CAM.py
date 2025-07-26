

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import json
import sys

# Add src directory to Python path for Colab
sys.path.append('/content/drive/MyDrive/hateful_memes_cnn/src')

class GradCAM:
    
    def __init__(self, model, target_layer_name=None):

        self.model = model
        self.model.eval()  # Set to evaluation mode
        
        # Storage for gradients and activations
        self.gradients = None
        self.activations = None
        
        # Find the target layer automatically
        if target_layer_name is None:
            target_layer_name = self._find_target_layer()
        
        self.target_layer_name = target_layer_name
        self._register_hooks()
        
        print(f"✅ Grad-CAM initialized for layer: {target_layer_name}")
    
    def _find_target_layer(self):
        """
        Automatically find the best layer for Grad-CAM visualization
        
        Returns:
            str: Name of the target convolutional layer
        """
        # For ResNet models, use the last convolutional layer
        for name, module in self.model.named_modules():
            if 'layer4' in name and 'conv2' in name:
                return name
            elif 'features' in name and isinstance(module, nn.Conv2d):
                target = name  # Keep updating to get the last one
        
        # Fallback: find any convolutional layer
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                target = name
        
        return target if 'target' in locals() else 'backbone.layer4.2.conv2'
    
    def _register_hooks(self):
        """
        Register forward and backward hooks to capture gradients and activations
        """
        def forward_hook(module, input, output):
            """Save activations during forward pass"""
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            """Save gradients during backward pass"""
            self.gradients = grad_output[0]
        
        # Find and register hooks on target layer
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                module.register_forward_hook(forward_hook)
                module.register_full_backward_hook(backward_hook)
                print(f"🔗 Hooks registered on layer: {name}")
                break
    
    def generate_cam(self, input_image, target_class=None):
        """
        Generate Grad-CAM heatmap for an input image
        
        Args:
            input_image: Preprocessed image tensor (1, 3, 224, 224)
            target_class: Class to generate CAM for (None = predicted class)
            
        Returns:
            np.array: Grad-CAM heatmap as numpy array
        """
        # Ensure input is on the correct device
        device = next(self.model.parameters()).device
        input_image = input_image.to(device)
        
        # Clear previous gradients
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_image)
        
        # Get target class (predicted class if not specified)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass for target class
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1.0
        
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Generate Grad-CAM
        gradients = self.gradients.cpu().data.numpy()[0]  # (channels, height, width)
        activations = self.activations.cpu().data.numpy()[0]  # (channels, height, width)
        
        # Calculate importance weights
        weights = np.mean(gradients, axis=(1, 2))  # Global average pooling
        
        # Weighted combination of activation maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, weight in enumerate(weights):
            cam += weight * activations[i]
        
        # Apply ReLU and normalize
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam, target_class
    
    def create_heatmap_overlay(self, original_image, cam, alpha=0.4):
        """
        Create heatmap overlay on original image
        
        Args:
            original_image: PIL Image or numpy array
            cam: Grad-CAM heatmap
            alpha: Transparency of heatmap overlay
            
        Returns:
            np.array: Image with heatmap overlay
        """
        # Convert PIL to numpy if needed
        if isinstance(original_image, Image.Image):
            original_image = np.array(original_image)
        
        # Resize CAM to match image size
        if original_image.shape[:2] != cam.shape:
            cam_resized = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
        else:
            cam_resized = cam
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay heatmap on original image
        overlay = heatmap * alpha + original_image * (1 - alpha)
        return overlay.astype(np.uint8)

def load_model_for_gradcam(model_path, model_type='basic'):

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

def analyze_sample_with_gradcam(model, image_path, gradcam_analyzer, save_dir="/content/drive/MyDrive/hateful_memes_cnn/results/gradcam"):
    """
    Analyze a single image sample with Grad-CAM
    
    Args:
        model: Trained CNN model
        image_path: Path to image file
        gradcam_analyzer: GradCAM instance
        save_dir: Directory to save results
        
    Returns:
        dict: Analysis results with prediction and confidence
    """
    # Create save directory - ensure it exists in Colab
    colab_results = '/content/drive/MyDrive/hateful_memes_cnn/results'
    os.makedirs(colab_results, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    
    # Load and preprocess image
    from data_preparation import get_image_transforms
    _, test_transform = get_image_transforms()
    
    # Load original image
    original_image = Image.open(image_path).convert('RGB')
    
    # Apply preprocessing
    input_tensor = test_transform(original_image).unsqueeze(0)
    
    # Get model prediction
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        predicted_class = outputs.argmax(dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    # Generate Grad-CAM
    cam, target_class = gradcam_analyzer.generate_cam(input_tensor, target_class=predicted_class)
    
    # Create visualization
    original_np = np.array(original_image.resize((224, 224)))
    heatmap_overlay = gradcam_analyzer.create_heatmap_overlay(original_np, cam)
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_np)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Grad-CAM heatmap only
    axes[1].imshow(cam, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(heatmap_overlay)
    class_name = "Hateful" if predicted_class == 1 else "Non-hateful"
    color = 'red' if predicted_class == 1 else 'green'
    axes[2].set_title(f'Prediction: {class_name}\nConfidence: {confidence:.3f}', 
                     fontsize=12, fontweight='bold', color=color)
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    sample_name = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(save_dir, f'gradcam_{sample_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'image_path': image_path,
        'predicted_class': predicted_class,
        'confidence': confidence,
        'class_name': class_name,
        'gradcam_saved': save_path
    }

def create_gradcam_gallery(model, test_loader, gradcam_analyzer, num_samples=16, save_path="/content/drive/MyDrive/hateful_memes_cnn/results/gradcam_gallery.png"):
    """
    Create a gallery of Grad-CAM visualizations for multiple samples
    
    Args:
        model: Trained CNN model
        test_loader: Test data loader
        gradcam_analyzer: GradCAM instance
        num_samples: Number of samples to visualize
        save_path: Path to save gallery image
    """
    device = next(model.parameters()).device
    
    # Collect samples from both classes
    hateful_samples = []
    nonhateful_samples = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            
            # Collect correctly predicted samples
            correct_mask = (predictions == labels)
            
            for i in range(len(images)):
                if not correct_mask[i]:
                    continue
                    
                if labels[i] == 1 and len(hateful_samples) < num_samples // 2:
                    hateful_samples.append((images[i:i+1], labels[i].item(), outputs[i:i+1]))
                elif labels[i] == 0 and len(nonhateful_samples) < num_samples // 2:
                    nonhateful_samples.append((images[i:i+1], labels[i].item(), outputs[i:i+1]))
                
                if len(hateful_samples) >= num_samples // 2 and len(nonhateful_samples) >= num_samples // 2:
                    break
            
            if len(hateful_samples) >= num_samples // 2 and len(nonhateful_samples) >= num_samples // 2:
                break
    
    # Create gallery visualization
    all_samples = hateful_samples + nonhateful_samples
    
    # Calculate grid dimensions
    n_cols = 4
    n_rows = (len(all_samples) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Denormalization parameters
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    
    for idx, (image_tensor, true_label, output) in enumerate(all_samples):
        if idx >= n_rows * n_cols:
            break
            
        row = idx // n_cols
        col = idx % n_cols
        
        # Generate Grad-CAM
        cam, predicted_class = gradcam_analyzer.generate_cam(image_tensor, target_class=true_label)
        
        # Denormalize image
        image_denorm = image_tensor[0].clone()
        for t, m, s in zip(image_denorm, mean, std):
            t.mul_(s).add_(m)
        image_denorm = torch.clamp(image_denorm, 0, 1)
        image_np = image_denorm.permute(1, 2, 0).cpu().numpy()
        
        # Create overlay
        image_resized = (image_np * 255).astype(np.uint8)
        heatmap_overlay = gradcam_analyzer.create_heatmap_overlay(image_resized, cam, alpha=0.4)
        
        # Plot
        axes[row, col].imshow(heatmap_overlay)
        
        # Get confidence
        probabilities = F.softmax(output, dim=1)
        confidence = probabilities[0][true_label].item()
        
        # Title
        class_name = "Hateful" if true_label == 1 else "Non-hateful"
        color = 'red' if true_label == 1 else 'green'
        axes[row, col].set_title(f'{class_name}\nConf: {confidence:.2f}', 
                                color=color, fontweight='bold', fontsize=10)
        axes[row, col].axis('off')
    
    # Hide unused subplots
    for idx in range(len(all_samples), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.suptitle('Grad-CAM Analysis Gallery: Model Focus Areas', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save gallery - ensure results directory exists
    colab_results = '/content/drive/MyDrive/hateful_memes_cnn/results'
    os.makedirs(colab_results, exist_ok=True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Grad-CAM gallery saved to: {save_path}")
    
    return {
        'gallery_path': save_path,
        'samples_analyzed': len(all_samples),
        'hateful_samples': len(hateful_samples),
        'nonhateful_samples': len(nonhateful_samples)
    }

def run_gradcam_analysis():
    """
    Main function to run complete Grad-CAM analysis
    """
    print("🔍 GRAD-CAM VISUAL ANALYSIS")
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
        model = load_model_for_gradcam(MODEL_PATH, MODEL_TYPE)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Initialize Grad-CAM
    print("🔧 Initializing Grad-CAM analyzer...")
    gradcam_analyzer = GradCAM(model)
    
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
    
    # Create Grad-CAM gallery
    print("🎨 Creating Grad-CAM gallery...")
    try:
        gallery_results = create_gradcam_gallery(
            model=model,
            test_loader=test_loader,
            gradcam_analyzer=gradcam_analyzer,
            num_samples=16
        )
        
        print(f"✅ Gallery created with {gallery_results['samples_analyzed']} samples")
        print(f"   - Hateful samples: {gallery_results['hateful_samples']}")
        print(f"   - Non-hateful samples: {gallery_results['nonhateful_samples']}")
        
    except Exception as e:
        print(f"❌ Failed to create gallery: {e}")
    
    # Summary
    print(f"\n📋 GRAD-CAM ANALYSIS COMPLETE")
    print("=" * 50)
    print("✅ Visual interpretability analysis completed")
    print("📊 Results saved in /content/drive/MyDrive/hateful_memes_cnn/results/gradcam_gallery.png")
    print("🎯 This addresses the 'Visual Interpretability' requirement")
    print("📄 Use these visualizations in academic report")
    print("\n💡 Key insights from Grad-CAM:")
    print("   - Shows which parts of memes model focuses on")
    print("   - Helps validate that model looks at relevant regions")
    print("   - Enables discussion of model interpretability")
    print("   - Supports claims about model attention patterns")

if __name__ == "__main__":
    run_gradcam_analysis()