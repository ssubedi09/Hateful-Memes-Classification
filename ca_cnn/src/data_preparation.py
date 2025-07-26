import json
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class HatefulMemesImageDataset(Dataset):
    """
    Fixed Dataset class for Hateful Memes (Image-only) with corrected path handling
    """
    
    def __init__(self, jsonl_file, img_dir, transform=None, max_samples=None):
        """
        Args:
            jsonl_file (str): Path to the .jsonl file containing metadata
            img_dir (str): Directory containing the images  
            transform (callable, optional): Optional transform to be applied on images
            max_samples (int, optional): Limit number of samples for testing
        """
        self.img_dir = img_dir
        self.transform = transform
        self.data = []
        self.missing_images = []
        
        print(f"Loading data from {jsonl_file}...")
        print(f"Image directory: {img_dir}")
        
        # Load data from JSONL file
        with open(jsonl_file, 'r') as f:
            for line_num, line in enumerate(f):
                if max_samples and line_num >= max_samples:
                    break
                    
                try:
                    item = json.loads(line.strip())
                    
                    # Try different path construction methods to find images
                    img_path = self._find_image_path(item)
                    
                    if img_path and os.path.exists(img_path):
                        # Update the item with the correct path
                        item['img_full_path'] = img_path
                        self.data.append(item)
                    else:
                        self.missing_images.append(item.get('img', 'unknown'))
                        
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_num}: {e}")
                    continue
        
        print(f"✅ Loaded {len(self.data)} valid samples")
        if self.missing_images:
            print(f"⚠️  Warning: {len(self.missing_images)} images not found")
            if len(self.missing_images) <= 5:
                print(f"Missing images: {self.missing_images}")
    
    def _find_image_path(self, item):
        """
        Try different methods to construct the correct image path
        
        Args:
            item (dict): JSON item containing image reference
            
        Returns:
            str: Valid image path if found, None otherwise
        """
        img_ref = item.get('img', '')
        sample_id = item.get('id', '')
        
        # Try different path construction methods
        possible_paths = [
            # Method 1: Use the path as stored in JSON (relative to dataset root)
            os.path.join(os.path.dirname(self.img_dir), img_ref),
            
            # Method 2: Use just the filename in the img directory
            os.path.join(self.img_dir, os.path.basename(img_ref)),
            
            # Method 3: Remove 'img/' prefix and use in img directory  
            os.path.join(self.img_dir, img_ref.replace('img/', '')),
            
            # Method 4: Construct from ID directly
            os.path.join(self.img_dir, f"{sample_id}.png"),
            
            # Method 5: Try with different extensions
            os.path.join(self.img_dir, f"{sample_id}.jpg"),
        ]
        
        # Return the first path that exists
        for path in possible_paths:
            if path and os.path.exists(path):
                return path
        
        return None
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns:
            tuple: (image, label) where image is processed tensor and label is int
        """
        if idx >= len(self.data):
            # Return dummy data for out-of-range indices
            dummy_image = torch.zeros(3, 224, 224) if self.transform else Image.new('RGB', (224, 224))
            dummy_label = torch.tensor(0, dtype=torch.long)
            return dummy_image, dummy_label
        
        try:
            item = self.data[idx]
            
            # Load image using the verified path
            img_path = item['img_full_path']
            image = Image.open(img_path)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            
            # Get label (0: non-hateful, 1: hateful)
            label = torch.tensor(item['label'], dtype=torch.long)
            
            return image, label
            
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            # Return dummy data if there's an error
            dummy_image = torch.zeros(3, 224, 224) if self.transform else Image.new('RGB', (224, 224))
            dummy_label = torch.tensor(0, dtype=torch.long)
            return dummy_image, dummy_label

def verify_dataset_structure(data_dir="./data/hateful_memes_expanded"):
    """
    Verify the dataset structure and find the correct image path pattern
    """
    print("🔍 VERIFYING DATASET STRUCTURE")
    print("=" * 60)
    
    # Check directories
    img_dir = os.path.join(data_dir, "img")
    train_file = os.path.join(data_dir, "train.jsonl")
    
    print(f"Dataset directory: {data_dir}")
    print(f"Image directory: {img_dir}")
    print(f"Train file: {train_file}")
    
    # Check if directories exist
    if not os.path.exists(data_dir):
        print(f"❌ Dataset directory not found: {data_dir}")
        return False
    
    if not os.path.exists(img_dir):
        print(f"❌ Image directory not found: {img_dir}")
        return False
        
    if not os.path.exists(train_file):
        print(f"❌ Train file not found: {train_file}")
        return False
    
    # Count images
    img_files = [f for f in os.listdir(img_dir) if f.endswith('.png') or f.endswith('.jpg')]
    print(f"✅ Found {len(img_files)} image files")
    
    # Test path construction with a few samples
    print(f"\n🧪 Testing path construction...")
    with open(train_file, 'r') as f:
        test_samples = []
        for i, line in enumerate(f):
            if i >= 5:  # Test with first 5 samples
                break
            test_samples.append(json.loads(line.strip()))
    
    success_count = 0
    for i, sample in enumerate(test_samples):
        img_ref = sample.get('img', '')
        sample_id = sample.get('id', '')
        
        # Try to find the image
        temp_dataset = HatefulMemesImageDataset.__new__(HatefulMemesImageDataset)
        temp_dataset.img_dir = img_dir
        img_path = temp_dataset._find_image_path(sample)
        
        if img_path:
            success_count += 1
            print(f"   ✅ Sample {i}: {img_ref} -> {os.path.basename(img_path)}")
        else:
            print(f"   ❌ Sample {i}: {img_ref} -> NOT FOUND")
    
    success_rate = success_count / len(test_samples) * 100
    print(f"\n📊 Path resolution success rate: {success_rate:.1f}%")
    
    return success_rate > 80  # Consider successful if >80% of paths resolve

def get_image_transforms():
    """
    Define image preprocessing transforms for training and testing.
    """
    
    # Training transforms with data augmentation
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Testing transforms (no augmentation)
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, test_transform

def create_train_val_split(data_dir="./data/hateful_memes_expanded", val_size=0.2, random_state=42):
    """
    Create train/validation split with proper stratification
    UPDATED: Adds ALL dev_seen.jsonl samples to training (not split)
    """
    print(f"\n📊 Creating train/validation split (val_size={val_size})...")
    print("🎯 STRATEGY: Split train.jsonl only, then add ALL dev_seen.jsonl to training")
    
    # Load and verify training data
    train_file = os.path.join(data_dir, "train.jsonl")
    dev_file = os.path.join(data_dir, "dev_seen.jsonl")
    img_dir = os.path.join(data_dir, "img")
    
    # Create temporary dataset to verify data can be loaded
    train_transform, _ = get_image_transforms()
    temp_dataset = HatefulMemesImageDataset(train_file, img_dir, transform=None, max_samples=100)
    
    if len(temp_dataset) == 0:
        print("❌ No valid samples found in training data!")
        return None, None
    
    print(f"✅ Verified dataset loading works ({len(temp_dataset)} samples tested)")
    
    # Load training data
    full_train_data = []
    with open(train_file, 'r') as f:
        for line in f:
            full_train_data.append(json.loads(line.strip()))
    
    print(f"📁 Loaded {len(full_train_data)} samples from train.jsonl")
    
    # Extract labels for stratification from train.jsonl ONLY
    labels = [item['label'] for item in full_train_data]
    
    # Create stratified split from train.jsonl ONLY
    train_indices, val_indices = train_test_split(
        range(len(full_train_data)),
        test_size=val_size,
        random_state=random_state,
        stratify=labels
    )
    
    # Create initial splits from train.jsonl
    train_split = [full_train_data[i] for i in train_indices]
    val_split = [full_train_data[i] for i in val_indices]
    
    print(f"📊 Initial split from train.jsonl:")
    print(f"   Train: {len(train_split)} samples")
    print(f"   Validation: {len(val_split)} samples")
    
    # Load dev_seen data and add ALL to training
    if os.path.exists(dev_file):
        dev_data = []
        with open(dev_file, 'r') as f:
            for line in f:
                dev_data.append(json.loads(line.strip()))
        
        print(f"📁 Loaded {len(dev_data)} samples from dev_seen.jsonl")
        print(f"➕ Adding ALL {len(dev_data)} dev samples to training")
        
        # Add ALL dev samples to training split
        train_split.extend(dev_data)
        
    else:
        print(f"⚠️  Warning: {dev_file} not found, using only train.jsonl split")
        dev_data = []
    
    # Save splits to temporary files
    train_temp_file = os.path.join(data_dir, "train_split.jsonl")
    val_temp_file = os.path.join(data_dir, "val_split.jsonl")
    
    with open(train_temp_file, 'w') as f:
        for item in train_split:
            f.write(json.dumps(item) + '\n')
    
    with open(val_temp_file, 'w') as f:
        for item in val_split:
            f.write(json.dumps(item) + '\n')
    
    # Verify final splits
    train_labels = [item['label'] for item in train_split]
    val_labels = [item['label'] for item in val_split]
    
    train_counts = Counter(train_labels)
    val_counts = Counter(val_labels)
    
    print(f"\n🎯 FINAL SPLIT RESULTS:")
    print(f"Train split: {len(train_split)} samples")
    print(f"  Non-hateful: {train_counts[0]} ({train_counts[0]/len(train_split)*100:.2f}%)")
    print(f"  Hateful: {train_counts[1]} ({train_counts[1]/len(train_split)*100:.2f}%)")
    
    print(f"Validation split: {len(val_split)} samples (from train.jsonl only)")
    print(f"  Non-hateful: {val_counts[0]} ({val_counts[0]/len(val_split)*100:.2f}%)")
    print(f"  Hateful: {val_counts[1]} ({val_counts[1]/len(val_split)*100:.2f}%)")
    
    # Show improvement summary
    original_train_size = int(len(full_train_data) * (1 - val_size))
    new_train_size = len(train_split)
    dev_contribution = len(dev_data)
    
    print(f"\n📈 IMPROVEMENT SUMMARY:")
    print(f"   Original train samples: {original_train_size} (80% of train.jsonl)")
    print(f"   Dev samples added: +{dev_contribution}")
    print(f"   Final training samples: {new_train_size}")
    print(f"   Total improvement: +{dev_contribution} samples ({dev_contribution/original_train_size*100:.1f}% increase)")
    print(f"   Validation samples: {len(val_split)} (unchanged, from train.jsonl only)")
    
    return train_temp_file, val_temp_file

def setup_data_loaders(data_dir="./data/hateful_memes_expanded", 
                      batch_size=32, 
                      val_size=0.2, 
                      num_workers=2,
                      random_state=42):
    """
    Setup complete data loading pipeline with fixed paths
    """
    
    print("\n🔄 SETTING UP DATA LOADERS")
    print("=" * 60)
    
    # Verify dataset structure first
    if not verify_dataset_structure(data_dir):
        print("❌ Dataset structure verification failed!")
        return None, None, None, None
    
    # Get transforms
    train_transform, test_transform = get_image_transforms()
    
    # File paths
    img_dir = os.path.join(data_dir, "img")
    test_file = os.path.join(data_dir, "test_seen.jsonl")
    
    # Create train/val split
    train_temp_file, val_temp_file = create_train_val_split(data_dir, val_size, random_state)
    
    if not train_temp_file:
        return None, None, None, None
    
    # Create datasets with fixed path handling
    print(f"\n📦 Creating datasets...")
    train_dataset = HatefulMemesImageDataset(train_temp_file, img_dir, train_transform)
    val_dataset = HatefulMemesImageDataset(val_temp_file, img_dir, test_transform)
    test_dataset = HatefulMemesImageDataset(test_file, img_dir, test_transform)
    
    # Check if datasets loaded successfully
    if len(train_dataset) == 0:
        print("❌ Training dataset is empty!")
        return None, None, None, None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=True if num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=True if num_workers > 0 else False
    )
    
    # Dataset info
    dataset_info = {
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'test_samples': len(test_dataset),
        'train_batches': len(train_loader),
        'val_batches': len(val_loader),
        'test_batches': len(test_loader),
        'batch_size': batch_size
    }
    
    print(f"\n📊 DATA LOADER SUMMARY:")
    print(f"Training: {dataset_info['train_samples']} samples, {dataset_info['train_batches']} batches")
    print(f"Validation: {dataset_info['val_samples']} samples, {dataset_info['val_batches']} batches")
    print(f"Test: {dataset_info['test_samples']} samples, {dataset_info['test_batches']} batches")
    print(f"Batch size: {batch_size}")
    
    return train_loader, val_loader, test_loader, dataset_info

def visualize_samples(data_loader, num_samples=8, save_path="./results/sample_visualization.png"):
    """
    Visualize sample images from data loader
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Get a batch
    data_iter = iter(data_loader)
    images, labels = next(data_iter)
    
    # Denormalization parameters
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    
    # Create visualization
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    fig.suptitle('Sample Images from Hateful Memes Dataset', fontsize=16)
    
    for i in range(min(num_samples, len(images))):
        img = images[i].clone()
        
        # Denormalize
        for t, m, s in zip(img, mean, std):
            t.mul_(s).add_(m)
        img = torch.clamp(img, 0, 1)
        
        # Plot
        row = i // 4
        col = i % 4
        axes[row, col].imshow(img.permute(1, 2, 0))
        
        label_text = "Hateful" if labels[i].item() == 1 else "Non-hateful"
        color = 'red' if labels[i].item() == 1 else 'green'
        axes[row, col].set_title(f'{label_text}', color=color, fontweight='bold')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📸 Sample visualization saved to: {save_path}")
    plt.show()

def get_device():
    """Get the best available device for training"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def main():
    """Main function to run complete data preparation pipeline"""
    
    print("🚀 HATEFUL MEMES CNN - UPDATED DATA PREPARATION")
    print("🔄 Now includes dev_seen.jsonl samples for maximum data utilization!")
    print("=" * 70)
    
    # Configuration
    DATA_DIR = "./data/hateful_memes_expanded"
    BATCH_SIZE = 32
    VAL_SIZE = 0.2
    NUM_WORKERS = 2  # Reduced for Mac
    RANDOM_STATE = 42
    
    # Check device
    device = get_device()
    print(f"Using device: {device}")
    
    # Setup Data Loaders with fixed path handling
    train_loader, val_loader, test_loader, dataset_info = setup_data_loaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        val_size=VAL_SIZE,
        num_workers=NUM_WORKERS,
        random_state=RANDOM_STATE
    )
    
    if train_loader is None:
        print("❌ Failed to setup data loaders!")
        return None, None, None, None
    
    # Visualize Samples
    print(f"\n🖼️  VISUALIZING SAMPLE DATA")
    try:
        visualize_samples(train_loader)
    except Exception as e:
        print(f"Visualization failed: {e}")
    
    # Test Data Loading
    print(f"\n🧪 TESTING DATA LOADING")
    try:
        for batch_idx, (images, labels) in enumerate(train_loader):
            print(f"✅ Batch {batch_idx}: Images {images.shape}, Labels {labels.shape}")
            print(f"   Label distribution: {labels.unique(return_counts=True)}")
            break
        
        print("✅ Data loading test successful!")
        
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return None, None, None, None
    
    # Summary
    print("\n" + "="*70)
    print("📋 UPDATED DATA PREPARATION SUMMARY")
    print("="*70)
    print(f"✅ Dataset successfully loaded from: {DATA_DIR}")
    print(f"✅ Training samples: {dataset_info['train_samples']} (includes dev_seen.jsonl)")
    print(f"✅ Validation samples: {dataset_info['val_samples']}")
    print(f"✅ Test samples: {dataset_info['test_samples']}")
    print(f"✅ Batch size: {BATCH_SIZE}")
    print(f"✅ Device: {device}")
    print(f"🎯 Maximum data utilization achieved!")
    print(f"✅ Data loaders ready for CNN training!")
    
    return train_loader, val_loader, test_loader, dataset_info

if __name__ == "__main__":
    # Run the complete updated data preparation pipeline
    train_loader, val_loader, test_loader, dataset_info = main()