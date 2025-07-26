import sys
sys.path.append('./src')

def run_cnn_training():
    print("🚀 STARTING CNN TRAINING PIPELINE")
    print("=" * 60)
    
    # Load data
    print("📦 Loading dataset...")
    try:
        from data_preparation import setup_data_loaders
        
        train_loader, val_loader, test_loader, dataset_info = setup_data_loaders(
            data_dir="./data/hateful_memes_expanded",
            batch_size=32,
            val_size=0.2,
            num_workers=2,
            random_state=42
        )
        
        print("✅ Data loaded successfully!")
        print(f"   Training samples: {dataset_info['train_samples']}")
        print(f"   Validation samples: {dataset_info['val_samples']}")
        print(f"   Test samples: {dataset_info['test_samples']}")
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return
    
    # Train CNN model
    print("\n🤖 Training CNN model...")
    try:
        from cnn_model import main_experiment
        
        results = main_experiment(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            lr=1e-4,        # Learning rate
            epochs=25,      # Training epochs
            dropout_rate=0.5
        )
        
        print("\n🎊 TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"🏆 Best Validation AUC: {results['best_val_auc']:.4f}")
        print(f"📊 Test Accuracy: {results['test_results']['accuracy']:.4f}")
        print(f"📊 Test AUC-ROC: {results['test_results']['auc_roc']:.4f}")
        print(f"📊 Test F1-Score: {results['test_results']['f1_score']:.4f}")
        
        # Results for academic report
        print(f"\n📄 FOR ACADEMIC REPORT:")
        print(f"   • CNN Architecture: ResNet-50 with transfer learning")
        print(f"   • Dataset: 10,000 images ")
        print(f"   • Image-only accuracy: {results['test_results']['accuracy']*100:.1f}%")
        print(f"   • AUC-ROC score: {results['test_results']['auc_roc']:.3f}")
        print(f"   • F1-score: {results['test_results']['f1_score']:.3f}")
        print(f"   • Model saved: ./models/best_model.pth")
        print(f"   • Visualizations: ./results/confusion_matrix.png")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return
    
    print(f"\n✅ SUCCESS! CNN component is ready!")
    print(f"📂 Check ./models/ for saved model")
    print(f"📊 Check ./results/ for plots and analysis")

if __name__ == "__main__":
    run_cnn_training()