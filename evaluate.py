"""
STEP 4: evaluate.py
Save this file as: evaluate.py
Description: Model evaluation and performance metrics
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_curve, 
    auc, 
    precision_recall_curve
)
from tensorflow import keras
import pandas as pd
import os

class ModelEvaluator:
    def __init__(self, model, class_names):
        """Initialize evaluator"""
        self.model = model
        self.class_names = class_names
        self.y_true = None
        self.y_pred = None
        self.y_pred_proba = None
    
    def evaluate_on_generator(self, test_generator):
        """Evaluate model on test data"""
        print("\n" + "="*70)
        print("EVALUATING MODEL ON TEST SET")
        print("="*70)
        
        # Get predictions
        print("\nGenerating predictions...")
        self.y_pred_proba = self.model.predict(test_generator, verbose=1)
        self.y_pred = np.argmax(self.y_pred_proba, axis=1)
        self.y_true = test_generator.classes
        
        # Calculate metrics
        print("\nCalculating metrics...")
        test_loss, test_accuracy = self.model.evaluate(test_generator, verbose=0)
        
        print("\n" + "="*70)
        print("MODEL EVALUATION RESULTS")
        print("="*70)
        print(f"Test Loss:     {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print("="*70 + "\n")
        
        return test_loss, test_accuracy
    
    def print_classification_report(self):
        """Print detailed classification report"""
        report = classification_report(
            self.y_true, 
            self.y_pred, 
            target_names=self.class_names,
            digits=4
        )
        
        print("\n" + "="*70)
        print("CLASSIFICATION REPORT")
        print("="*70)
        print(report)
        print("="*70 + "\n")
        
        return report
    
    def plot_confusion_matrix(self, save_path='confusion_matrix.png'):
        """Plot confusion matrix"""
        print("Creating confusion matrix...")
        
        cm = confusion_matrix(self.y_true, self.y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def plot_normalized_confusion_matrix(self, save_path='confusion_matrix_normalized.png'):
        """Plot normalized confusion matrix"""
        print("Creating normalized confusion matrix...")
        
        cm = confusion_matrix(self.y_true, self.y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm_normalized, 
            annot=True, 
            fmt='.2%', 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Percentage'}
        )
        plt.title('Normalized Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def plot_roc_curves(self, save_path='roc_curves.png'):
        """Plot ROC curves for all classes"""
        print("Creating ROC curves...")
        
        n_classes = len(self.class_names)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        # Convert to one-hot
        y_true_bin = np.eye(n_classes)[self.y_true]
        
        # Calculate ROC for each class
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], self.y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plot
        plt.figure(figsize=(12, 8))
        colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
        
        for i, color in zip(range(n_classes), colors):
            plt.plot(
                fpr[i], 
                tpr[i], 
                color=color, 
                lw=2,
                label=f'{self.class_names[i]} (AUC = {roc_auc[i]:.3f})'
            )
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=9)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def plot_precision_recall_curves(self, save_path='precision_recall_curves.png'):
        """Plot Precision-Recall curves"""
        print("Creating Precision-Recall curves...")
        
        n_classes = len(self.class_names)
        y_true_bin = np.eye(n_classes)[self.y_true]
        
        plt.figure(figsize=(12, 8))
        colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
        
        for i, color in zip(range(n_classes), colors):
            precision, recall, _ = precision_recall_curve(
                y_true_bin[:, i], 
                self.y_pred_proba[:, i]
            )
            plt.plot(
                recall, 
                precision, 
                color=color, 
                lw=2,
                label=self.class_names[i]
            )
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves', fontsize=16, fontweight='bold')
        plt.legend(loc="lower left", fontsize=9)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def analyze_per_class_performance(self):
        """Analyze per-class performance"""
        print("\nAnalyzing per-class performance...")
        
        report_dict = classification_report(
            self.y_true, 
            self.y_pred, 
            target_names=self.class_names,
            output_dict=True
        )
        
        df = pd.DataFrame(report_dict).transpose()
        df = df.iloc[:-3]  # Remove summary rows
        
        print("\n" + "="*70)
        print("PER-CLASS PERFORMANCE")
        print("="*70)
        print(df.to_string())
        print("="*70 + "\n")
        
        # Save to CSV
        df.to_csv('per_class_performance.csv')
        print("✓ Saved: per_class_performance.csv\n")
        
        return df
    
    def generate_evaluation_report(self, output_dir='evaluation_results'):
        """Generate complete evaluation report"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*70)
        print("GENERATING COMPREHENSIVE EVALUATION REPORT")
        print("="*70 + "\n")
        
        # Classification report
        self.print_classification_report()
        
        # Create visualizations
        print("Creating visualizations...")
        self.plot_confusion_matrix(f'{output_dir}/confusion_matrix.png')
        self.plot_normalized_confusion_matrix(f'{output_dir}/confusion_matrix_normalized.png')
        self.plot_roc_curves(f'{output_dir}/roc_curves.png')
        self.plot_precision_recall_curves(f'{output_dir}/precision_recall_curves.png')
        
        # Per-class analysis
        df = self.analyze_per_class_performance()
        
        print("\n" + "="*70)
        print(f"✓ All results saved to: {output_dir}/")
        print("="*70 + "\n")
        
        return df


# Main execution
if __name__ == "__main__":
    print("\n" + "="*70)
    print("VITAMIN DEFICIENCY DETECTION - MODEL EVALUATION")
    print("="*70)
    
    # Load model
    print("\n[STEP 1/5] Loading trained model...")
    MODEL_PATH = 'vitamin_deficiency_model_final.h5'
    import os
    import tensorflow as tf

    MODEL_PATH = "vitamin_deficiency_model_final.h5"

    if not os.path.exists(MODEL_PATH):
         print("✗ ERROR: Model not found at:", MODEL_PATH)
         print("Please train the model first using train_model.py")
         exit()

    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Class names
    class_names = [
        'Vitamin_A_Deficiency',
        'Vitamin_B_Deficiency',
        'Vitamin_C_Deficiency',
        'Vitamin_D_Deficiency',
        'Vitamin_E_Deficiency',
        'Normal_Skin'
    ]
    
    # Load test data
    print("\n[STEP 2/5] Loading test dataset...")
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    TEST_DIR = 'data/processed/test'
    
    if not os.path.exists(TEST_DIR):
        print(f"\n✗ ERROR: Test data not found at: {TEST_DIR}")
        print("Please run data preprocessing first:")
        print("  python data_preprocessing.py")
        exit(1)
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    
    print(f"✓ Test dataset loaded: {test_generator.samples} images")
    
    # Initialize evaluator
    print("\n[STEP 3/5] Initializing evaluator...")
    evaluator = ModelEvaluator(model, class_names)
    print("✓ Evaluator initialized")
    
    # Evaluate model
    print("\n[STEP 4/5] Evaluating model...")
    test_loss, test_accuracy = evaluator.evaluate_on_generator(test_generator)
    
    # Generate report
    print("\n[STEP 5/5] Generating evaluation report...")
    evaluator.generate_evaluation_report('evaluation_results')
    
    # Final summary
    print("\n" + "="*70)
    print("✓ EVALUATION COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nFinal Results:")
    print(f"  Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"  Test Loss:     {test_loss:.4f}")
    print(f"\n  Results saved in: evaluation_results/")
    print("="*70 + "\n")
    