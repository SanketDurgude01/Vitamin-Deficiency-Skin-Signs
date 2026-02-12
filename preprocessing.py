"""
STEP 2: data_preprocessing.py
Save this file as: data_preprocessing.py
Description: Data preprocessing and preparation for training
"""

import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import shutil

class DataPreprocessor:
    def __init__(self, img_size=224):
        """Initialize data preprocessor"""
        self.img_size = img_size
        self.categories = [
            'Vitamin_A_Deficiency',
            'Vitamin_B_Deficiency',
            'Vitamin_C_Deficiency',
            'Vitamin_D_Deficiency',
            'Vitamin_E_Deficiency',
            'Normal_Skin'
        ]
    
    def create_directory_structure(self, base_path='data'):
        """Create organized directory structure"""
        print("\n" + "="*70)
        print("Creating directory structure...")
        print("="*70)
        
        directories = [
            f'{base_path}/raw',
            f'{base_path}/processed/train',
            f'{base_path}/processed/validation',
            f'{base_path}/processed/test'
        ]
        
        for directory in directories:
            for category in self.categories:
                path = os.path.join(directory, category)
                os.makedirs(path, exist_ok=True)
        
        print(f"✓ Directory structure created at: {base_path}")
        print(f"✓ Created {len(self.categories)} categories")
        print("="*70)
    
    def preprocess_image(self, img_path, save_path=None):
        """Preprocess a single image"""
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            print(f"✗ Error reading: {img_path}")
            return None
        
        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize
        img = cv2.resize(img, (self.img_size, self.img_size))
        
        # Enhance contrast using CLAHE
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Denoise
        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        return img
    
    def batch_preprocess(self, input_dir, output_dir):
        """Preprocess all images in directory"""
        print("\n" + "="*70)
        print("Batch Processing Images...")
        print("="*70)
        
        processed_count = 0
        error_count = 0
        
        for category in self.categories:
            input_path = os.path.join(input_dir, category)
            output_path = os.path.join(output_dir, category)
            
            if not os.path.exists(input_path):
                print(f"⚠ Category not found: {category}")
                continue
            
            os.makedirs(output_path, exist_ok=True)
            
            # Get all images
            image_files = [f for f in os.listdir(input_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            print(f"\n{category}:")
            print(f"  Found: {len(image_files)} images")
            
            # Process each image
            for img_file in image_files:
                input_file = os.path.join(input_path, img_file)
                output_file = os.path.join(output_path, img_file)
                
                img = self.preprocess_image(input_file, output_file)
                if img is not None:
                    processed_count += 1
                else:
                    error_count += 1
            
            print(f"  Processed: {len(image_files)} images")
        
        print("\n" + "="*70)
        print(f"✓ Successfully processed: {processed_count} images")
        if error_count > 0:
            print(f"✗ Failed to process: {error_count} images")
        print("="*70)
    
    def split_dataset(self, source_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """Split dataset into train, validation, and test sets"""
        print("\n" + "="*70)
        print("Splitting Dataset...")
        print("="*70)
        print(f"Train: {train_ratio*100:.0f}% | Validation: {val_ratio*100:.0f}% | Test: {test_ratio*100:.0f}%")
        print("-"*70)
        
        for category in self.categories:
            category_path = os.path.join(source_dir, category)
            
            if not os.path.exists(category_path):
                print(f"⚠ Skipping {category} - not found")
                continue
            
            # Get all images
            images = [f for f in os.listdir(category_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if len(images) == 0:
                print(f"⚠ No images in {category}")
                continue
            
            # Split data
            train_imgs, temp_imgs = train_test_split(
                images, test_size=(1-train_ratio), random_state=42
            )
            val_imgs, test_imgs = train_test_split(
                temp_imgs, 
                test_size=(test_ratio/(val_ratio+test_ratio)), 
                random_state=42
            )
            
            # Copy files to respective directories
            splits = {
                'train': train_imgs,
                'validation': val_imgs,
                'test': test_imgs
            }
            
            for split_name, split_imgs in splits.items():
                dest_dir = os.path.join('data/processed', split_name, category)
                os.makedirs(dest_dir, exist_ok=True)
                
                for img_name in split_imgs:
                    src = os.path.join(category_path, img_name)
                    dst = os.path.join(dest_dir, img_name)
                    shutil.copy2(src, dst)
            
            # Print statistics
            print(f"{category:30s} → Train:{len(train_imgs):4d} | Val:{len(val_imgs):4d} | Test:{len(test_imgs):4d}")
        
        print("="*70)
    
    def visualize_samples(self, data_dir, samples_per_class=3, save_path='dataset_samples.png'):
        """Create visualization of sample images"""
        print("\n" + "="*70)
        print("Creating Sample Visualization...")
        print("="*70)
        
        fig, axes = plt.subplots(
            len(self.categories), 
            samples_per_class, 
            figsize=(15, 3*len(self.categories))
        )
        
        for i, category in enumerate(self.categories):
            category_path = os.path.join(data_dir, category)
            
            if not os.path.exists(category_path):
                continue
            
            images = [f for f in os.listdir(category_path)[:samples_per_class]
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for j, img_file in enumerate(images):
                img_path = os.path.join(category_path, img_file)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                ax = axes[i, j] if len(self.categories) > 1 else axes[j]
                ax.imshow(img)
                ax.axis('off')
                
                if j == 0:
                    ax.set_title(
                        category.replace('_', ' '), 
                        fontsize=10, 
                        fontweight='bold'
                    )
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved: {save_path}")
        print("="*70)
        plt.close()
    
    def analyze_dataset(self, data_dir):
        """Analyze and print dataset statistics"""
        stats = {}
        total_images = 0
        
        print("\n" + "="*70)
        print("DATASET STATISTICS")
        print("="*70)
        
        for category in self.categories:
            category_path = os.path.join(data_dir, category)
            
            if not os.path.exists(category_path):
                stats[category] = 0
                continue
            
            count = len([f for f in os.listdir(category_path)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            stats[category] = count
            total_images += count
            
            print(f"{category:35s}: {count:5d} images")
        
        print("-"*70)
        print(f"{'TOTAL':35s}: {total_images:5d} images")
        print("="*70)
        
        return stats
    
    def remove_corrupted_images(self, data_dir):
        """Remove corrupted or invalid images"""
        print("\n" + "="*70)
        print("Checking for Corrupted Images...")
        print("="*70)
        
        removed_count = 0
        
        for category in self.categories:
            category_path = os.path.join(data_dir, category)
            
            if not os.path.exists(category_path):
                continue
            
            for img_file in os.listdir(category_path):
                img_path = os.path.join(category_path, img_file)
                
                try:
                    img = Image.open(img_path)
                    img.verify()
                except Exception as e:
                    print(f"✗ Removing corrupted: {img_path}")
                    os.remove(img_path)
                    removed_count += 1
        
        if removed_count > 0:
            print(f"✓ Removed {removed_count} corrupted images")
        else:
            print("✓ No corrupted images found")
        print("="*70)


# Main execution
if __name__ == "__main__":
    print("\n" + "="*70)
    print("VITAMIN DEFICIENCY DETECTION - DATA PREPROCESSING")
    print("="*70)
    
    preprocessor = DataPreprocessor(img_size=224)
    
    # Step 1: Create directory structure
    print("\n[STEP 1/7] Creating directory structure...")
    preprocessor.create_directory_structure('data')
    
    # Step 2: Analyze raw dataset
    print("\n[STEP 2/7] Analyzing raw dataset...")
    preprocessor.analyze_dataset('data/raw')
    
    # Step 3: Remove corrupted images
    print("\n[STEP 3/7] Removing corrupted images...")
    preprocessor.remove_corrupted_images('data/raw')
    
    # Step 4: Batch preprocess images
    print("\n[STEP 4/7] Preprocessing images...")
    preprocessor.batch_preprocess('data/raw', 'data/processed_temp')
    
    # Step 5: Split dataset
    print("\n[STEP 5/7] Splitting dataset...")
    preprocessor.split_dataset('data/processed_temp')
    
    # Step 6: Create visualizations
    print("\n[STEP 6/7] Creating visualizations...")
    preprocessor.visualize_samples('data/processed/train', samples_per_class=3)
    
    # Step 7: Final statistics
    print("\n[STEP 7/7] Final dataset statistics:")
    print("\nTrain Set:")
    preprocessor.analyze_dataset('data/processed/train')
    print("\nValidation Set:")
    preprocessor.analyze_dataset('data/processed/validation')
    print("\nTest Set:")
    preprocessor.analyze_dataset('data/processed/test')
    
    print("\n" + "="*70)
    print("✓ DATA PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nNext step: Train the model using model.py")
    print("="*70 + "\n")