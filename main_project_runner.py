# main_project_runner.py
"""
Main script to orchestrate the Vitamin Deficiency Detection Project.
1. Data Preprocessing (Requires 'data/raw' directory with images)
2. Model Training (Requires split data)
3. Launch API Server (Requires trained model)
4. Evaluation (Requires trained model and test data)

NOTE: For actual execution, you must have your image dataset in the 'data/raw' directory 
and run 'app.py' in a separate terminal for the frontend to connect.
"""
import os
import subprocess
import time

# Placeholder for required modules (assuming all are in the current directory)
import preprocessing
from model import VitaminDeficiencyModel
import evaluate # For evaluation logic

# --- Configuration ---
DATA_RAW_DIR = 'data/raw'
TRAIN_DIR = 'data/processed/train'
VAL_DIR = 'data/processed/validation'
TEST_DIR = 'data/processed/test'
FINAL_MODEL_PATH = 'vitamin_deficiency_model_final.h5'

def setup_directories():
    """Initial directory setup."""
    print("Setting up data directory structure...")
    preprocessor = preprocessing.DataPreprocessor()
    preprocessor.create_directory_structure('data')
    # Cleanup temporary directory from preprocessing
    if os.path.exists('data/processed_temp'):
        import shutil
        shutil.rmtree('data/processed_temp')
    print("Directory structure setup complete.")
    
def run_data_pipeline():
    """Execute the data preprocessing script."""
    print("\n" + "="*70)
    print("1. STARTING DATA PREPROCESSING PIPELINE")
    print("="*70)
    
    # Run the main execution block of data_preprocessing.py
    os.system('python data_preprocessing.py')
    
    print("\n" + "="*70)
    print("DATA PREPROCESSING COMPLETE.")
    print("="*70)

def train_model():
    """Build, compile, and train the model."""
    print("\n" + "="*70)
    print("2. STARTING MODEL TRAINING PIPELINE")
    print("="*70)
    
    # Initialize and build model
    model_obj = VitaminDeficiencyModel(num_classes=6, img_size=224)
    model_obj.build_model()
    model_obj.compile_model(learning_rate=0.001)
    
    # Check if data exists
    if not os.path.exists(TRAIN_DIR) or not os.path.exists(VAL_DIR):
        print(f"ERROR: Training or Validation data not found. Run preprocessing first.")
        return
        
    # Create data generators
    train_gen, val_gen = model_obj.create_data_generators(TRAIN_DIR, VAL_DIR, batch_size=32)
    
    # Train the model (Using small epochs for demonstration)
    print("Starting initial training (5 epochs for demo)...")
    model_obj.train(train_gen, val_gen, epochs=5)
    
    # Fine-tune the model (Using small epochs for demonstration)
    print("Starting fine-tuning (2 epochs for demo)...")
    model_obj.fine_tune(train_gen, val_gen, epochs=2)
    
    # Save the final model
    model_obj.save_model(FINAL_MODEL_PATH)
    
    print("\n" + "="*70)
    print("MODEL TRAINING COMPLETE.")
    print("="*70)

def evaluate_model():
    """Evaluate the trained model using the evaluate.py logic."""
    print("\n" + "="*70)
    print("3. STARTING MODEL EVALUATION")
    print("="*70)
    
    if not os.path.exists(FINAL_MODEL_PATH):
        print(f"ERROR: Model file not found at {FINAL_MODEL_PATH}. Cannot evaluate.")
        return
    
    os.system('python evaluate.py')

    print("\n" + "="*70)
    print("MODEL EVALUATION COMPLETE.")
    print("="*70)

def launch_api():
    """Launches the Flask API server."""
    print("\n" + "="*70)
    print("4. LAUNCHING FLASK API SERVER (app.py)")
    print("="*70)
    print("Please open the frontend/index.html file in your browser.")
    print("API running at: http://127.0.0.1:5000")
    print("Press Ctrl+C to stop the server.")
    
    # This needs to be run continuously, so it's best executed in a separate process/terminal
    # For a simple script, we'll use os.system, but be aware it blocks the script.
    # In a real setup, use 'nohup python app.py &' or a proper process manager.
    os.system('python app.py')

if __name__ == "__main__":
    # 1. Ensure directories are set up (can be run once)
    setup_directories()
    
    # 2. Run data processing (requires data in data/raw)
    # run_data_pipeline() 
    
    # 3. Train the model (requires preprocessed data)
    # train_model()
    
    # 4. Evaluate the model (requires trained model and test data)
    # evaluate_model()
    
    # 5. Launch the API server
    print("\nNOTE: Data preprocessing and training steps are commented out.")
    print("Uncomment them if you have the dataset available.")
    print("Starting API now...")
    launch_api()