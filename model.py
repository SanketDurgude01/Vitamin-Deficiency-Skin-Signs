"""
STEP 1: model.py
Save this file as: model.py
Description: Deep learning model for vitamin deficiency detection
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
import numpy as np

class VitaminDeficiencyModel:
    def __init__(self, num_classes=6, img_size=224):
        """Initialize the model"""
        self.num_classes = num_classes
        self.img_size = img_size
        self.model = None
        self.class_names = [
            'Vitamin_A_Deficiency',
            'Vitamin_B_Deficiency',
            'Vitamin_C_Deficiency',
            'Vitamin_D_Deficiency',
            'Vitamin_E_Deficiency',
            'Normal_Skin'
        ]
        
    def build_model(self):
        """Build the CNN model using EfficientNetB0"""
        # Load pre-trained base model
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(self.img_size, self.img_size, 3)
        )
        base_model.trainable = False
        
        # Build complete model
        inputs = keras.Input(shape=(self.img_size, self.img_size, 3))
        
        # Data augmentation
        x = layers.RandomFlip("horizontal")(inputs)
        x = layers.RandomRotation(0.1)(x)
        x = layers.RandomZoom(0.1)(x)
        
        # Preprocessing
        x = tf.keras.applications.efficientnet.preprocess_input(x)
        
        # Base model
        x = base_model(x, training=False)
        

        
        # Custom layers
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = keras.Model(inputs, outputs)
        print("✓ Model built successfully!")
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model"""
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )
        print("✓ Model compiled successfully!")
        
    def create_data_generators(self, train_dir, val_dir, batch_size=32):
        """Create data generators for training"""
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        print(f"✓ Training samples: {train_generator.samples}")
        print(f"✓ Validation samples: {val_generator.samples}")
        
        return train_generator, val_generator
    
    def train(self, train_generator, val_generator, epochs=50):
        """Train the model"""
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'vitamin_deficiency_model_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            )
        ]
        
        print(f"\n{'='*70}")
        print(f"Starting training for {epochs} epochs...")
        print(f"{'='*70}\n")
        
        history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        print(f"\n{'='*70}")
        print("Training completed!")
        print(f"{'='*70}\n")
        
        return history
    
    def fine_tune(self, train_generator, val_generator, epochs=20):
        """Fine-tune the model by unfreezing some layers"""
        print("\n" + "="*70)
        print("Starting fine-tuning phase...")
        print("="*70 + "\n")
        
        # Unfreeze the base model
        base_model = self.model.layers[4]
        base_model.trainable = True
        
        # Freeze early layers
        for layer in base_model.layers[:100]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        self.compile_model(learning_rate=1e-5)
        
        # Continue training
        history = self.train(train_generator, val_generator, epochs=epochs)
        
        print("✓ Fine-tuning completed!")
        return history
    
    def predict_image(self, img_path):
        """Predict vitamin deficiency from image"""
        # Load and preprocess image
        img = keras.preprocessing.image.load_img(
            img_path, 
            target_size=(self.img_size, self.img_size)
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        
        # Make prediction
        predictions = self.model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        
        result = {
            'class': self.class_names[predicted_class_idx],
            'confidence': float(confidence),
            'all_probabilities': {
                name: float(prob) 
                for name, prob in zip(self.class_names, predictions[0])
            }
        }
        
        return result
    
    def save_model(self, path='vitamin_deficiency_model.h5'):
        """Save the trained model"""
        self.model.save(path)
        print(f"✓ Model saved to {path}")
    
    def load_model(self, path='vitamin_deficiency_model.h5'):
        """Load a trained model"""
        self.model = keras.models.load_model(path)
        print(f"✓ Model loaded from {path}")


# Example usage
if __name__ == "__main__":
    print("\n" + "="*70)
    print("VITAMIN DEFICIENCY DETECTION MODEL")
    print("="*70 + "\n")
    
    # Initialize model
    print("[1] Initializing model...")
    model = VitaminDeficiencyModel(num_classes=6, img_size=224)
    
    # Build model
    print("\n[2] Building model architecture...")
    model.build_model()
    
    # Compile model
    print("\n[3] Compiling model...")
    model.compile_model()
    
    # Print model summary
    print("\n[4] Model Summary:")
    print("-" * 70)
    model.model.summary()
    
    print("\n" + "="*70)
    print("Model initialized successfully!")
    print("="*70)
    print("\nTo train the model:")
    print("1. Prepare your dataset in data/processed/train and data/processed/validation")
    print("2. Run the following code:")
    print()
    print("   train_gen, val_gen = model.create_data_generators(")
    print("       'data/processed/train',")
    print("       'data/processed/validation'")
    print("   )")
    print("   history = model.train(train_gen, val_gen, epochs=50)")
    print("   model.save_model('vitamin_deficiency_model_final.h5')")
    print()
    print("="*70 + "\n")
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def create_data_generators(train_dir, val_dir):
    datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    val_gen = datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    return train_gen, val_gen


def build_model(num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)),
        MaxPooling2D(),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def train(train_gen, val_gen, epochs=50):
    model = build_model(train_gen.num_classes)

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs
    )

    return model, history


def save_model(model, path):
    model.save(path)
