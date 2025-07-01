import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.applications.densenet import preprocess_input
import numpy as np
from tensorflow.keras.optimizers import Adam
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import seaborn as sns
import pandas as pd

def create_densenet201_model(input_shape=(224, 224, 3), num_classes=2, weights='imagenet'):
    """
    Creates a DenseNet201 model for transfer learning.
    The base model is frozen and a new classifier head is added.
    """
    # Load the DenseNet201 model with pre-trained ImageNet weights, without the top classification layer
    base_model = DenseNet201(weights=weights, include_top=False, input_shape=input_shape)

    # Freeze the layers of the base model to keep the pre-trained features
    base_model.trainable = False

    # Create a new model on top
    inputs = Input(shape=input_shape)
    # Preprocess input for DenseNet201
    x = preprocess_input(inputs)
    # Run the base model in inference mode
    x = base_model(x, training=False)
    # Add a global spatial average pooling layer
    x = layers.GlobalAveragePooling2D()(x)
    # Add a fully-connected layer
    x = layers.Dense(1024, activation='relu')(x)
    # Add a dropout layer for regularization
    x = layers.Dropout(0.5)(x)
    # Add a final softmax layer for classification
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    return model

def load_dataset(healthy_dir='augmented_healthy', tumor_dir='augmented_braintumor', img_size=(224, 224)):
    """Load and preprocess the brain MRI dataset"""
    
    def load_images_from_folder(folder_path, label):
        images = []
        labels = []
        
        supported_formats = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
        
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(supported_formats):
                img_path = os.path.join(folder_path, filename)
                try:
                    # Load and preprocess image
                    img = tf.keras.utils.load_img(img_path, target_size=img_size)
                    img_array = tf.keras.utils.img_to_array(img)
                    # Normalization is now handled by preprocess_input in the model
                    
                    images.append(img_array)
                    labels.append(label)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    continue
        
        return images, labels
    
    # Load healthy images (label 0)
    healthy_images, healthy_labels = load_images_from_folder(healthy_dir, 0)
    print(f"Loaded {len(healthy_images)} healthy images")
    
    # Load tumor images (label 1)
    tumor_images, tumor_labels = load_images_from_folder(tumor_dir, 1)
    print(f"Loaded {len(tumor_images)} tumor images")
    
    # Combine datasets
    all_images = np.array(healthy_images + tumor_images)
    all_labels = np.array(healthy_labels + tumor_labels)
    
    # Convert labels to categorical
    all_labels = tf.keras.utils.to_categorical(all_labels, num_classes=2)
    
    return all_images, all_labels

def create_data_generators(images, labels, validation_split=0.2, batch_size=8):
    """Create training and validation data generators"""
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=validation_split, random_state=42, stratify=labels
    )
    
    # Create data generators with augmentation for training
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    
    train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
    val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)
    
    return train_generator, val_generator, len(X_train), len(X_val)

# Example usage
if __name__ == "__main__":
    # Load dataset
    print("Loading dataset...")
    images, labels = load_dataset()
    
    # Create data generators
    train_gen, val_gen, train_size, val_size = create_data_generators(images, labels, batch_size=8)
    
    # Create DenseNet201 model for binary classification
    print("Creating DenseNet201 model...")
    chosen_model = create_densenet201_model(input_shape=(224, 224, 3), num_classes=2, weights='imagenet')
    
    # Compile the model
    chosen_model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    print("\nDenseNet201 Model Summary:")
    chosen_model.summary()
    
    # Train models
    print(f"\nTraining data: {train_size} samples")
    print(f"Test data: {val_size} samples")
    
    # Train the model
    history = chosen_model.fit(
        train_gen,
        epochs=50,
        validation_data=val_gen,
        verbose=1
    )
    
    # Save the trained model
    chosen_model.save('brain_mri_densenet201_tidak_seimbang.h5')
    print("Model saved as 'brain_mri_densenet201_tidak_seimbang.h5'")

    # --- EVALUASI DAN SIMPAN ROC CURVE & CONFUSION MATRIX UNTUK TRAIN DAN TEST ---
    print("\nEvaluating model on training and test sets...")
    
    def evaluate_dataset(generator, dataset_name):
        """Evaluate model on a dataset and return metrics"""
        generator.reset()
        y_true = []
        y_pred = []
        y_score = []

        for i in tqdm(range(len(generator)), desc=f"Evaluating on {dataset_name} set"):
            x_batch, y_batch = generator[i]
            preds = chosen_model.predict_on_batch(x_batch)
            y_true.extend(np.argmax(y_batch, axis=1))
            y_pred.extend(np.argmax(preds, axis=1))
            y_score.extend(preds[:, 1])  # Probabilitas kelas 1 (tumor)

        return np.array(y_true), np.array(y_pred), np.array(y_score)
    
    # Evaluasi pada data training
    print("\n=== TRAINING SET EVALUATION ===")
    y_true_train, y_pred_train, y_score_train = evaluate_dataset(train_gen, "training")
    
    # ROC Curve untuk Training
    fpr_train, tpr_train, thresholds_train = roc_curve(y_true_train, y_score_train)
    roc_auc_train = auc(fpr_train, tpr_train)
    
    # Simpan ROC data training
    with open('roc_curve_train_data_densenet201_tidak_seimbang.pkl', 'wb') as f:
        pickle.dump({'fpr': fpr_train, 'tpr': tpr_train, 'thresholds': thresholds_train, 'auc': roc_auc_train}, f)
    print(f"Training ROC AUC: {roc_auc_train:.4f}")
    print("Training ROC curve data saved as 'roc_curve_train_data_densenet201_tidak_seimbang.pkl'")
    
    # Confusion Matrix untuk Training
    cm_train = confusion_matrix(y_true_train, y_pred_train)
    np.save('confusion_matrix_train_densenet201_tidak_seimbang.npy', cm_train)
    print("Training confusion matrix saved as 'confusion_matrix_train_densenet201_tidak_seimbang.npy'")
    
    # Evaluasi pada data test
    print("\n=== TEST SET EVALUATION ===")
    y_true_test, y_pred_test, y_score_test = evaluate_dataset(val_gen, "test")
    
    # ROC Curve untuk Test
    fpr_test, tpr_test, thresholds_test = roc_curve(y_true_test, y_score_test)
    roc_auc_test = auc(fpr_test, tpr_test)
    
    # Simpan ROC data test
    with open('roc_curve_test_data_densenet201_tidak_seimbang.pkl', 'wb') as f:
        pickle.dump({'fpr': fpr_test, 'tpr': tpr_test, 'thresholds': thresholds_test, 'auc': roc_auc_test}, f)
    print(f"Test ROC AUC: {roc_auc_test:.4f}")
    print("Test ROC curve data saved as 'roc_curve_test_data_densenet201_tidak_seimbang.pkl'")
    
    # Confusion Matrix untuk Test
    cm_test = confusion_matrix(y_true_test, y_pred_test)
    np.save('confusion_matrix_test_densenet201_tidak_seimbang.npy', cm_test)
    print("Test confusion matrix saved as 'confusion_matrix_test_densenet201_tidak_seimbang.npy'")
    
    # === PLOT INDIVIDUAL ROC CURVES ===
    # ROC Training only
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_train, tpr_train, label=f'Training ROC (AUC = {roc_auc_train:.3f})', color='blue', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Training Set - DenseNet201')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.savefig('roc_curve_training_densenet201_tidak_seimbang.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Training ROC curve saved as 'roc_curve_training_densenet201_tidak_seimbang.png'")

    # ROC Test only
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_test, tpr_test, label=f'Test ROC (AUC = {roc_auc_test:.3f})', color='red', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Test Set - DenseNet201')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.savefig('roc_curve_test_densenet201_tidak_seimbang.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Test ROC curve saved as 'roc_curve_test_densenet201_tidak_seimbang.png'")

    # === PLOT INDIVIDUAL CONFUSION MATRICES ===
    class_names = ['Healthy', 'Tumor']
    
    # Confusion Matrix Training only
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_train, annot=True, fmt="d", cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Training Set - DenseNet201')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix_training_densenet201_tidak_seimbang.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Training confusion matrix saved as 'confusion_matrix_training_densenet201_tidak_seimbang.png'")

    # Confusion Matrix Test only
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_test, annot=True, fmt="d", cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Test Set - DenseNet201')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix_test_densenet201_tidak_seimbang.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Test confusion matrix saved as 'confusion_matrix_test_densenet201_tidak_seimbang.png'")

    # === SAVE ROC CURVE TRAINING IN MULTIPLE FORMATS ===
    # ROC Training - CSV
    roc_train_df = pd.DataFrame({
        'fpr': fpr_train,
        'tpr': tpr_train,
        'thresholds': thresholds_train
    })
    roc_train_df.to_csv('roc_curve_train_data_densenet201_tidak_seimbang.csv', index=False)
    
    # ROC Training - NPY
    np.save('roc_curve_train_fpr_densenet201_tidak_seimbang.npy', fpr_train)
    np.save('roc_curve_train_tpr_densenet201_tidak_seimbang.npy', tpr_train)
    np.save('roc_curve_train_thresholds_densenet201_tidak_seimbang.npy', thresholds_train)
    
    print(f"Training ROC AUC: {roc_auc_train:.4f}")
    print("Training ROC curve saved in multiple formats:")
    print("  - roc_curve_train_data_densenet201_tidak_seimbang.pkl")
    print("  - roc_curve_train_data_densenet201_tidak_seimbang.csv")
    print("  - roc_curve_train_fpr_densenet201_tidak_seimbang.npy, roc_curve_train_tpr_densenet201_tidak_seimbang.npy, roc_curve_train_thresholds_densenet201_tidak_seimbang.npy")
    print("  - roc_curve_training_densenet201_tidak_seimbang.png")

    # === SAVE CONFUSION MATRIX TRAINING IN MULTIPLE FORMATS ===
    # CM Training - PKL
    with open('confusion_matrix_train_data_densenet201_tidak_seimbang.pkl', 'wb') as f:
        pickle.dump(cm_train, f)
    
    # CM Training - CSV
    cm_train_df = pd.DataFrame(cm_train, index=['Healthy', 'Tumor'], columns=['Healthy', 'Tumor'])
    cm_train_df.to_csv('confusion_matrix_train_data_densenet201_tidak_seimbang.csv')
    
    print("Training confusion matrix saved in multiple formats:")
    print("  - confusion_matrix_train_densenet201_tidak_seimbang.npy")
    print("  - confusion_matrix_train_data_densenet201_tidak_seimbang.pkl")
    print("  - confusion_matrix_train_data_densenet201_tidak_seimbang.csv")
    print("  - confusion_matrix_training_densenet201_tidak_seimbang.png")

    # === SAVE ROC CURVE TEST IN MULTIPLE FORMATS ===
    # ROC Test - CSV
    roc_test_df = pd.DataFrame({
        'fpr': fpr_test,
        'tpr': tpr_test,
        'thresholds': thresholds_test
    })
    roc_test_df.to_csv('roc_curve_test_data_densenet201_tidak_seimbang.csv', index=False)
    
    # ROC Test - NPY
    np.save('roc_curve_test_fpr_densenet201_tidak_seimbang.npy', fpr_test)
    np.save('roc_curve_test_tpr_densenet201_tidak_seimbang.npy', tpr_test)
    np.save('roc_curve_test_thresholds_densenet201_tidak_seimbang.npy', thresholds_test)
    
    print(f"Test ROC AUC: {roc_auc_test:.4f}")
    print("Test ROC curve saved in multiple formats:")
    print("  - roc_curve_test_data_densenet201_tidak_seimbang.pkl")
    print("  - roc_curve_test_data_densenet201_tidak_seimbang.csv")
    print("  - roc_curve_test_fpr_densenet201_tidak_seimbang.npy, roc_curve_test_tpr_densenet201_tidak_seimbang.npy, roc_curve_test_thresholds_densenet201_tidak_seimbang.npy")
    print("  - roc_curve_test_densenet201_tidak_seimbang.png")

    # === SAVE CONFUSION MATRIX TEST IN MULTIPLE FORMATS ===
    # CM Test - PKL
    with open('confusion_matrix_test_data_densenet201_tidak_seimbang.pkl', 'wb') as f:
        pickle.dump(cm_test, f)
    
    # CM Test - CSV
    cm_test_df = pd.DataFrame(cm_test, index=['Healthy', 'Tumor'], columns=['Healthy', 'Tumor'])
    cm_test_df.to_csv('confusion_matrix_test_data_densenet201_tidak_seimbang.csv')
    
    print("Test confusion matrix saved in multiple formats:")
    print("  - confusion_matrix_test_densenet201_tidak_seimbang.npy")
    print("  - confusion_matrix_test_data_densenet201_tidak_seimbang.pkl")
    print("  - confusion_matrix_test_data_densenet201_tidak_seimbang.csv")
    print("  - confusion_matrix_test_densenet201_tidak_seimbang.png")

    # === SAVE TRAINING HISTORY IN MULTIPLE FORMATS ===
    if history:
        # Training History - PKL
        with open('training_history_data_densenet201_tidak_seimbang.pkl', 'wb') as f:
            pickle.dump(history.history, f)
        
        # Training History - CSV
        history_df = pd.DataFrame(history.history)
        history_df.to_csv('training_history_data_densenet201_tidak_seimbang.csv', index=False)
        
        # Training History - NPY
        np.save('training_history_data_densenet201_tidak_seimbang.npy', history.history)
        
        # Training History - PNG
        plt.figure(figsize=(12, 4))
        
        # Plot training & test accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
        plt.plot(history.history['val_accuracy'], label='Test Accuracy', color='red')
        plt.title('Model Accuracy - DenseNet201')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot training & test loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss', color='blue')
        plt.plot(history.history['val_loss'], label='Test Loss', color='red')
        plt.title('Model Loss - DenseNet201')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history_densenet201_tidak_seimbang.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Training history saved in multiple formats:")
        print("  - training_history_data_densenet201_tidak_seimbang.pkl")
        print("  - training_history_data_densenet201_tidak_seimbang.csv")
        print("  - training_history_data_densenet201_tidak_seimbang.npy")
        print("  - training_history_densenet201_tidak_seimbang.png")

    # === SAVE AUC DATA IN MULTIPLE FORMATS ===
    # AUC Training
    auc_train_data = {'auc': roc_auc_train, 'fpr': fpr_train, 'tpr': tpr_train}
    with open('auc_train_data_densenet201_tidak_seimbang.pkl', 'wb') as f:
        pickle.dump(auc_train_data, f)
    
    auc_train_df = pd.DataFrame({
        'metric': ['AUC'],
        'value': [roc_auc_train]
    })
    auc_train_df.to_csv('auc_train_data_densenet201_tidak_seimbang.csv', index=False)
    np.save('auc_train_value_densenet201_tidak_seimbang.npy', roc_auc_train)
    
    plt.figure(figsize=(6, 4))
    plt.bar(['Training AUC'], [roc_auc_train], color='blue', alpha=0.7)
    plt.ylim(0, 1)
    plt.ylabel('AUC Score')
    plt.title('Training AUC Score - DenseNet201')
    plt.text(0, roc_auc_train + 0.02, f'{roc_auc_train:.4f}', ha='center', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.savefig('auc_curve_training_densenet201_tidak_seimbang.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Training AUC saved in multiple formats:")
    print("  - auc_train_data_densenet201_tidak_seimbang.pkl")
    print("  - auc_train_data_densenet201_tidak_seimbang.csv")
    print("  - auc_train_value_densenet201_tidak_seimbang.npy")
    print("  - auc_curve_training_densenet201_tidak_seimbang.png")

    # AUC Test
    auc_test_data = {'auc': roc_auc_test, 'fpr': fpr_test, 'tpr': tpr_test}
    with open('auc_test_data_densenet201_tidak_seimbang.pkl', 'wb') as f:
        pickle.dump(auc_test_data, f)
    
    auc_test_df = pd.DataFrame({
        'metric': ['AUC'],
        'value': [roc_auc_test]
    })
    auc_test_df.to_csv('auc_test_data_densenet201_tidak_seimbang.csv', index=False)
    np.save('auc_test_value_densenet201_tidak_seimbang.npy', roc_auc_test)
    
    plt.figure(figsize=(6, 4))
    plt.bar(['Test AUC'], [roc_auc_test], color='red', alpha=0.7)
    plt.ylim(0, 1)
    plt.ylabel('AUC Score')
    plt.title('Test AUC Score - DenseNet201')
    plt.text(0, roc_auc_test + 0.02, f'{roc_auc_test:.4f}', ha='center', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.savefig('auc_curve_test_densenet201_tidak_seimbang.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Test AUC saved in multiple formats:")
    print("  - auc_test_data_densenet201_tidak_seimbang.pkl")
    print("  - auc_test_data_densenet201_tidak_seimbang.csv")
    print("  - auc_test_value_densenet201_tidak_seimbang.npy")
    print("  - auc_curve_test_densenet201_tidak_seimbang.png")

    # Combined AUC data - CSV
    auc_combined_df = pd.DataFrame({
        'dataset': ['Training', 'Test'],
        'auc_score': [roc_auc_train, roc_auc_test]
    })
    auc_combined_df.to_csv('auc_combined_data_densenet201_tidak_seimbang.csv', index=False)
    print("Combined AUC data saved as 'auc_combined_data_densenet201_tidak_seimbang.csv'")

    # === SUMMARY METRICS ===
    print("\n=== PERFORMANCE SUMMARY - DENSENET201 ===")
    print("TRAINING SET:")
    print(f"  Accuracy:  {accuracy_score(y_true_train, y_pred_train):.4f}")
    print(f"  Precision: {precision_score(y_true_train, y_pred_train):.4f}")
    print(f"  Recall:    {recall_score(y_true_train, y_pred_train):.4f}")
    print(f"  F1-Score:  {f1_score(y_true_train, y_pred_train):.4f}")
    print(f"  ROC AUC:   {roc_auc_train:.4f}")
    
    print("\nTEST SET:")
    print(f"  Accuracy:  {accuracy_score(y_true_test, y_pred_test):.4f}")
    print(f"  Precision: {precision_score(y_true_test, y_pred_test):.4f}")
    print(f"  Recall:    {recall_score(y_true_test, y_pred_test):.4f}")
    print(f"  F1-Score:  {f1_score(y_true_test, y_pred_test):.4f}")
    print(f"  ROC AUC:   {roc_auc_test:.4f}")