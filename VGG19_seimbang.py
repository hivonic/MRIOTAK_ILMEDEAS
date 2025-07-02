import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.applications import VGG19
from tensorflow.keras.optimizers import Adam
import numpy as np
from tensorflow.keras.applications.vgg19 import preprocess_input # Import preprocess_input for VGG19
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
# Add precision_recall_curve import
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import pickle
import random # Diperlukan untuk penyeimbangan dataset
from tqdm import tqdm
import seaborn as sns
import pandas as pd

def create_vgg19_model(input_shape=(224, 224, 3), num_classes=2, weights='imagenet'):
    """
    Creates a VGG19 model for transfer learning.
    The base model is frozen and a new classifier head is added.
    """
    # Load the VGG19 model with pre-trained ImageNet weights, without the top classification layer
    base_model = VGG19(weights=weights, include_top=False, input_shape=input_shape)

    # Freeze the layers of the base model to keep the pre-trained features
    base_model.trainable = False

    # Create a new model on top
    inputs = Input(shape=input_shape)
    # Preprocess input for VGG19
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
    # Kita akan memuat semua gambar healthy terlebih dahulu
    all_healthy_images, all_healthy_labels = load_images_from_folder(healthy_dir, 0)
    print(f"Loaded {len(all_healthy_images)} healthy images")
    
    # Load tumor images (label 1)
    # Kita akan memuat semua gambar tumor terlebih dahulu
    all_tumor_images, all_tumor_labels = load_images_from_folder(tumor_dir, 1)
    print(f"Loaded {len(all_tumor_images)} tumor images")

    # --- Penyeimbangan Dataset ---
    num_healthy_samples = len(all_healthy_images)
    num_tumor_samples = len(all_tumor_images)
    
    # Target jumlah sampel per kelas adalah jumlah sampel kelas yang lebih kecil (healthy)
    target_count_per_class = num_healthy_samples

    # Jika jumlah sampel tumor lebih banyak dari target, lakukan sampling acak
    if num_tumor_samples > target_count_per_class:
        # Gabungkan gambar dan label tumor, acak, lalu ambil sejumlah target_count_per_class
        combined_tumor_data = list(zip(all_tumor_images, all_tumor_labels))
        random.shuffle(combined_tumor_data)
        sampled_tumor_data = combined_tumor_data[:target_count_per_class]
        tumor_images, tumor_labels = zip(*sampled_tumor_data)
        tumor_images = list(tumor_images) # Konversi kembali ke list
        tumor_labels = list(tumor_labels) # Konversi kembali ke list
        print(f"Sampled {len(tumor_images)} tumor images to balance the dataset.")
    else:
        # Jika tumor lebih sedikit atau sama, gunakan semua gambar tumor
        tumor_images = all_tumor_images
        tumor_labels = all_tumor_labels

    # Gunakan semua gambar healthy
    healthy_images = all_healthy_images
    healthy_labels = all_healthy_labels
    
    # Combine datasets
    all_images = np.array(healthy_images + tumor_images)
    all_labels = np.array(healthy_labels + tumor_labels)
    
    # Tambahkan pengacakan pada dataset gabungan untuk memastikan urutan yang acak
    # sebelum pembagian train/val
    combined_data = list(zip(all_images, all_labels))
    random.shuffle(combined_data)
    all_images, all_labels = zip(*combined_data)
    all_images = np.array(all_images)
    
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
    # Create output directories with custom parent folder
    base_output_dir = "Output VGG19 Seimbang"
    output_dirs = {
        'models': os.path.join(base_output_dir, 'models'),
        'roc': os.path.join(base_output_dir, 'roc_curves'),
        'pr': os.path.join(base_output_dir, 'pr_curves'), 
        'confusion': os.path.join(base_output_dir, 'confusion_matrices'),
        'history': os.path.join(base_output_dir, 'training_history'),
        'auc': os.path.join(base_output_dir, 'auc_scores'),
        'pr_auc': os.path.join(base_output_dir, 'pr_auc_scores')
    }
    
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    images, labels = load_dataset()
    
    # Create data generators
    train_gen, val_gen, train_size, val_size = create_data_generators(images, labels, batch_size=8)
    
    # Create VGG19 model for binary classification
    print("Creating VGG19 model...")
    chosen_model = create_vgg19_model(input_shape=(224, 224, 3), num_classes=2, weights='imagenet')
    
    # Compile the model
    chosen_model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    print("\nVGG19 Model Summary:")
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
    
    # Save the trained VGG19 model
    chosen_model.save(os.path.join(output_dirs['models'], 'brain_mri_vgg19_seimbang.h5'))
    print(f"Model saved as '{os.path.join(output_dirs['models'], 'brain_mri_vgg19_seimbang.h5')}'")

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
    
    # PR Curve untuk Training
    precision_train, recall_train, pr_thresholds_train = precision_recall_curve(y_true_train, y_score_train)
    pr_auc_train = average_precision_score(y_true_train, y_score_train)
    
    # Confusion Matrix untuk Training
    cm_train = confusion_matrix(y_true_train, y_pred_train)
    
    # Evaluasi pada data test
    print("\n=== TEST SET EVALUATION ===")
    y_true_test, y_pred_test, y_score_test = evaluate_dataset(val_gen, "test")
    
    # ROC Curve untuk Test
    fpr_test, tpr_test, thresholds_test = roc_curve(y_true_test, y_score_test)
    roc_auc_test = auc(fpr_test, tpr_test)
    
    # PR Curve untuk Test
    precision_test, recall_test, pr_thresholds_test = precision_recall_curve(y_true_test, y_score_test)
    pr_auc_test = average_precision_score(y_true_test, y_score_test)
    
    # Confusion Matrix untuk Test
    cm_test = confusion_matrix(y_true_test, y_pred_test)
    
    print(f"Training ROC AUC: {roc_auc_train:.4f}")
    print(f"Training PR AUC: {pr_auc_train:.4f}")
    print(f"Test ROC AUC: {roc_auc_test:.4f}")
    print(f"Test PR AUC: {pr_auc_test:.4f}")
    
    # === PLOT ROC CURVES ===
    # ROC Training
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_train, tpr_train, label=f'Training ROC (AUC = {roc_auc_train:.3f})', color='blue', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Training Set - VGG19 (Balanced)')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dirs['roc'], 'roc_curve_training_vgg19_seimbang.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ROC Test
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_test, tpr_test, label=f'Test ROC (AUC = {roc_auc_test:.3f})', color='red', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Test Set - VGG19 (Balanced)')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dirs['roc'], 'roc_curve_test_vgg19_seimbang.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # === PLOT PR CURVES ===
    # PR Training
    plt.figure(figsize=(8, 6))
    plt.plot(recall_train, precision_train, label=f'Training PR (AUC = {pr_auc_train:.3f})', color='blue', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - Training Set - VGG19 (Balanced)')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dirs['pr'], 'pr_curve_training_vgg19_seimbang.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # PR Test
    plt.figure(figsize=(8, 6))
    plt.plot(recall_test, precision_test, label=f'Test PR (AUC = {pr_auc_test:.3f})', color='red', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - Test Set - VGG19 (Balanced)')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dirs['pr'], 'pr_curve_test_vgg19_seimbang.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # === PLOT CONFUSION MATRICES ===
    class_names = ['Healthy', 'Tumor']
    
    # Confusion Matrix Training
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_train, annot=True, fmt="d", cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Training Set - VGG19 (Balanced)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(output_dirs['confusion'], 'confusion_matrix_training_vgg19_seimbang.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Confusion Matrix Test
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_test, annot=True, fmt="d", cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Test Set - VGG19 (Balanced)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(output_dirs['confusion'], 'confusion_matrix_test_vgg19_seimbang.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # === SAVE ROC CURVE DATA ===
    # ROC Training - Multiple formats
    with open(os.path.join(output_dirs['roc'], 'roc_curve_train_data_vgg19_seimbang.pkl'), 'wb') as f:
        pickle.dump({'fpr': fpr_train, 'tpr': tpr_train, 'thresholds': thresholds_train, 'auc': roc_auc_train}, f)
    
    roc_train_df = pd.DataFrame({
        'fpr': fpr_train,
        'tpr': tpr_train,
        'thresholds': thresholds_train
    })
    roc_train_df.to_csv(os.path.join(output_dirs['roc'], 'roc_curve_train_data_vgg19_seimbang.csv'), index=False)
    
    np.save(os.path.join(output_dirs['roc'], 'roc_curve_train_fpr_vgg19_seimbang.npy'), fpr_train)
    np.save(os.path.join(output_dirs['roc'], 'roc_curve_train_tpr_vgg19_seimbang.npy'), tpr_train)
    np.save(os.path.join(output_dirs['roc'], 'roc_curve_train_thresholds_vgg19_seimbang.npy'), thresholds_train)

    # ROC Test - Multiple formats
    with open(os.path.join(output_dirs['roc'], 'roc_curve_test_data_vgg19_seimbang.pkl'), 'wb') as f:
        pickle.dump({'fpr': fpr_test, 'tpr': tpr_test, 'thresholds': thresholds_test, 'auc': roc_auc_test}, f)
    
    roc_test_df = pd.DataFrame({
        'fpr': fpr_test,
        'tpr': tpr_test,
        'thresholds': thresholds_test
    })
    roc_test_df.to_csv(os.path.join(output_dirs['roc'], 'roc_curve_test_data_vgg19_seimbang.csv'), index=False)
    
    np.save(os.path.join(output_dirs['roc'], 'roc_curve_test_fpr_vgg19_seimbang.npy'), fpr_test)
    np.save(os.path.join(output_dirs['roc'], 'roc_curve_test_tpr_vgg19_seimbang.npy'), tpr_test)
    np.save(os.path.join(output_dirs['roc'], 'roc_curve_test_thresholds_vgg19_seimbang.npy'), thresholds_test)

    # === SAVE PR CURVE DATA ===
    # PR Training - Multiple formats
    with open(os.path.join(output_dirs['pr'], 'pr_curve_train_data_vgg19_seimbang.pkl'), 'wb') as f:
        pickle.dump({'precision': precision_train, 'recall': recall_train, 'thresholds': pr_thresholds_train, 'auc': pr_auc_train}, f)
    
    pr_train_df = pd.DataFrame({
        'precision': precision_train,
        'recall': recall_train,
        'thresholds': np.append(pr_thresholds_train, 1)
    })
    pr_train_df.to_csv(os.path.join(output_dirs['pr'], 'pr_curve_train_data_vgg19_seimbang.csv'), index=False)
    
    np.save(os.path.join(output_dirs['pr'], 'pr_curve_train_precision_vgg19_seimbang.npy'), precision_train)
    np.save(os.path.join(output_dirs['pr'], 'pr_curve_train_recall_vgg19_seimbang.npy'), recall_train)
    np.save(os.path.join(output_dirs['pr'], 'pr_curve_train_thresholds_vgg19_seimbang.npy'), pr_thresholds_train)

    # PR Test - Multiple formats
    with open(os.path.join(output_dirs['pr'], 'pr_curve_test_data_vgg19_seimbang.pkl'), 'wb') as f:
        pickle.dump({'precision': precision_test, 'recall': recall_test, 'thresholds': pr_thresholds_test, 'auc': pr_auc_test}, f)
    
    pr_test_df = pd.DataFrame({
        'precision': precision_test,
        'recall': recall_test,
        'thresholds': np.append(pr_thresholds_test, 1)
    })
    pr_test_df.to_csv(os.path.join(output_dirs['pr'], 'pr_curve_test_data_vgg19_seimbang.csv'), index=False)
    
    np.save(os.path.join(output_dirs['pr'], 'pr_curve_test_precision_vgg19_seimbang.npy'), precision_test)
    np.save(os.path.join(output_dirs['pr'], 'pr_curve_test_recall_vgg19_seimbang.npy'), recall_test)
    np.save(os.path.join(output_dirs['pr'], 'pr_curve_test_thresholds_vgg19_seimbang.npy'), pr_thresholds_test)

    # === SAVE CONFUSION MATRIX DATA ===
    # CM Training
    np.save(os.path.join(output_dirs['confusion'], 'confusion_matrix_train_vgg19_seimbang.npy'), cm_train)
    with open(os.path.join(output_dirs['confusion'], 'confusion_matrix_train_data_vgg19_seimbang.pkl'), 'wb') as f:
        pickle.dump(cm_train, f)
    cm_train_df = pd.DataFrame(cm_train, index=['Healthy', 'Tumor'], columns=['Healthy', 'Tumor'])
    cm_train_df.to_csv(os.path.join(output_dirs['confusion'], 'confusion_matrix_train_data_vgg19_seimbang.csv'))

    # CM Test
    np.save(os.path.join(output_dirs['confusion'], 'confusion_matrix_test_vgg19_seimbang.npy'), cm_test)
    with open(os.path.join(output_dirs['confusion'], 'confusion_matrix_test_data_vgg19_seimbang.pkl'), 'wb') as f:
        pickle.dump(cm_test, f)
    cm_test_df = pd.DataFrame(cm_test, index=['Healthy', 'Tumor'], columns=['Healthy', 'Tumor'])
    cm_test_df.to_csv(os.path.join(output_dirs['confusion'], 'confusion_matrix_test_data_vgg19_seimbang.csv'))

    # === SAVE TRAINING HISTORY ===
    if history:
        with open(os.path.join(output_dirs['history'], 'training_history_data_vgg19_seimbang.pkl'), 'wb') as f:
            pickle.dump(history.history, f)
        
        history_df = pd.DataFrame(history.history)
        history_df.to_csv(os.path.join(output_dirs['history'], 'training_history_data_vgg19_seimbang.csv'), index=False)
        
        np.save(os.path.join(output_dirs['history'], 'training_history_data_vgg19_seimbang.npy'), history.history)
        
        # Training History Plot
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
        plt.plot(history.history['val_accuracy'], label='Test Accuracy', color='red')
        plt.title('Model Accuracy - VGG19 (Balanced Dataset)')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss', color='blue')
        plt.plot(history.history['val_loss'], label='Test Loss', color='red')
        plt.title('Model Loss - VGG19 (Balanced Dataset)')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dirs['history'], 'training_history_vgg19_seimbang.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # === SAVE ROC AUC DATA ===
    # ROC AUC Training
    auc_train_data = {'auc': roc_auc_train, 'fpr': fpr_train, 'tpr': tpr_train}
    with open(os.path.join(output_dirs['auc'], 'auc_train_data_vgg19_seimbang.pkl'), 'wb') as f:
        pickle.dump(auc_train_data, f)
    
    auc_train_df = pd.DataFrame({
        'metric': ['ROC_AUC'],
        'value': [roc_auc_train]
    })
    auc_train_df.to_csv(os.path.join(output_dirs['auc'], 'auc_train_data_vgg19_seimbang.csv'), index=False)
    np.save(os.path.join(output_dirs['auc'], 'auc_train_value_vgg19_seimbang.npy'), roc_auc_train)
    
    # ROC AUC Training Bar Chart
    plt.figure(figsize=(6, 4))
    plt.bar(['Training ROC AUC'], [roc_auc_train], color='blue', alpha=0.7)
    plt.ylim(0, 1)
    plt.ylabel('ROC AUC Score')
    plt.title('Training ROC AUC Score - VGG19 (Balanced)')
    plt.text(0, roc_auc_train + 0.02, f'{roc_auc_train:.4f}', ha='center', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dirs['auc'], 'auc_curve_training_vgg19_seimbang.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ROC AUC Test
    auc_test_data = {'auc': roc_auc_test, 'fpr': fpr_test, 'tpr': tpr_test}
    with open(os.path.join(output_dirs['auc'], 'auc_test_data_vgg19_seimbang.pkl'), 'wb') as f:
        pickle.dump(auc_test_data, f)
    
    auc_test_df = pd.DataFrame({
        'metric': ['ROC_AUC'],
        'value': [roc_auc_test]
    })
    auc_test_df.to_csv(os.path.join(output_dirs['auc'], 'auc_test_data_vgg19_seimbang.csv'), index=False)
    np.save(os.path.join(output_dirs['auc'], 'auc_test_value_vgg19_seimbang.npy'), roc_auc_test)
    
    # ROC AUC Test Bar Chart
    plt.figure(figsize=(6, 4))
    plt.bar(['Test ROC AUC'], [roc_auc_test], color='red', alpha=0.7)
    plt.ylim(0, 1)
    plt.ylabel('ROC AUC Score')
    plt.title('Test ROC AUC Score - VGG19 (Balanced)')
    plt.text(0, roc_auc_test + 0.02, f'{roc_auc_test:.4f}', ha='center', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dirs['auc'], 'auc_curve_test_vgg19_seimbang.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # === SAVE PR AUC DATA ===
    # PR AUC Training
    pr_auc_train_data = {'pr_auc': pr_auc_train, 'precision': precision_train, 'recall': recall_train}
    with open(os.path.join(output_dirs['pr_auc'], 'pr_auc_train_data_vgg19_seimbang.pkl'), 'wb') as f:
        pickle.dump(pr_auc_train_data, f)
    
    pr_auc_train_df = pd.DataFrame({
        'metric': ['PR_AUC'],
        'value': [pr_auc_train]
    })
    pr_auc_train_df.to_csv(os.path.join(output_dirs['pr_auc'], 'pr_auc_train_data_vgg19_seimbang.csv'), index=False)
    np.save(os.path.join(output_dirs['pr_auc'], 'pr_auc_train_value_vgg19_seimbang.npy'), pr_auc_train)
    
    # PR AUC Training Bar Chart
    plt.figure(figsize=(6, 4))
    plt.bar(['Training PR AUC'], [pr_auc_train], color='blue', alpha=0.7)
    plt.ylim(0, 1)
    plt.ylabel('PR AUC Score')
    plt.title('Training PR AUC Score - VGG19 (Balanced)')
    plt.text(0, pr_auc_train + 0.02, f'{pr_auc_train:.4f}', ha='center', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dirs['pr_auc'], 'pr_auc_curve_training_vgg19_seimbang.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # PR AUC Test
    pr_auc_test_data = {'pr_auc': pr_auc_test, 'precision': precision_test, 'recall': recall_test}
    with open(os.path.join(output_dirs['pr_auc'], 'pr_auc_test_data_vgg19_seimbang.pkl'), 'wb') as f:
        pickle.dump(pr_auc_test_data, f)
    
    pr_auc_test_df = pd.DataFrame({
        'metric': ['PR_AUC'],
        'value': [pr_auc_test]
    })
    pr_auc_test_df.to_csv(os.path.join(output_dirs['pr_auc'], 'pr_auc_test_data_vgg19_seimbang.csv'), index=False)
    np.save(os.path.join(output_dirs['pr_auc'], 'pr_auc_test_value_vgg19_seimbang.npy'), pr_auc_test)
    
    # PR AUC Test Bar Chart
    plt.figure(figsize=(6, 4))
    plt.bar(['Test PR AUC'], [pr_auc_test], color='red', alpha=0.7)
    plt.ylim(0, 1)
    plt.ylabel('PR AUC Score')
    plt.title('Test PR AUC Score - VGG19 (Balanced)')
    plt.text(0, pr_auc_test + 0.02, f'{pr_auc_test:.4f}', ha='center', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dirs['pr_auc'], 'pr_auc_curve_test_vgg19_seimbang.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # === SUMMARY METRICS ===
    print("\n=== PERFORMANCE SUMMARY - VGG19 (BALANCED DATASET) ===")
    print("TRAINING SET:")
    print(f"  Accuracy:  {accuracy_score(y_true_train, y_pred_train):.4f}")
    print(f"  Precision: {precision_score(y_true_train, y_pred_train):.4f}")
    print(f"  Recall:    {recall_score(y_true_train, y_pred_train):.4f}")
    print(f"  F1-Score:  {f1_score(y_true_train, y_pred_train):.4f}")
    print(f"  ROC AUC:   {roc_auc_train:.4f}")
    print(f"  PR AUC:    {pr_auc_train:.4f}")
    
    print("\nTEST SET:")
    print(f"  Accuracy:  {accuracy_score(y_true_test, y_pred_test):.4f}")
    print(f"  Precision: {precision_score(y_true_test, y_pred_test):.4f}")
    print(f"  Recall:    {recall_score(y_true_test, y_pred_test):.4f}")
    print(f"  F1-Score:  {f1_score(y_true_test, y_pred_test):.4f}")
    print(f"  ROC AUC:   {roc_auc_test:.4f}")
    print(f"  PR AUC:    {pr_auc_test:.4f}")

    print(f"\n=== FILES SAVED IN ORGANIZED DIRECTORIES ===")
    print(f"Models: {output_dirs['models']}")
    print(f"ROC Curves: {output_dirs['roc']}")
    print(f"PR Curves: {output_dirs['pr']}")
    print(f"Confusion Matrices: {output_dirs['confusion']}")
    print(f"Training History: {output_dirs['history']}")
    print(f"ROC AUC Scores: {output_dirs['auc']}")
    print(f"PR AUC Scores: {output_dirs['pr_auc']}")