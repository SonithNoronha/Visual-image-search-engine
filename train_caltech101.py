import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import os

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 20 
NUM_CLASSES = 101 
OPTIMIZERS = ['sgd', 'rmsprop', 'adam']

if not os.path.exists('results'):
    os.makedirs('results')

def preprocess_image(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.one_hot(label, depth=NUM_CLASSES)
    return image, label

(dataset, ), info = tfds.load(
    'caltech101',
    split=['train'],
    as_supervised=True,
    with_info=True,
)

dataset_size = info.splits['train'].num_examples
train_size = int(0.8 * dataset_size)

dataset = dataset.shuffle(dataset_size, seed=42)

train_dataset = dataset.take(train_size)
validation_dataset = dataset.skip(train_size)

train_dataset = train_dataset.map(preprocess_image).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
validation_dataset = validation_dataset.map(preprocess_image).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print(f"Dataset prepared: {train_size} training samples, {dataset_size - train_size} validation samples.")

def create_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

histories = {}

for optimizer_name in OPTIMIZERS:
    print(f"\n--- Training with optimizer: {optimizer_name.upper()} ---")

    model = create_model((IMG_SIZE, IMG_SIZE, 3), NUM_CLASSES)
    
    model.compile(
        optimizer=optimizer_name,
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    
   
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=validation_dataset
    )
    
    histories[optimizer_name] = history

plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))


for optimizer_name, history in histories.items():
    ax1.plot(history.history['val_accuracy'], label=optimizer_name.upper())
ax1.set_title('Validation Accuracy Comparison')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
ax1.legend()

for optimizer_name, history in histories.items():
    ax2.plot(history.history['val_loss'], label=optimizer_name.upper())
ax2.set_title('Validation Loss Comparison')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
ax2.legend()

plot_path = os.path.join('results', 'optimizer_comparison.png')
plt.savefig(plot_path)
print(f"\nComparison plot saved to: {plot_path}")

plt.show()

for optimizer_name, history in histories.items():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(f'Performance Metrics for Optimizer: {optimizer_name.upper()}', fontsize=16)

    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Accuracy over Epochs')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Loss over Epochs')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plot_path = os.path.join('results', f'performance_{optimizer_name}.png')
    plt.savefig(plot_path)
    print(f"Performance plot for {optimizer_name.upper()} saved to: {plot_path}")

    plt.show()