import tensorflow as tf
import tensorflow_datasets as tfds
import os

IMG_SIZE = 224 
BATCH_SIZE = 32
EPOCHS = 10 
NUM_CLASSES = 101
MODEL_SAVE_PATH = os.path.join('models', 'caltech_classifier_vgg16.keras')

def preprocess_image(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.keras.applications.vgg16.preprocess_input(image)
    label = tf.one_hot(label, depth=NUM_CLASSES)
    return image, label

print("Loading and preparing Caltech101 dataset...")
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
print("Dataset prepared.")

def create_model(input_shape, num_classes):
    base_model = tf.keras.applications.VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    base_model.trainable = False
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.Flatten(name="feature_flattener"),
        tf.keras.layers.Dense(512, activation='relu', name='feature_dense_layer'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax', name='output_layer')
    ])
    return model

print("Creating and compiling model with VGG16 base...")
model = create_model((IMG_SIZE, IMG_SIZE, 3), NUM_CLASSES)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

print("Starting model training...")
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=validation_dataset
)

print(f"\nTraining complete. Saving model to {MODEL_SAVE_PATH}...")
model.save(MODEL_SAVE_PATH)
print("Model saved successfully!")