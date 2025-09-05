import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os
import pickle

IMG_SIZE = 224
MODEL_PATH = os.path.join('models', 'caltech_classifier_vgg16.keras')
FEATURES_PATH = os.path.join('data', 'features.pkl')
IMAGE_LIST_PATH = os.path.join('data', 'image_list.pkl')

print("Loading pre-trained model to build feature extractor...")
base_model = tf.keras.models.load_model(MODEL_PATH)

feature_extractor = tf.keras.Model(
    inputs=base_model.inputs,
    outputs=base_model.get_layer('feature_dense_layer').output 
)
print("Feature extractor created.")

def preprocess_for_extraction(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.keras.applications.vgg16.preprocess_input(image)
    return image, label

print("Indexing dataset... This may take a while.")
(dataset, ), _ = tfds.load('caltech101', split=['train'], as_supervised=True, with_info=True)

image_list = []
feature_list = []

dataset_preprocessed = dataset.map(preprocess_for_extraction).batch(32)

for img_batch, label_batch in dataset_preprocessed:
    features = feature_extractor.predict(img_batch, verbose=0)
    feature_list.extend(features)

print("Storing original images for display...")
for img, label in dataset: 
    image_list.append(img.numpy())

with open(FEATURES_PATH, 'wb') as f:
    pickle.dump(feature_list, f)
with open(IMAGE_LIST_PATH, 'wb') as f:
    pickle.dump(image_list, f)

print(f"\nIndexing complete. Saved {len(feature_list)} features and images. ✅")