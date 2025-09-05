import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from PIL import Image

IMG_SIZE = 224
MODEL_PATH = 'models/caltech_classifier_vgg16.keras'
FEATURES_PATH = 'data/features.pkl'
IMAGE_LIST_PATH = 'data/image_list.pkl'

def preprocess_query_image(image_array):
    image = tf.convert_to_tensor(image_array)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.keras.applications.vgg16.preprocess_input(image)
    return image

@st.cache_resource
def load_all():
    base_model = tf.keras.models.load_model(MODEL_PATH)
    feature_extractor = tf.keras.Model(
        inputs=base_model.inputs,
        outputs=base_model.get_layer('feature_dense_layer').output
    )
    with open(FEATURES_PATH, 'rb') as f:
        features = pickle.load(f)
    with open(IMAGE_LIST_PATH, 'rb') as f:
        images = pickle.load(f)
    return feature_extractor, features, images

st.title("Visual Image Search Engine ")
st.write("Upload an image to find visually similar ones from the Caltech-101 dataset.")

try:
    feature_extractor, features, images = load_all()

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        query_image = Image.open(uploaded_file).convert('RGB')
        st.image(query_image, caption='Your Query Image', width=250)
        
        if st.button('Search'):
            with st.spinner('Finding similar images...'):
                query_img_array = np.array(query_image)
                img_preprocessed = preprocess_query_image(query_img_array)
                
                query_features = feature_extractor.predict(tf.expand_dims(img_preprocessed, axis=0))
                
                from sklearn.metrics.pairwise import cosine_similarity
                similarities = cosine_similarity(query_features, features)
                scores = similarities[0]

                SIMILARITY_THRESHOLD = 0.7

                top_indices_initial = np.argsort(scores)[-20:][::-1]
                
                good_results = []
                for idx in top_indices_initial:
                    if scores[idx] >= SIMILARITY_THRESHOLD:
                        good_results.append((idx, scores[idx]))

                top_5_results = good_results[:5]

                st.write("---")
                if not top_5_results:
                    st.warning("No confident matches found. The most similar images were below the quality threshold.")
                else:
                    st.success(f"Found {len(top_5_results)} high-confidence matches:")
                    cols = st.columns(len(top_5_results))
                    for i, (idx, score) in enumerate(top_5_results):
                        with cols[i]:
                            st.image(images[idx], caption=f"Score: {score:.2f}")

except FileNotFoundError:
    st.error("Model or data files not found. Please run `train_model.py` and `create_index.py` first.")
except Exception as e:
    st.error(f"An error occurred: {e}")

