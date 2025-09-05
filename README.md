# Visual Image Search Engine 🖼️

A visual search engine built with TensorFlow, VGG16, and Streamlit. This application takes an uploaded image and finds the most visually similar images from the Caltech-101 dataset.

## Features
- **Transfer Learning:** Uses a pre-trained VGG16 model for powerful feature extraction.
- **Similarity Search:** Employs cosine similarity to find the closest matches.
- **Interactive UI:** A simple and intuitive web interface built with Streamlit.
- **Confidence Filtering:** Only shows results that meet a minimum similarity threshold.

## How to Run This Project

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/visual-image-search-engine.git](https://github.com/your-username/visual-image-search-engine.git)
    cd visual-image-search-engine
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Generate the Model and Index Files:**
    You must run the following scripts in order.
    ```bash
    # This will take some time to train the model
    python train_model.py

    # This will index the dataset and create the .pkl files
    python create_index.py
    ```

4.  **Run the Streamlit App:**
    ```bash
    streamlit run app.py
    ```

## Technologies Used
- TensorFlow / Keras
- Streamlit
- Scikit-learn
- NumPy
- Matplotlib
