import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from transformers import pipeline

# Load the image classification model
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Load the text generation model
text_generator = pipeline('text-generation', model='gpt2')

# Define a function to preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))
    image_array = np.array(image)
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
    return np.expand_dims(image_array, axis=0)

# Define a function to get image predictions
def predict_image(image):
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)[0]
    return decoded_predictions[0][1]

# Define a function to handle the conversational response
def get_conversational_response(image_description, user_query):
    conversation_input = f"The image contains {image_description}. User query: {user_query}"
    response = text_generator(conversation_input, max_length=100)[0]['generated_text']
    return response

# Streamlit app
def main():
    st.title("Conversational Image Recognition Chatbot")

    # Upload an image
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Get image description
        image_description = predict_image(image)
        st.write(f"Image Description: {image_description}")

        # User query
        user_query = st.text_input("Ask a question about the image:")

        if user_query:
            response = get_conversational_response(image_description, user_query)
            st.write(f"Response: {response}")

if __name__ == "__main__":
    main()
