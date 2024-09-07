import streamlit as st
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests

# Set up Streamlit app
st.title("Conversational Image Recognition Chatbot")
st.write("Upload an image and ask questions about it!")

# Load pre-trained models
@st.cache_resource
def load_models():
    # Load the object detection model (use a pre-trained model)
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-vqa-base")
    
    # Load the NLP model for conversational responses
    nlp = pipeline("conversational", model="microsoft/DialoGPT-medium")
    return processor, model, nlp

processor, img_model, nlp_model = load_models()

# File uploader for the image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # User input for questions
    user_question = st.text_input("Ask a question about the image:")

    if user_question:
        # Prepare the image and the question for the model
        inputs = processor(images=image, text=user_question, return_tensors="pt")

        # Generate a response using the image recognition model
        output = img_model.generate(**inputs)
        answer = processor.decode(output[0], skip_special_tokens=True)

        st.write(f"Image Recognition Response: {answer}")

        # Generate a conversational response using the NLP model
        chat_input = f"The image recognition result is '{answer}'. User asked: {user_question}"
        conversation = nlp_model(chat_input)

        # Display the conversational response
        st.write(f"Chatbot Response: {conversation[-1]['generated_text']}")
