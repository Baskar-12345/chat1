import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions # type: ignore

import numpy as np

class ImageRecognizer:
    def __init__(self):
        self.model = MobileNetV2(weights='imagenet')

    def recognize(self, img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = self.model.predict(x)
        return decode_predictions(preds, top=3)[0]
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

class ChatBot:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = TFGPT2LMHeadModel.from_pretrained("gpt2")

    def respond(self, text):
        inputs = self.tokenizer.encode(text, return_tensors="tf")
        outputs = self.model.generate(inputs, max_length=50, num_return_sequences=1)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

import streamlit as st
from PIL import Image

def main():
    st.title("AI Chatbot with Image Recognition")

    st.sidebar.header("Upload an Image")
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Save uploaded file
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Recognize objects in the image
        recognizer = ImageRecognizer()
        predictions = recognizer.recognize("temp_image.jpg")
        st.write("Object Detection Results:")
        for pred in predictions:
            st.write(f"{pred[1]}: {pred[2]*100:.2f}%")

    st.sidebar.header("Chat with AI")
    user_input = st.sidebar.text_input("You:", "")
    
    if user_input:
        chatbot = ChatBot()
        response = chatbot.respond(user_input)
        st.sidebar.text_area("AI:", response, height=200)

if __name__ == "__main__":
    main()
