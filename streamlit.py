import streamlit as st
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import tensorflow as tf

# Set the title of the web app
st.title('Pneumonia Detection Using VGG16')

st.text("Coded by Manith Jayaba")

# Load the Keras model
model = load_model('chest_xray.h5')

# Create a file uploader for the test image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

# Perform the prediction when an image is uploaded
if uploaded_file is not None:
    img = tf.keras.utils.load_img(uploaded_file, target_size=(224, 224))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    img_data = preprocess_input(x)
    classes = model.predict(img_data)
    result = int(classes[0][0])
    if result == 0:
        st.write("Person is Affected By PNEUMONIA")
    else:
        st.write("Result is Normal")
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)