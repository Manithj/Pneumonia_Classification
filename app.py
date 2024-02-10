from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input # import the model`s skeleton
import numpy as np
import tensorflow as tf

model=load_model('chest_xray.h5')

#load the test image
#tf.keras.utils.load_img
img=tf.keras.utils.load_img('image.jpeg',target_size=(224,224))

x=tf.keras.preprocessing.image.img_to_array(img) # image as a numpy array

x=np.expand_dims(x, axis=0)

img_data=preprocess_input(x) # organize for the prediction

classes=model.predict(img_data)

result=int(classes[0][0])

if result== 0:
    print("Person is Affected By PNEUMONIA")
else:
    print("Result is Normal")