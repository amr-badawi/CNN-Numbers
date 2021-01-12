import tensorflow as tf
from tensorflow import keras 
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten 
from keras.utils import to_categorical
from PIL import Image
import numpy as np
import glob


#given any image file path the function can predict the label of the image
directory = r"write path here\trainingSet/"
# image_path="Write the image path here"
image_path=directory+"6/img_1704.jpg"  #random image of number 6 

def Predict(image_path):
    
    Pimg=[]
    #Create the model
    #set the hyperparameters
    num_filters = 8
    filter_size = 3
    pool_size = 2
    model = Sequential([Conv2D(num_filters, filter_size, input_shape=(28, 28, 1)),MaxPooling2D(pool_size=pool_size),
    Flatten(),Dense(10, activation='softmax'),])
    
    #load the .h5 model 
    model.load_weights('cnn.h5')
    
    #Load Image from the file path, load in numpy array ,expand the dimensions for keras
    
    img=Image.open(image_path)
    imgs=np.array(img)
    Pimg.append(imgs)
    Pimg=np.array(Pimg)
    Pimg=np.expand_dims(Pimg, axis=3)
    #predict
    Plbl=model.predict(Pimg[:1]) 
    print(np.argmax(Plbl, axis=1)) #printed label should be 6 
    
  
Predict(image_path)    
    
    
    