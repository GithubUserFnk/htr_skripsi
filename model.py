import tensorflow as tf 
import numpy as np
import os
import mnist # get data from
import matplotlib.pyplot as plt # grph
from keras.models import Sequential # ann architecture
from keras.layers import Dense # Layer in ann
from keras.utils import to_categorical
from keras.models import load_model

# import dataset mnist untuk menggunakan train dan test 
mnist = tf.keras.datasets.mnist

# normize the images
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, test_images = (train_images / 255) - 0.5, (test_images / 255) - 0.5

train_images = train_images.reshape((-1, 784))
test_images = test_images.reshape((-1, 784))

#Defining the model
def baselineK_model():
    # create model
    model = Sequential()
    model.add( Dense(64, activation='relu', input_dim=784))
    model.add( Dense(64, activation='relu'))
    model.add( Dense(10, activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# In[Main]
if not (os.path.exists('./data/modelnewv1.h5')):
    print("No model found, creating new")
    model = baselineK_model()
    print(model.summary())
    
    # train model
    model.fit(
        train_images,
        to_categorical(train_labels),
        epochs=500,
        batch_size=32
    )
    model.save('./data/modelnewv1.h5')
else:
    print('Model found')
    model = load_model('./data/modelnewv1.h5')
    print(model.summary())
    
# evaluate the model
scores = model.evaluate(
    test_images,
    to_categorical(test_labels)
)
print("ANN Error: %.2f%%" % (100-scores[1]*100))