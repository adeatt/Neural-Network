import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()                     #load the data from the api 


class_names = ["T-shirt/top", "Trousers", "Pullover", "Dress", "Coat", "Sancal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

train_images = train_images/255.0                                                               #each pixel as deciaml numbers
test_images = test_images/255.0




model = keras.Sequential([                                                      
    keras.layers.Flatten(input_shape = (28,28)),                                                #Flatten the image(28 by 28 pixels) to align with the 784 input neurons 
    keras.layers.Dense(128, activation="relu"),                                                 #second(hidden) layer with 128 neurons wich is fully connected to the input layer, the activation function is ReLu
    keras.layers.Dense(10, activation="softmax")                                                #softmax haha just the probability of the neurons, needs to add up to 1
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])   #loss decides how much the wheights have to be tuned dependent on how wrong the output is. if we are far away from the correct output we can change the weight by a lot of number. Other way around not so much
 
model.fit(train_images, train_labels, epochs=5)                                                 #gives the same images in a differant order, increases the accursascy of the model                                   

test_loss, test_acc = model.evaluate(test_images, test_labels)

prediction = model.predict(test_images)

for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("Prediction" + class_names[np.argmax(prediction[0])])
    plt.show()

#it only outputs ankle boot as every prediction ?=???????????????????????????????????
