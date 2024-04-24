import tensorflow as tf
import keras                                                                #MNIST fashiono dataset
import numpy as np
import matplotlib.pyplot as plt



fashion_mnist = tf.keras.datasets.fashion_mnist 

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


Class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',       #Defined the train_labes/test_labels(which are just int) with their name( exp: trainlabel 0 = T-shirt/top)
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


train_images = train_images / 255.0                                                      # values have to be sclaed between 0 and 1 (bcs the max. of values is 255, we divide each pixel by 255)
test_images  = test_images / 255.0


plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(Class_names[train_labels[i]])
plt.show()


model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation ='relu'),
    tf.keras.layers.Dense(10)
])