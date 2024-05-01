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


model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation ='relu'),
    tf.keras.layers.Dense(10)
])


model.compile(optimizer = 'adam',
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics =  ['accuracy'])


model.fit(train_images, train_labels, epochs=10)


test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print("\n Test accuarcy: ", test_acc, "Test_loss:", test_loss)


probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images)

