import tensorflow as tf 
from tensorflow import keras
import numpy as np 
import matplotlib.pyplot as plt 
# make sure you have ^ these libs
# else it's not going to work

data = keras.datasets.mnist

# splits the data into a train set and a test set
(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']

train_images = train_images / 255.0
test_images = test_images / 255.0

# our model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
    ])        

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# trains our model
model.fit(train_images, train_labels, epochs=5)

# finds accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)

# predicts on digits
prediction = model.predict(test_images)

# shows us a plot of the number and then shows prediction and actual digit
for i in range(10):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("Prediction: " + class_names[np.argmax(prediction[i])])
    plt.show()


