# -*- coding: utf-8 -*-
"""DL_CNN_Lattuce.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1CCCfQjIRc_SGe5CG2FZrAmLbro9Tj9JU
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

from google.colab import drive
drive.mount('/gdrive', force_remount=True)

train_dataGenerator = test_dataGenerator= ImageDataGenerator(rescale=1./255)

train_generator = train_dataGenerator.flow_from_directory(
    '/gdrive/My Drive/Lattuce/train', 
    target_size=(24,24),
    batch_size=3,
    class_mode='binary')

test_generator = test_dataGenerator.flow_from_directory(
    '/gdrive/My Drive/Lattuce/test', 
    target_size=(24,24),
    batch_size=3,
    class_mode='binary')

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(24,24,3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot, plot_model


SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

hist = model.fit_generator(
        train_generator,
        steps_per_epoch=15,
        epochs=50,
        validation_data=test_generator,
        validation_steps=5)

print("-- Evaluate --")
scores = model.evaluate_generator(test_generator, steps=5)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

print("-- Predict --")
output = model.predict_generator(test_generator, steps=5)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
print(test_generator.class_indices)
print(output)

from keras.preprocessing import image
test_image = image.load_img('/gdrive/My Drive/Lattuce/predict2.jpg', target_size = (24,24))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
if result[0][0] >= 0.5:
  prediction = 'normal' 
else:
  prediction = 'cercospora'
print(prediction)

import matplotlib.pyplot as plt

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

from keras.models import load_model

model.save('DL_CNN_farm.h5')

from google.colab import files
files.download('DL_CNN_farm.h5')

