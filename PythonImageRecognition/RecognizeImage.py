import datetime
import cv2 as cs
import numpy as np
import matplotlib.pyplot as plt
import random
from keras import Sequential, layers, models, datasets
from PIL import Image

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

model = models.load_model('image_classifier_20240111-192221.keras', compile=False)

img = Image.open('huhcat.jpg')
img = img.resize((32, 32))

prediction = model.predict(np.array([img]) / 255)
index = np.argmax(prediction)

plt.xticks([])
plt.yticks([])
plt.imshow(img, cmap=plt.cm.get_cmap('binary'))
#plt.xlabel(f'Prediction ({round(prediction[0][index] * 100, 2)} %): {class_names[index]}')
label =f'Predictions:\n'
for i in range(10):
    if i == index:
        label += f'*'
    label += f'{round(prediction[0][i] * 100, 2)} % - {class_names[i]}\n'
plt.xlabel(label)
plt.show()