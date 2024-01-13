import datetime
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from keras import layers, models, datasets
from PIL import Image

def getFiles(pattern):
    return [f for f in os.listdir('.') if os.path.isfile(f) and re.fullmatch(pattern, f)]

def identify(img):
    prediction = model.predict(np.array([img]) / 255)
    index = np.argmax(prediction)
    return f'{class_names[index]} - {round(prediction[0][index] * 100, 2)} %'

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

modelFiles = getFiles('(image_classifier_)[0-9]{8}-[0-9]{6}(.keras)')
if modelFiles == None or modelFiles.__len__() == 0:
    print('No model found. Use "train".')
    exit()
model = models.load_model(modelFiles[-1], compile=False)
print(f'Model loaded: {modelFiles[-1]}')

imageFiles = getFiles('.*(.jpg)|.*(.jpeg)')
if imageFiles == None or imageFiles.__len__() == 0:
    print('No JPEG images found for identification.')
    exit()

result = ''

for i in range(imageFiles.__len__() * 2):
    if i % 2 == 0: continue
    imgIndex = i // 2
    img = Image.open(imageFiles[imgIndex])
    i = i % 16
    plt.subplot(4, 4, i)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.get_cmap('binary'))
    plt.xlabel(imageFiles[imgIndex])
    img = img.resize((32, 32))
    plt.subplot(4, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.get_cmap('binary'))
    imgClass = identify(img)
    result += f'{imageFiles[imgIndex]}: {imgClass}\n'
    plt.xlabel(imgClass)
    if i == 1:
        plt.gcf().canvas.manager.set_window_title('Identification result')
    elif i == 15:
        plt.figure()

print(f'Identification result:\n{result}')
resultFileName = f'identification_result_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.txt'
resultFile = open(resultFileName, 'w')
resultFile.write(result)
print(f'Identification result saved to: {resultFileName}')
resultFile.close()
plt.show()