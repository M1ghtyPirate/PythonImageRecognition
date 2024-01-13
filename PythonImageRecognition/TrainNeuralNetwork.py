import datetime
import matplotlib.pyplot as plt
import random
from keras import layers, models, datasets

(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
training_images, testing_images = training_images / 255, testing_images / 255

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    imgIndex = random.randint(0, len(training_images))
    plt.imshow(training_images[imgIndex], cmap=plt.cm.get_cmap('binary'))
    plt.xlabel(class_names[training_labels[imgIndex][0]])
             
plt.gcf().canvas.manager.set_window_title('Sample training images')
plt.show()

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))

loss, accuracy = model.evaluate(testing_images, testing_labels)
print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')

fileName = f'image_classifier_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.keras'
model.save(fileName)
print(f'Model trained: {fileName}')