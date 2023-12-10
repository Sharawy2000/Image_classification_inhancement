import numpy as np
import os
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tqdm import tqdm

class_names = ['mountain','sea', 'forest']
class_names_label = {class_name: i for i, class_name in enumerate(class_names)}
nb_classes = len(class_names)
IMAGE_SIZE = (150, 150)

def Load_data():
    datasets = ['seg_train', 'seg_test']
    output = []

    for dataset in datasets:
        images = []
        labels = []

        print("Loading {}".format(dataset))

        for folder in os.listdir(dataset):
            label = class_names_label[folder]

            for file in tqdm(os.listdir(os.path.join(dataset, folder))):
                img_path = os.path.join(os.path.join(dataset, folder), file)
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, IMAGE_SIZE)

                images.append(image)
                labels.append(label)

        images = np.array(images, dtype='float32')
        labels = np.array(labels, dtype='int32')
        output.append((images, labels))
    return output

def split_data(images, labels):
    (train_images, train_labels), (test_images, test_labels) = Load_data()
    train_images, train_labels = shuffle(images, labels, random_state=25)
    n_train = train_labels.shape[0]
    n_test = test_labels.shape[0]
    print("Number of training examples: {}".format(n_train))
    print("Number of testing examples: {}".format(n_test))
    print("Each image is of size: {}".format(IMAGE_SIZE))
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    return (train_images, train_labels), (test_images, test_labels)

def CNN_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(6, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
    return model
def display_image(image, label):
    plt.figure()
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.title('Image label: ' + class_names[label])
    plt.show()


def main(image_path):
    data = Load_data()
    (train_images, train_labels), (test_images, test_labels) = split_data(data[0][0], data[0][1])
    model = CNN_model()
    history = model.fit(train_images, train_labels, batch_size=128, epochs=20, validation_split=0.2)
    val_loss, val_acc = model.evaluate(test_images, test_labels)
    print(f'Validation loss: {val_loss:.4f}')
    print(f'Validation accuracy: {val_acc:.4f}')

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    processed_image = cv2.resize(image, IMAGE_SIZE)
    processed_image = np.expand_dims(processed_image, axis=0)
    processed_image = processed_image / 255.0
    prediction = model.predict(processed_image)
    pred_label = np.argmax(prediction[0])
    display_image(image, pred_label)

if __name__ == "__main__":
    image_path = "input_image.jpg"
    main(image_path)