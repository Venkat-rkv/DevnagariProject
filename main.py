import os
from contextlib import redirect_stdout

import imageio
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix)
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (BatchNormalization, Conv2D, Dense,
                                     Dropout, Flatten, MaxPooling2D)


def run():
    # Loading the dataset
    # https://machinelearningmastery.com/how-to-load-large-datasets-from-directories-for-deep-learning-with-keras/
    train = '/Users/venkat/PycharmProjects/ML_Project_Hindi/DevanagariHandwrittenChar/Test'
    test = '/Users/venkat/PycharmProjects/ML_Project_Hindi/DevanagariHandwrittenChar/Train'

    # Pre-processing of data
    # Loading images from folders
    # Referred from https://stackoverflow.com/questions/30230592/loading-all-images-using-imread-from-a-given-folder
    train_img = []
    train_labels = []
    for inner_train in os.listdir(train):
        if 'character' not in inner_train:
            continue
        y = inner_train.split('_')[-1]
        for item in os.listdir(train + "/" + inner_train):
            file = imageio.imread(train + "/" + inner_train + "/" + item)
            train_img.append(file)
            train_labels.append(y)
    # Convert into numpy arrays
    train_img = np.array(train_img)
    train_labels = np.array(train_labels)

    test_img = []
    test_labels = []
    for inner_test in os.listdir(train):
        if 'character' not in inner_test:
            continue
        y = inner_test.split("_")[-1]
        for item in os.listdir(test + "/" + inner_test):
            file = imageio.imread(test + "/" + inner_test + "/" + item)
            test_img.append(file)
            test_labels.append(y)
    # Convert into numpy arrays
    test_img = np.array(test_img)
    test_labels = np.array(test_labels)

    # Finding labels
    # Referred from https://www.programiz.com/python-programming/methods/dictionary/fromkeys
    unique_train = list(dict.fromkeys(train_labels))
    print(unique_train)
    unique_test = list(dict.fromkeys(test_labels))
    print(unique_test)

    # Setting-up Labels
    char = {'pa': 0, 'na': 1, 'ga': 2, 'dhaa': 3, 'ka': 4, 'ra': 5,
            'taa': 6, 'ja': 7, 'daa': 8, 'thaa': 9, 'cha': 10}

    # Mapping labels to the characters and storing them as list
    train_labels_char = []
    for i in train_labels:
        train_labels_char.append(char[i])

    test_labels_char = []
    for i in test_labels:
        test_labels_char.append(char[i])

    # Converting list into numpy array and checking on data distribution
    # Referred from https://numpy.org/doc/stable/reference/generated/numpy.bincount.html
    train_labels_char = np.array(train_labels_char)
    print(train_labels_char)
    print(np.bincount(train_labels_char))

    test_labels_char = np.array(test_labels_char)
    print(test_labels_char)
    print(np.bincount(test_labels_char))

    # VGG-3 Architecture
    # Referred from https://furahadamien.com/papers/artificial_neural_nets.pdf
    vgg3_model = Sequential(
        [
            Conv2D(64, kernel_size=(3, 3), activation='relu', padding="same", strides=1, input_shape=(32, 32, 1)),
            Conv2D(64, kernel_size=(3, 3), activation='relu', padding="same", strides=1, input_shape=(32, 32, 1)),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(128, kernel_size=(3, 3), activation='relu', padding="same", strides=1, input_shape=(32, 32, 1)),
            Conv2D(128, kernel_size=(3, 3), activation='relu', padding="same", strides=1, input_shape=(32, 32, 1)),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(256, kernel_size=(3, 3), activation='relu', padding="same", strides=1, input_shape=(32, 32, 1)),
            MaxPooling2D(pool_size=(2, 2)),
            BatchNormalization(),

            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.25),
            Dense(11, activation='softmax'),
        ]
    )

    # LeNet Architecture
    # Referred from https://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/
    lenet_model = Sequential(
        [
            Conv2D(32, kernel_size=(3, 3), activation='relu', padding="same", strides=1, input_shape=(32, 32, 1)),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(64, kernel_size=(3, 3), activation='relu', padding="same", strides=1, input_shape=(32, 32, 1)),
            MaxPooling2D(pool_size=(2, 2)),

            BatchNormalization(),
            Dropout(0.25),

            Flatten(),

            Dense(512, activation='relu'),
            Dense(11, activation='softmax'),
        ]
    )

    # VGG-13 Architecture
    # https://pytorch.org/hub/pytorch_vision_vgg/
    vgg13_model = Sequential(
        [
            Conv2D(64, kernel_size=(3, 3), activation='relu', padding="same", strides=1, input_shape=(32, 32, 1)),
            Conv2D(64, kernel_size=(3, 3), activation='relu', padding="same", strides=1, input_shape=(32, 32, 1)),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(128, kernel_size=(3, 3), activation='relu', padding="same", strides=1, input_shape=(32, 32, 1)),
            Conv2D(128, kernel_size=(3, 3), activation='relu', padding="same", strides=1, input_shape=(32, 32, 1)),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(256, kernel_size=(3, 3), activation='relu', padding="same", strides=1, input_shape=(32, 32, 1)),
            Conv2D(256, kernel_size=(3, 3), activation='relu', padding="same", strides=1, input_shape=(32, 32, 1)),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(512, kernel_size=(3, 3), activation='relu', padding="same", strides=1, input_shape=(32, 32, 1)),
            Conv2D(512, kernel_size=(3, 3), activation='relu', padding="same", strides=1, input_shape=(32, 32, 1)),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(512, kernel_size=(3, 3), activation='relu', padding="same", strides=1, input_shape=(32, 32, 1)),
            Conv2D(512, kernel_size=(3, 3), activation='relu', padding="same", strides=1, input_shape=(32, 32, 1)),
            MaxPooling2D(pool_size=(2, 2)),
            BatchNormalization(),
            Dropout(0.25),

            Flatten(),

            Dense(512, activation='relu'),
            Dense(128, activation='relu'),
            Dense(11, activation='softmax'),
        ]
    )

    # VGG-16 Architecture
    # Referred from https://www.geeksforgeeks.org/vgg-16-cnn-model/
    vgg16_model = Sequential(
        [
            Conv2D(64, kernel_size=(3, 3), activation='relu', padding="same", strides=1, input_shape=(32, 32, 1)),
            Conv2D(64, kernel_size=(3, 3), activation='relu', padding="same", strides=1, input_shape=(32, 32, 1)),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(128, kernel_size=(3, 3), activation='relu', padding="same", strides=1, input_shape=(32, 32, 1)),
            Conv2D(128, kernel_size=(3, 3), activation='relu', padding="same", strides=1, input_shape=(32, 32, 1)),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(256, kernel_size=(3, 3), activation='relu', padding="same", strides=1, input_shape=(32, 32, 1)),
            Conv2D(256, kernel_size=(3, 3), activation='relu', padding="same", strides=1, input_shape=(32, 32, 1)),
            Conv2D(256, kernel_size=(3, 3), activation='relu', padding="same", strides=1, input_shape=(32, 32, 1)),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(512, kernel_size=(3, 3), activation='relu', padding="same", strides=1, input_shape=(32, 32, 1)),
            Conv2D(512, kernel_size=(3, 3), activation='relu', padding="same", strides=1, input_shape=(32, 32, 1)),
            Conv2D(512, kernel_size=(3, 3), activation='relu', padding="same", strides=1, input_shape=(32, 32, 1)),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(512, kernel_size=(3, 3), activation='relu', padding="same", strides=1, input_shape=(32, 32, 1)),
            Conv2D(512, kernel_size=(3, 3), activation='relu', padding="same", strides=1, input_shape=(32, 32, 1)),
            Conv2D(512, kernel_size=(3, 3), activation='relu', padding="same", strides=1, input_shape=(32, 32, 1)),
            MaxPooling2D(pool_size=(2, 2)),
            BatchNormalization(),
            Dropout(0.25),

            Flatten(),

            Dense(4096, activation='relu'),
            Dense(4096, activation='relu'),
            Dense(11, activation='softmax'),
        ]
    )

    # compile the network for fitting
    # referred from https://numpy.org/doc/stable/reference/generated/numpy.save.html
    # Referred from https://www.programcreek.com/python/example/94454/contextlib.redirect_stdout
    vgg3_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    vgg3_model.summary()
    with open('VGG3_modelsummary.npy', 'w') as f:
        with redirect_stdout(f):
            vgg3_model.summary()

    lenet_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    lenet_model.summary()
    with open('LeNet_modelsummary.npy', 'w') as f:
        with redirect_stdout(f):
            lenet_model.summary()

    vgg13_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    vgg13_model.summary()
    with open('VGG13_modelsummary.npy', 'w') as f:
        with redirect_stdout(f):
            vgg13_model.summary()

    vgg16_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    vgg16_model.summary()
    with open('VGG16_modelsummary.npy', 'w') as f:
        with redirect_stdout(f):
            vgg16_model.summary()

    # Fit the Model
    # referred from https://keras.io/guides/training_with_built_in_methods/
    VGG3_model = vgg3_model.fit(train_img, train_labels_char, batch_size=32, epochs=6,
                                validation_data=(test_img, test_labels_char), verbose=1)
    np.save('VGG3_Model_Score.npy', VGG3_model)

    LeNet_model = lenet_model.fit(train_img, train_labels_char, batch_size=32, epochs=6,
                                  validation_data=(test_img, test_labels_char), verbose=1)
    np.save('LeNet_Model_Score.npy', LeNet_model)

    VGG13_model = vgg13_model.fit(train_img, train_labels_char, batch_size=32, epochs=6,
                                  validation_data=(test_img, test_labels_char), verbose=1)
    np.save('VGG13_Model_Score.npy', VGG13_model)

    VGG16_model = vgg16_model.fit(train_img, train_labels_char, batch_size=32, epochs=6,
                                  validation_data=(test_img, test_labels_char), verbose=1)
    np.save('VGG16_Model_Score.npy', VGG16_model)

    def plot_utils(model, fittedmodel, test_img, test_labels_char, model_name):
        exactname = model_name.split(".npy")[0]
        # Visualization of loss and accuracy for models incorporated
        # Referred from https://www.pluralsight.com/guides/data-visualization-deep-learning-model-using-matplotlib
        plt.plot(fittedmodel.history['loss'], 'r', label='Training Loss')
        plt.plot(fittedmodel.history['val_loss'], 'g', label='Validation Loss')
        plt.title(exactname + ' - Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(exactname + ' - Loss.png')
        plt.figure()

        plt.plot(fittedmodel.history['accuracy'], 'r', label='Training Accuracy')
        plt.plot(fittedmodel.history['val_accuracy'], 'g', label='Validation Accuracy')
        plt.title(exactname + ' - Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(exactname + ' - Accuracy.png')
        plt.show()

        # Model's performance is evaluated on testdata
        score_test = model.evaluate(test_img, test_labels_char)
        np.save(model_name + ' - Testing_Score', score_test)
        print('Loss: {:.4f}'.format(score_test[0]))
        print('Accuracy: {:.4f}'.format(score_test[1]))
        print('Error: {:.4f}'.format(1 - score_test[1]))

        # Classification Report
        # Referred from https://stackoverflow.com/questions/54167910/keras-how-to-use-argmax-for-predictions
        # Referred from https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
        predictions = model.predict(test_img)
        print(predictions)
        prediction_onehot = np.argmax(predictions, axis=1)
        print(prediction_onehot)
        characters = ['0:pa', '1:na', '2:ga', '3:dhaa', '4:ka',
                      '5:ra', '6:taa', '7:ja', '8:daa', '9:thaa', '10:cha']
        ClassificationReport = classification_report(test_labels_char, prediction_onehot, target_names=characters)
        print(ClassificationReport)
        np.save(model_name + ' - Classification_Report.npy', ClassificationReport)

        # Confusion Matrix
        # Referred from https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html
        cm = confusion_matrix(test_labels_char, prediction_onehot)
        print(cm)
        labels = ['pa', 'na', 'ga', 'dhaa', 'ka',
                  'ra', 'taa', 'ja', 'daa', 'thaa', 'cha']
        graph = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        graph.plot()
        plt.savefig(exactname + ' - Confusion_Matrix.png')
        plt.show()

    plot_utils(vgg3_model, VGG3_model, test_img, test_labels_char, "VGG3 Model")
    plot_utils(lenet_model, LeNet_model, test_img, test_labels_char, "LeNet Model")
    plot_utils(vgg13_model, VGG13_model, test_img, test_labels_char, "VGG13 Model")
    plot_utils(vgg16_model, VGG16_model, test_img, test_labels_char, "VGG16 Model")


if __name__ == "__main__":
    run()
