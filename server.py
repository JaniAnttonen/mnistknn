from knn import KNearestNeighbor
import numpy as np
from mnist import MNIST
from flask import Flask, Blueprint, flash, redirect, render_template, request, url_for
from PIL import Image

app = Flask(__name__)


@app.route("/")
def index():
    mndata = MNIST('./data')

    # Load data to variables
    train_images, train_labels = mndata.load_training()
    test_images, test_labels = mndata.load_testing()

    # Create random range of test examples to include
    exampleindeces = np.random.random_integers(0, high=9999, size=4)

    # Assign test data to numpy arrays
    images = np.asarray(test_images)
    labels = np.asarray(test_labels)

    # Construct the KNN classifier
    classifier = KNearestNeighbor()

    # Load the classifier with train data
    classifier.train(np.asarray(train_images), np.asarray(train_labels))

    # Predict the labels with KNN
    predictions = np.rint(classifier.predict(images[exampleindeces], 3))
    # Save ground truth labels for checking if prediction was correct
    truths = labels[exampleindeces]

    i = 1
    for index in exampleindeces:
        two_d = (np.reshape(images[index], (28, 28)) * 255).astype(np.uint8)
        im = Image.fromarray(two_d, 'L')
        filename = "static/" + str(i) + ".png"
        im.save(filename)
        i += 1

    # Render the page
    return render_template('index.html',
                           preds=predictions,
                           truths=truths
                           )

if __name__ == "__main__":
    app.run(host="0.0.0.0")
