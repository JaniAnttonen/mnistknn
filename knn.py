import numpy as np
import pandas as pd
from mnist import MNIST


class KNearestNeighbor(object):
    """
    K nearest neighbors classifier
    University of Turku course project by
    @author Jani Anttonen
    """
    def __init__(self):
        print "Constructing the classifier."
        self.X_train = None
        self.y_train = None

    def train(self, X, y):
        """
        Just saves the data to the
        object for referencing later
        """
        print ("Loading " +
               str(X.shape[0]) +
               " training images to memory.")

        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1):
        """
        Main method for predicting a class for MNIST data
        """
        self.printSeparator()
        dists = self.compute_distances(X)

        return self.predict_labels(dists, k=k)

    def compute_distances(self, X):
        """
        L2 distance (euclidean)
        """
        print ("Computing distances for " +
               str(X.shape[0]) +
               " images against training data.")
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]

        # Create a distance matrix that's the same size as the input
        dists = np.zeros((num_test, num_train))

        M = np.dot(X, self.X_train.T)

        te = np.square(X).sum(axis=1)
        tr = np.square(self.X_train).sum(axis=1)

        dists = np.sqrt(-2 * M + tr + np.matrix(te).T)

        return dists

    def compute_confusion_matrix(self, all_data, all_labels, k=1):
        """
        Computes the confusion matrix of all data and its predictions
        (false positives and true negatives)
        """
        self.printSeparator()
        print ("Computing the confusion matrix for " +
               str(all_data.shape[0]) +
               " testing images and " +
               str(k) +
               " nearest neighbors.")
        # Initialize confusion matrix
        conf = np.zeros((10, 10), dtype=int)
        # Initialize count of correct predictions
        correct = 0

        # Predict labels for all data
        dists = self.compute_distances(all_data)
        preds = self.predict_labels(dists, k)

        # Compare predictions to ground truth
        for i in xrange(all_labels.shape[0]):
            conf[all_labels[i], int(preds[i])] += 1
            if all_labels[i] == int(preds[i]):
                correct += 1

        conf = pd.DataFrame(data=conf, index=range(0, 10), columns=range(0, 10))
        
        self.printSeparator()
        print "Predicted " + str(correct) + "/" + str(all_labels.shape[0]) + " correctly."
        print "Amounting to " + str(float(correct)/float(all_labels.shape[0])*100) + " % prediction accuracy."

        return conf


    def predict_labels(self, dists, k=1):
        """
        Predicts the labels for test data
        by checking k (default 1) nearest
        neighbors.
        """
        print "Predicting labels for testing images based on the distances."
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in xrange(num_test):
            closest_y = []
            labels = self.y_train[np.argsort(dists[i, :])].flatten()
            closest_y = labels[0:k]
            counts = [0] * 10

            for luoks in closest_y[:k]:
                counts[luoks] += 1

            max_count = max(counts)
            closest = counts.index(max_count)

            y_pred[i] = closest

        return y_pred

    def printSeparator(self):
        print "----------------------------------------------------------------"


def main(argv=None):
    """
    Main program loop, which loads
    the data and passes it to the classifier
    """
    # Specify folder with the data
    mndata = MNIST('./data')

    # Load data to variables
    train_images, train_labels = mndata.load_training()
    test_images, test_labels = mndata.load_testing()

    # Create random range of test examples to include
    exampleindeces = np.random.random_integers(0, high=9999, size=100)

    # Assign test data to numpy arrays
    images = np.asarray(test_images)
    labels = np.asarray(test_labels)

    # Construct the KNN classifier
    classifier = KNearestNeighbor()

    # Load the classifier with train data
    classifier.train(np.asarray(train_images), np.asarray(train_labels))

    # Predict the labels with KNN
    # predictions = classifier.predict(images[exampleindeces], 3)
    # Save ground truth labels for checking if prediction was correct
    # truths = labels[exampleindeces]

    # Compute the confusion matrix
    conf = classifier.compute_confusion_matrix(images, labels, 3)
    print conf

    # Create an array of boolean values of the prediction accuracy
    # checks = [predictions[i] == truths[i] for i in range(predictions.shape[0])]

    # Print the results
    # for check in checks:
    #     print check




if __name__ == '__main__':
    main()
