"""
This implementation of the k-nearest neighbor algorithm allows for
specification of p-norm to be used to determine distance between
points and the number of neighbors to include in vote.

Please see knn() method to learn of other ways to get the most out
of this algorithm, such as by automatically performing cross
validation, performing polynomial feature expansion, etc.
"""
from .General import *


def _get_dist(x_train, row, norm):
    """
    Gets the p-norm distance between each record in the training
    set and the row.
    :param x_train: n-by-d matrix of training features
    :param row: 1-by-d array
    :param norm: p-norm distance metric
    :return: returns n-by-1 array of p-norm distances between
             each of the n training records and the test row
    """
    return ((abs(x_train - row) ** norm).sum(axis=1) ** (1/norm)).reshape(-1, 1)


def _get_knn(nn, k):
    """
    Gets the k-nearest neighbors based on distance metric in column 0
    :param nn: an n-by-2 matrix of distances and labels
    :param k: number of closest neighbors to return
    :return: a sorted n-by-2 array of the k nearest neighbors, in order
             of increasing distance
    """
    return nn[nn[:, 0].argsort()][:k, 1]


def _get_vote(classes):
    """
    Takes array of k-nearest neighbor classes and returns the most
    frequent class. Removes furthest neighbor in case of tie.
    :param classes: array of classes
    :return: most frequent class
    """
    max_count = 0
    max_class = None
    tie = False

    for class_ in set(classes):
        count = (classes == class_).sum()
        if count < max_count:
            continue
        elif count == max_count:
            tie = True
        else:
            tie = False
            max_count = count
            max_class = class_
    return _get_vote(classes[:-1]) if tie else max_class


def predict(x_train, y_train, x_test, k, norm):
    """
    Predicts the class label from an array feature matrix
    test set.
    :param x_train: n-by-d matrix of training features
    :param y_train: n-by-1 vector of training labels
    :param x_test: m-by-d matrix of test features
    :param k: number of nearest neighbors to include in vote
    :param norm: float, p-norm distance metric
    :return: m-by-1 vector of predict test set labels
    """
    y_pred = np.array([])
    for test_row in x_test:
        distance = _get_dist(x_train, test_row, norm)
        nn = _get_knn(np.concatenate([distance, y_train], axis=1), k)
        y_pred = np.append(y_pred, _get_vote(nn))
    return y_pred.reshape(-1, 1)


def knn(x, y, k=3, norm=2, neg=-1, cv=1, test_prop=0.2, scale=False,
        randomize=True, poly_expand=1, accuracy=True):
    """
    Performs k-nearest neighbor (multi-class) classification and returns
    scores (accuracy or confusion matrix).

    Can also specify whether to scale the data, perform cross validation,
    or whether to perform a polynomial expansion of given degree on the
    feature space.

    :param x: feature data
    :param y: target variable
    :param k: number of nearest neighbors to include in votes
    :param norm: p-norm to calculate distance between points
    :param cv: integer number of cross validation folds
    :param test_prop: test proportion when CV = 1
    :param scale: boolean indicating whether to standardize features
    :param randomize: boolean indication whether to shuffle data
    :param poly_expand: degree number to expand feature space
    :param accuracy: boolean; if changed to False and it's binary classification,
                     then will return confusion matrix instead of accuracy.
    :param neg: label for negative class if binary and wants confusion matrix
                ignored if accuracy = True
    :return: list of accuracy scores for each test set evaluation, unless
             accuracy == False and it's a binary classification problem - then
             it will return a list of confusion matrices
    """

    # expand feature space by polynomial
    x = expand_features(x, poly_expand, False)

    # determine bin sizes for cross validation splits
    bin_sizes = get_bin_sizes(len(y), cv, test_prop)

    # permute order for cross validation
    order = np.random.permutation(len(y)) if randomize else np.arange(0, len(y))

    # initialize list to store accuracies OR confusion matrices from each split
    scores = []

    for split in range(cv):
        # get train-test split and standardize if scale = True
        x_test, y_test, x_train, y_train = get_split(x, y, bin_sizes, order, split)
        x_train, x_test = standardize(x_train, x_test) if scale else (x_train, x_test)

        # predict on feature test set
        y_pred = predict(x_train, y_train, x_test, k, norm)

        if accuracy or len(np.unique(y)) != 2:
            scores.append(((y_test - y_pred) == 0).sum() / len(y_pred))
        else:
            scores.append(get_conf_mat(y_pred, y_test, neg))

    return scores

