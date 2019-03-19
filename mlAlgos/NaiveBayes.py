"""
This is an implementation of a Naive Bayes classifier under the following assumptions:
    1) class labels can be approximated by the multinomial distribution
    2) the likelihood of the class labels can be approximated by a Poisson
       distribution with parameter lambda, for each class.
    3) the parameter lambda has a prior gamma distribution with parameters (2, 1)

The maximum a posteriori estimates are then derived to learn the unknown parameters.
These are overly simplified and convenient assumption, but usually perform well.
"""
from .General import *


def _get_class_probs(y):
    """
    Gets the class proportions (class priors) for the target variable
    :param y: n-length array of targets
    :return: dictionary with keys = target variable,
             values = target variable probability
    """
    class_p = {}
    for class_ in np.unique(y):
        class_p[str(class_)] = (y == class_).sum() / len(y)
    return class_p


def _get_class_lambdas(x, y):
    """
    Gets the likelihood estimates for lambda parameter,
    for each class in y.
    :param x: n-by-d training data feature matrix
    :param y: n-by-1 training data label vector
    :return: dictionary where keys = class labels (as string)
             and values = 1-by-d dimensional arrays
    """
    class_lambdas = {}
    for class_ in np.unique(y):
        sub_x = x[y[:, 0] == class_]
        class_lambdas[str(class_)] = (1 + sub_x.sum(axis=0)) / (len(sub_x) + 1)
    return class_lambdas


def _get_likelihood(features, lambdas):
    """
    Gets the likelihood probability estimate of a test set record
    for a given set of lambda estimates for a given class
    :param features: 1-by-d array of a test set record
    :param lambdas: calculated lambda estimates
    :return: the likelihood estimate (evaluated on a scaled Poisson)
    """
    lamb_raise_x = lambdas ** features
    exp_raise_lamb = np.exp(-1 * lambdas)
    return (lamb_raise_x * exp_raise_lamb).prod()


def predict(x_0, class_p, class_lambdas):
    """
    Evaluates the MAP solution for a given test set record,
    based on learned parameters.
    :param x_0: 1-by-d array of test record features
    :param class_p: class probability estimates
    :param class_lambdas: dictionary of learned lambdas
    :return: highest probability prediction
    """
    max_prob = -1
    prediction = None

    for class_ in class_lambdas:
        likelihood = _get_likelihood(x_0, class_lambdas[class_])
        class_prob = class_p[class_]
        posterior = likelihood * class_prob

        if posterior > max_prob:
            prediction = int(class_)
            max_prob = posterior

    return prediction


def train(x_train, y_train):
    """
    Learns the prior distribution of class and the MAP estimates
    for lambdas in each class under the assumptions outlined at
    the top of this file
    :param x_train: n-by-d matrix of training feature data
    :param y_train: n-by-1 array of training labels
    :return: dictionary of multinomial class probabilities and
             dictionary of learned lambda parameters for all
             feature dimensions for each class
    """
    return _get_class_probs(y_train), _get_class_lambdas(x_train, y_train)


def naive_bayes(x, y, cv=1, test_prop=0.2, scale=False, neg=0,
                randomize=True, poly_expand=1, accuracy=True):
    """
    Performs naive bayes (multi-class) classification and returns scores
    (accuracy or confusion matrix).

    Can also specify whether to scale the data, perform cross validation,
    or whether to perform a polynomial expansion of given degree on the
    feature space.

    :param x: feature data matrix
    :param y: target array
    :param cv: integer number of cross validation folds
    :param test_prop: test proportion when CV = 1
    :param scale: boolean indicating whether to standardize features
    :param randomize: boolean indication whether to shuffle data
    :param poly_expand: degree number to expand feature space
    :param neg: negative class label in target array; used for confusion matrix
    :param accuracy: boolean; if changed to False and it's binary classification,
                     then will return confusion matrix instead of accuracy.
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

        # train model on data
        class_p, class_lambdas = train(x_train, y_train)

        # predict, append confusion matrix to scores list
        y_pred = np.array([predict(row, class_p, class_lambdas) for row in x_test]).reshape(-1, 1)

        if accuracy or len(class_p) != 2:
            scores.append(((y_test - y_pred) == 0).sum()/len(y_pred))
        else:
            scores.append(get_conf_mat(y_pred, y_test, neg))

    return scores

