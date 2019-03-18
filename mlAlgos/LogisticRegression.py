from .General import *


def get_yx(y_train, x_train):
    """
    Gets the n-by-m shaped matrix needed to calculate
    the derivative to update the weights for a given
    iteration. Used in derivative calculation.
    :param y_train: n-length target vector
    :param x_train: n-by-m shaped feature matrix
    :return: n-by-m shaped matrix
    """
    return np.multiply(y_train, x_train)


def get_sigmoid(x_train, y_train, w):
    """
    Gets the sigmoid function for a given training set.
    :param x_train: feature training set
    :param y_train: training set target variables
    :param w: array of weights for logistic regression
    :return: n-length array (# of data points in training set)
    """
    power = y_train * np.dot(x_train, w)
    return np.exp(power) / (1 + np.exp(power))


def predict(x_test, w):
    """
    Predicts the target variable between negative and
    positive classes (-1 and 1).
    :param x_test: test feature set
    :param w: array of weights for logistic regression
    :return: array of predictions of length x_test.shape[0]
    """
    power = np.dot(x_test, w)
    pred = np.sign((np.exp(power) / (1 + np.exp(power))) - 0.5)
    for i in range(len(pred)):
        pred[i] = np.random.choice([-1, 1]) if pred[i] == 0 else pred[i]
    return pred


def get_first_deriv(sigmoid, yx):
    """
    Gets the gradient from a test set iteration.
    :param sigmoid: n-length sigmoid vector
    :param yx: n-by-m shaped matrix
    :return: m-length array
    """
    return (np.multiply(1 - sigmoid, yx)).sum(axis=0).reshape(-1, 1)


def get_second_deriv(x_train, sigmoid):
    """
    Gets the Hessian matrix from the training feature data and the
    sigmoid function array. Subtracts 10e-3 from diagonal to avoid
    singular matrix issues upon inversion.
    :param x_train: n-by-m matrix
    :param sigmoid: n-length array
    :return: m-by-m Hessian matrix
    """
    inner = np.multiply(sigmoid, (1-sigmoid)).reshape(-1, 1)
    second_der = np.dot(np.transpose(np.multiply(inner, x_train)), x_train)

    # adjust to compensate for singularity issues
    return -1 * second_der - np.identity(second_der.shape[0])*10e-3


def update_weights(x_train, y_train, w, step, yx, newton):
    """
    Gets the updated weights vector. Will follow different procedure
    depending on whether Newton method is followed.
    :param x_train: n-by-m matrix of training feature data
    :param y_train: n length vector of training data targets
    :param w: current m-length w vector
    :param step: step size in case of regular logistic regression
    :param yx: n-by-m matrix used in first derivative
    :param newton: boolean, True = follow Newton method
    :return: updated m-length w vector
    """
    sigmoid = get_sigmoid(x_train, y_train, w)
    first_der = get_first_deriv(sigmoid, yx)
    if newton:
        second_der = get_second_deriv(x_train, sigmoid.flatten())
        return w - np.dot(np.linalg.inv(second_der), first_der)
    return w + step * first_der


def logistic_regression(x, y, cv=1, iterations=1000, step=10e-6,
                        test_prop=0.2, scale=False, randomize=True,
                        newton=False, poly_expand=1):
    """
    Gets confusion matrices from a logistic regression classifier all
    in one function. Cross validation folds can be specified, otherwise
    will be evaluated on a random test set of 20% of input data.

    Can also alter iterations, step-size, whether to use Newton solver,
    whether to scale the data, and whether to perform a polynomial
    expansion on the feature space.

    * Notes:
        - positive and negative class have to be labelled 1 and -1
        - Newton solver prone to overflow
        
    :param x: feature data
    :param y: target variable
    :param cv: integer number of cross validation folds
    :param iterations: number of iterations to increa
    :param step: step size / regularization parameter to increment weight update
    :param test_prop: test proportion when CV = 1
    :param scale: boolean indicating whether to standardize features
    :param randomize: boolean indication whether to shuffle data
    :param newton: boolean of whether to use Newton solver
    :param poly_expand: degree number to expand feature space
    :return:
    """

    # expand feature space by polynomial
    x = expand_features(x, poly_expand, False)

    # determine bin sizes for cross validation splits
    bin_sizes = get_bin_sizes(len(y), cv, test_prop)

    # permute order for cross validation
    order = np.random.permutation(len(y)) if randomize else np.arange(0, len(y))

    # initialize array to store RMSE scores from each split
    scores = []

    for split in range(cv):
        # get train-test split and standardize if scale = True
        x_test, y_test, x_train, y_train = get_split(x, y, bin_sizes, order, split)
        x_train, x_test = standardize(x_train, x_test) if scale else (x_train, x_test)

        # initialize weight vector
        weights = np.zeros(x_train.shape[1]).reshape(-1, 1)

        # define yx component of training function as it is constant between iterations
        yx = get_yx(y_train, x_train)

        # keep updating weights for iteration number
        for i in range(iterations):
            weights = update_weights(x_train, y_train, weights, step, yx, newton)

        y_pred = predict(x_test, weights)
        scores.append(get_conf_mat(y_pred, y_test))

    return scores
