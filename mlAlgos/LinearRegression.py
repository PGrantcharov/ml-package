"""
File contains methods to perform linear regression, without (linear_regression())
or with (ridge_regression()) regularization.

Can manually train model by calling individual methods from this file and .General,
or can utilize comprehensive methods to perform training, cross-validation (optional),
prediction, and evaluation all in one method.
"""
from .General import *


def get_weights(x, y, lamb=0.0):
    """
    Gets the linear regression coefficient array.
    :param x: training data feature matrix
    :param y: training data output array
    :param lamb: lambda value for ridge regression
    :return: feature regression coefficient array
    """
    lamb_i = lamb * np.identity(x.shape[1])
    x_t_x = np.dot(x.T, x)
    x_t_y = np.dot(x.T, y)
    return np.dot(np.linalg.inv(lamb_i + x_t_x), x_t_y)


def predict(x, w):
    """
    Predicts output values from input matrix and coefficient array
    :param x: feature matrix
    :param w: array of feature weights
    :return: array of output values
    """
    return np.dot(x, w)


def linear_regression(x, y, cv=1, test_prop=0.2, scale=False, randomize=True,
                      lamb=0.0, poly_expand=1, rmse=True):
    """
    Get scores from linear regression model all in one method.

    By default, will train on random 80% of data and return RMSE score on
    other 20%.

    Can perform cross validation by specifying number of splits, change score
    metric to R^2 instead of RMSE, set custom test proportion (ignored if
    CV != 1), scale features, expand feature space by polynomial degree, and
    turn off random shuffling of data.

    :param x: feature data
    :param y: target variable
    :param cv: integer number of cross validation folds
    :param test_prop: test proportion when CV = 1
    :param scale: boolean indicating whether to standardize features
    :param randomize: boolean indication whether to shuffle data
    :param lamb: float lambda regularization parameter >= 0
    :param poly_expand: degree number to expand feature space
    :return: array of RMSE scores for each test set evaluation
    """
    # expand feature space by polynomial
    x = expand_features(x, poly_expand)

    # determine bin sizes for cross validation splits
    bin_sizes = get_bin_sizes(len(y), cv, test_prop)

    # permute order for cross validation
    order = np.random.permutation(len(y)) if randomize else np.arange(0, len(y))

    # initialize array to store RMSE scores from each split
    scores = np.array([])

    for split in range(cv):
        # get train-test split and standardize if scale = True
        x_test, y_test, x_train, y_train = get_split(x, y, bin_sizes, order, split)
        x_train, x_test = standardize(x_train, x_test) if scale else (x_train, x_test)

        # get weights, predict and get RMSE
        weights = get_weights(x_train, y_train, lamb)
        y_pred = predict(x_test, weights)
        score = get_rmse(y_pred, y_test) if rmse else get_r_squared(y_pred, y_test)
        scores = np.append(scores, score)

    return scores


def ridge_regression(x, y, lamb=0.01, cv=1, test_prop=0.8, scale=False,
                     randomize=True, poly_expand=1):
    """
    Get scores from linear regression model all in one method.

    By default, will train on random 80% of data and return RMSE score on
    other 20%.

    Can perform cross validation by specifying number of splits, set custom
    test proportion (ignored if CV != 1), scale features, expand feature
    space by polynomial degree, and turn off random shuffling of data.

    :param x: feature data
    :param y: target variable
    :param lamb: float lambda regularization parameter >= 0
    :param cv: integer number of cross validation folds
    :param test_prop: test proportion when CV = 1
    :param scale: boolean indicating whether to standardize features
    :param randomize: boolean indication whether to shuffle data
    :param poly_expand: degree number to expand feature space
    :return: array of RMSE scores for each test set evaluation
    """
    return linear_regression(x, y, cv, test_prop, scale, randomize, lamb, poly_expand)

