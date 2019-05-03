import numpy as np


def standardize(x_train, x_test):
    """
    TODO: vectorize this

    Standardizes the training and test feature matrices
    :param x_train: training data array
    :param x_test: testing data array
    :return: standardized training and testing matrices
    """
    train = x_train.copy()
    test = x_test.copy()
    for col in range(train.shape[1] - 1):
        mean = train[:, col].mean()
        std = train[:, col].std()
        train[:, col] = (train[:, col] - mean) / std
        test[:, col] = (test[:, col] - mean) / std
    return train, test


def get_rmse(pred, test):
    """
    Returns RMSE value for given predictions vs. actual values
    :param pred: prediction array
    :param test: test array
    :return: RMSE value
    """
    return np.sqrt(((pred - test) ** 2).sum() / len(pred))


def get_r_squared(pred, test):
    """
    Returns R^2 value for given predictions vs. actual values
    :param pred: n-by-1 prediction array
    :param test: n-by-1 test array
    :return: R^2 value
    """
    n = len(pred)
    num = n * (pred * test).sum() - (pred.sum() * test.sum())
    den = np.sqrt(((n * (pred ** 2).sum()) - (pred.sum() ** 2)) * ((n * (test ** 2).sum()) - (test.sum() ** 2)))
    return (num / den) ** 2


def expand_features(x, p, offset=True):
    """
    1) Constructs new X matrix for given degree

    2) Adds offset column of 1s to the end of feature matrix

    :param x: original X matrix
    :param p: degree
    :param offset: degree
    :return: feature expanded
    """
    new_x = x.copy()
    for i in range(1, p):
        new_x = np.append(new_x, x ** (i + 1), axis=1)
    new_x = np.append(new_x, np.ones(shape=(x.shape[0], 1)), axis=1) if offset else new_x
    return new_x


def get_split(x, y, bins, rand_order, split):
    """
    Gets train/test split from from feature matrix and target array
    :param x: feature matrix
    :param y: target array
    :param bins: array of bin sizes
    :param rand_order: array of shuffled indices
    :param split: CV fold number
    :return: x_train, y_train, x_test, y_test
    """
    start = bins[:split].sum()
    end = start + bins[split]
    test_index = rand_order[start:end]
    train_index = np.append(rand_order[:start], rand_order[end:])
    return x[train_index], y[train_index], x[test_index], y[test_index]


def get_bin_sizes(n, bins, test_prop):
    """
    An algorithm that gets the number of records that need to be in eacg
    of the bins.
    :param n: number of data points that need to be split
    :param bins: integer representing number of desired bins
    :param test_prop: proportion to use in test set if no cross validation
    :return: array of length 'bins' with integer bin counts
    """
    if bins == 1:
        return np.array([int(n*(1-test_prop)), n - int(n*(1-test_prop))])

    if n % bins == 0:
        bin_sizes = np.repeat(n / bins, bins)
    else:
        bin_sizes = np.array([])
        while bins > 0:
            nearest = np.around(n / bins)
            bin_sizes = np.append(bin_sizes, nearest)
            n = n - nearest
            bins = bins - 1
    return bin_sizes.astype(int)


def get_conf_mat(y_pred, y_test, neg=0):
    """
    Returns confusion matrix from a prediction and test
    set, where the columns correspond to TRUTH and the
    rows correspond to prediction.
    :param y_pred: prediction array of targets
    :param y_test: test array of targets
    :param neg: value of negative target in target set
    :return: 2-by-2 numpy array confusion matrix
    """
    tp = ((y_pred + y_test) == 2).sum()
    tn = ((y_pred + y_test) == 2 * neg).sum()
    fp = ((y_pred - y_test) == 1 - neg).sum()
    fn = ((y_pred - y_test) == -1 + neg).sum()
    return np.array([[tp, fp], [fn, tn]])


def pred_accuracy(conf_matrix):
    """
    Gets the prediction accuracy from a confusion matrix
    and returns accuracy as single float.
    :param conf_matrix: confusion matrix
    :return: float
    """
    return np.trace(conf_matrix/conf_matrix.sum())
