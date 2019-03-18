from .General import *


def get_class_probs(y):
    """
    Gets the class probabilities for the target variable
    :param y: n-length array of targets
    :return: dictionary with keys = target variable,
             values = target variable probability
    """
    class_p = {}
    for class_ in np.unique(y):
        class_p[str(class_)] = (y == class_).sum() / len(y)
    return class_p


def get_class_lambdas(x, y):
    class_lambdas = {}
    for class_ in np.unique(y):
        sub_x = x[y[:, 0] == class_]
        class_lambdas[str(class_)] = (1 + sub_x.sum(axis=0)) / (len(sub_x) + 1)
    return class_lambdas


def get_likelihood(features, lambdas):
    lamb_raise_x = lambdas ** features
    exp_raise_lamb = np.exp(-1 * lambdas)
    return (lamb_raise_x * exp_raise_lamb).prod()


def predict(x_0, class_p, class_lambdas):
    max_prob = -1
    prediction = None

    for class_ in class_lambdas:
        likelihood = get_likelihood(x_0, class_lambdas[class_])
        class_prob = class_p[class_]
        posterior = likelihood * class_prob

        if posterior > max_prob:
            prediction = int(class_)
            max_prob = posterior

    return prediction


def cross_val(x, y, splits):
    bin_sizes = get_bin_sizes(len(y), splits)

    rand_order = np.arange(0, len(y))
    np.random.shuffle(rand_order)

    scores = np.array([[0, 0], [0, 0]])
    lambdas = np.array([])
    for split in range(len(bin_sizes)):
        # partition data
        x_train, y_train, x_test, y_test = get_split(x, y, bin_sizes, rand_order, split)

        # train model on data
        class_p = get_class_probs(y_train)
        class_lambdas = get_class_lambdas(x_train, y_train)

        lambdas = np.append(lambdas, class_lambdas)

        # predict on feature test set
        y_pred = np.array([predict(row, class_p, class_lambdas) for row in x_test]).reshape(-1, 1)

        # evaluate on test set
        scores = scores + get_scores(y_pred, y_test)

    return scores, lambdas

