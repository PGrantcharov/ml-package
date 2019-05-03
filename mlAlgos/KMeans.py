import numpy as np


def _min_mu(val, mu):
    return np.argmin(np.sum((val - mu) ** 2, axis=1))


# updaters
def update_c(dist, mu):
    return np.array([_min_mu(val, mu) for val in dist])


def update_mu(dist, c, k):
    return np.array([np.mean(dist[c == i], axis=0) for i in range(k)])


# loss function
def get_loss(dist, mu, c):
    return np.sum([np.sum((dist[c == i] - mu[i]) ** 2) for i in np.unique(c)])


def kmeans(k, data, iterations=20):
    """
    Provide data and parameter K to get a
    :param data: n-by-m matrix of n data points with m dimension
    :param k: number of clusters
    :param iterations: number of iterations to run algorithm
    :return:
    """
    # instantiate
    mu = np.random.multivariate_normal(mean=np.mean(data, axis=0), cov=np.identity(data.shape[1]), size=k)
    c = update_c(data, mu)
    loss = np.array([get_loss(data, mu, c)])

    # iterate for 20 loops
    for i in range(iterations):
        c = update_c(data, mu)
        mu = update_mu(data, c, k)
        loss = np.append(loss, get_loss(data, mu, c))

    return mu, c, loss[1:]

