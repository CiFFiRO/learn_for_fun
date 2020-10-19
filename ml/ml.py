import numpy as np
import random
import matplotlib.pyplot
import time
import math


def regression(data, t, phi, M, param):
    N = data.shape[0]
    PHI = np.zeros((N, M), dtype=np.float32)
    for i in range(N):
        PHI[i] = np.array([phi(j, data[i]) for j in range(M)], dtype=np.float32)
    w = np.linalg.inv((PHI.transpose().dot(PHI) + param * np.identity(M, dtype=np.float32))).dot(PHI.transpose()).dot(t)

    def predict(x):
        phi_vec = np.array([phi(i, x) for i in range(M)])
        return w.dot(phi_vec)

    return predict


def split_data(data, t):
    N = data.shape[0]
    M = data.shape[1]
    data_copy = np.zeros((N, M+1), dtype=np.float32)
    for i in range(N):
        data_copy[i][:M] = data[i]
        data_copy[i][M] = t[i]

    np.random.shuffle(data_copy)
    new_t = data_copy[:, M]
    data_copy = data_copy[:, 0:M]

    train = (data_copy[0:math.floor(0.6*N)], new_t[0:math.floor(0.6*N)])
    valid = (data_copy[train[0].shape[0]:train[0].shape[0]+math.floor(0.2*N)], new_t[train[1].shape[0]:train[1].shape[0]+math.floor(0.2*N)])
    test = (data_copy[train[0].shape[0]+valid[0].shape[0]:], new_t[train[1].shape[0]+valid[1].shape[0]:])
    return train, valid, test


def MSE(model, data, t):
    result = 0.0
    for i in range(data.shape[0]):
        result += (t[i] - model(data[i]))**2
    return result / data.shape[0]


def generate_data(size, dim, gen_coord, func, eps):
    data = np.zeros((size, dim), dtype=np.float32)
    t = np.zeros((size, 1), dtype=np.float32)

    for i in range(size):
        for j in range(dim):
            data[i][j] = gen_coord()
        t[i][0] = func(data[i]) + eps(data[i])

    return data, t


# (x, f(x), is points or curve? True\False)
def plot2d(info):
    fig, ax = matplotlib.pyplot.subplots()
    lim = [float('+inf'), float('-inf')]
    for x, f, curve in info:
        if curve:
            ax.plot(x, f)
        else:
            ax.plot(x, f, marker=".", linewidth=0)
        lim = [min(x.min(), lim[0]), max(x.max(), lim[1])]
    ax.set_facecolor('white')
    ax.set_xlim(lim)
    ax.set_ylim([-1, 1])
    ax.set_xlabel('Argument')
    ax.set_ylabel('Value')
    ax.set_title('Graph')
    fig.canvas.set_window_title('View')
    fig.set_facecolor('white')

    matplotlib.pyplot.show()


if __name__ == '__main__':
    s = time.time()

    e = time.time()
    print('time', e - s)




















