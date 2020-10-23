import numpy as np
import random
import matplotlib.pyplot
import time
import math


def regression(data, t, phi, M, param):
    N = data.shape[0]
    PHI = np.zeros((N, M), dtype=np.float64)
    for i in range(N):
        PHI[i] = np.array([phi(j, data[i]) for j in range(M)], dtype=np.float64)
    w = np.linalg.inv((PHI.transpose().dot(PHI) + param * np.identity(M, dtype=np.float64))).dot(PHI.transpose()).dot(t)

    def predict(x):
        phi_vec = np.array([phi(i, x) for i in range(M)])
        return w.dot(phi_vec)

    return predict


def split_data(data, t):
    N = data.shape[0]
    M = data.shape[1]
    K = t.shape[1]
    data_copy = np.zeros((N, M+K), dtype=np.float64)
    for i in range(N):
        data_copy[i][:M] = data[i]
        data_copy[i][M:] = t[i]

    np.random.shuffle(data_copy)
    new_t = data_copy[:, M:]
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
    data = np.zeros((size, dim), dtype=np.float64)
    t = np.zeros((size, 1), dtype=np.float64)

    for i in range(size):
        for j in range(dim):
            data[i][j] = gen_coord()
        t[i][0] = func(data[i]) + eps(data[i])

    return data, t


# (x, f(x), is points or curve? True\False)
def plot2d(info):
    fig, ax = matplotlib.pyplot.subplots()
    lim_x = [float('+inf'), float('-inf')]
    lim_f = [float('+inf'), float('-inf')]
    for x, f, curve in info:
        if curve:
            ax.plot(x, f)
        else:
            ax.plot(x, f, marker=".", linewidth=0)
        lim_x = [min(x.min(), lim_x[0]), max(x.max(), lim_x[1])]
        lim_f = [min(f.min(), lim_f[0]), max(f.max(), lim_f[1])]
    ax.set_facecolor('white')
    ax.set_xlim(lim_x)
    ax.set_ylim(lim_f)
    ax.set_xlabel('Argument')
    ax.set_ylabel('Value')
    ax.set_title('Graph')
    fig.canvas.set_window_title('View')
    fig.set_facecolor('white')

    matplotlib.pyplot.show()


def classification_error(model, test):
    data, t = test
    N = data.shape[0]
    K = t.shape[1]
    F = [0] * K
    C = [0] * K
    for i in range(N):
        result = model(data[i]).argmax()
        class_x = t[i].argmax()
        if result == class_x:
            F[result] += 1
        C[class_x] += 1
    error = 1.0
    for i in range(K):
        if C[i] > 0:
            error *= F[i]/C[i]
    return 1.0 - error


def logistic_regression(learn, valid, K, gamma, alpha, eps, limit):
    learn_data, learn_t = learn[0].copy(), learn[1].copy()
    learn_data = 2.0*(learn_data - learn_data.min())/(learn_data.max() - learn_data.min()) - 1.0
    N = learn_data.shape[0]
    M = learn_data.shape[1]
    batch_size = 64
    prev_W = np.zeros((K, M), dtype=np.float64)
    prev_b = np.zeros(K, dtype=np.float64)
    for i in range(K):
        for j in range(M):
            prev_W[i][j] = random.uniform(-1, 1)
        prev_b[i] = random.uniform(-1, 1)

    def get_model(W, b):
        def model(x):
            z = W.dot(x)+b
            z = np.exp(z)
            z /= sum(z)
            return z
        return model

    prev_error = float('+inf')
    number = 0
    while prev_error > eps and number < limit:
        prev_model = get_model(prev_W, prev_b)
        grad_by_W = np.zeros((K, M), dtype=np.float64)
        grad_by_b = np.zeros(K, dtype=np.float64)
        indexes = list(range(N))
        random.shuffle(indexes)
        for i in range(batch_size):
            idx = indexes[i]
            grad_by_W += (prev_model(learn_data[idx])-learn_t[idx]).reshape((K, 1)).dot(learn_data[idx].reshape((1, M)))+alpha*prev_W
            grad_by_b += prev_model(learn_data[idx])-learn_t[idx]
        grad_by_W /= batch_size
        grad_by_b /= batch_size
        W = prev_W - gamma * grad_by_W
        b = prev_b - gamma * grad_by_b

        model = get_model(W, b)
        prev_error = classification_error(model, valid)
        prev_W = W
        prev_b = b
        number += 1

    return get_model(prev_W, prev_b)


def gen_data_t_1(size):
    K = 4
    data = np.zeros((size, 2), dtype=np.float64)
    t = np.zeros((size, K), dtype=np.float64)

    centers = np.array([[5.0, 5.0], [3.0, -2.0], [-7.0, 1.0], [0.0, 0.0]])
    r = np.array([3.0, 2.0, 4.0, 0.5])
    for i in range(size):
        index = random.randint(0, K-1)
        data[i][0] = random.uniform(centers[index][0]-r[index], centers[index][0]+r[index])
        data[i][1] = random.uniform(centers[index][1] - math.sqrt(r[index]**2-(centers[index][0]-data[i][0])**2),
                                    centers[index][1] + math.sqrt(r[index]**2-(centers[index][0]-data[i][0])**2))
        t[i][index] = 1.0

    return data, t


def example_classification_k(get_model):
    data, t = gen_data_t_1(1000)
    train, valid, test = split_data(data, t)

    model = get_model(train, valid)
    print(classification_error(model, train), classification_error(model, valid), classification_error(model, test))

    T = test[0].shape[0]
    plot_data = [[], [], [], []]
    for i in range(T):
        plot_data[model(test[0][i]).argmax()].append(test[0][i])
    plot_arg = []
    for i in range(4):
        if len(plot_data[i]) > 0:
            plot_arg.append((np.array([x[0] for x in plot_data[i]]), np.array([x[1] for x in plot_data[i]]), False))

    plot2d(plot_arg)


def example_1():
    def f(train, valid):
        return logistic_regression(train, valid, 4, 3, 0.15, 1e-2, 100)
    example_classification_k(f)


def solution_tree(data, t, step_number, limit_level, limit_number, limit_entropy):
    K = t.shape[1]
    N = t.shape[0]
    M = data.shape[1]

    nodes = [[]]

    def H(index):
        result = 0.0
        cnt = np.zeros(K, dtype=np.float64)
        for i in range(N):
            v = 0
            while True:
                if v == index:
                    cnt[t[i].argmax()] += 1
                    break
                if len(nodes[v]) < 4:
                    break
                left, right, psi, tao = nodes[v]
                if psi(data[i]) < tao:
                    v = left
                else:
                    v = right
        s = cnt.sum()
        for i in range(K):
            if cnt[i] > 0:
                p = cnt[i] / s
                result += p*math.log2(p)
        return -result, s, cnt

    queue = [(0, 0)]
    while len(queue) > 0:
        v, level = queue.pop(0)
        H_v, N_v, cnt = H(v)
        if level > limit_level or N_v < limit_number or H_v < limit_entropy:
            nodes[v] = [cnt.argmax()]
            continue
        left, right = len(nodes), len(nodes) + 1
        nodes.append([])
        nodes.append([])
        nodes[v] = [left, right, None, None]
        max_value = float('-inf')
        argmax = [None, None]
        for i in range(M):
            psi = lambda x, idx=i: x[idx]
            nodes[v][2] = psi
            min_x, max_x = data[:, i].min(), data[:, i].max()
            tao = min_x
            step = (max_x-min_x)/step_number
            while tao < max_x:
                nodes[v][3] = tao
                tao += step

                H_left, N_left, _ = H(left)
                H_right, N_right, _ = H(right)

                I = H_v - (N_left/N_v)*H_left - (N_right/N_v)*H_right
                if max_value < I:
                    max_value = I
                    argmax[0] = psi
                    argmax[1] = nodes[v][3]

        nodes[v][2] = argmax[0]
        nodes[v][3] = argmax[1]

        queue.append((left, level+1))
        queue.append((right, level+1))

    def get_model(nodes):
        def model(x):
            v = 0
            class_x = None
            while True:
                if len(nodes[v]) < 4:
                    class_x = nodes[v][0]
                    break
                left, right, psi, tao = nodes[v]
                if psi(x) < tao:
                    v = left
                else:
                    v = right
            result = np.zeros(K, dtype=np.float64)
            result[class_x] = 1.0
            return result
        return model

    return get_model(nodes)


def example_2():
    def f(train, valid):
        return solution_tree(train[0], train[1], 25, 6, 5, 1e-3)
    example_classification_k(f)


if __name__ == '__main__':
    s = time.time()

    example_2()

    e = time.time()
    print('time', e - s)
