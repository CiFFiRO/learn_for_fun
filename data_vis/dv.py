import numpy as np
import matplotlib
import matplotlib.pyplot
import graphviz
import cv2
import os
import sklearn.manifold


def plot_ex_1():
    tree = graphviz.Graph()
    limit = 16
    queue = [0]
    tree.node('0', '0')
    number = 0
    while len(queue) > 0:
        v = queue.pop(0)
        if v+2 > limit:
            break
        l = number + 1
        r = number + 2
        tree.node(str(l), str(l))
        tree.node(str(r), str(r))
        tree.edge(str(v), str(l))
        tree.edge(str(v), str(r))
        queue.append(l)
        queue.append(r)
        number += 2
    tree.render('ex_1.gv', view=True)


def plot_ex_2():
    proc = graphviz.Digraph()
    proc.node('C', 'Создание')
    proc.node('Rd', 'Готов к выполнению')
    proc.node('R', 'Выполняется')
    proc.node('B', 'Блокирован')
    proc.node('Z', 'Зомби')
    proc.edge('C', 'Rd')
    proc.edge('Rd', 'R')
    proc.edge('R', 'B')
    proc.edge('B', 'Rd')
    proc.edge('R', 'Z')
    proc.render('ex_2.gv', view=True)


def clear_noise_gauss(image):
    return cv2.GaussianBlur(image, (3, 3), 3)


def SIFT(image):
    detector = cv2.SIFT_create()
    features, descriptors = detector.detectAndCompute(cv2.cvtColor(clear_noise_gauss(image), cv2.COLOR_RGB2GRAY), None)
    return [(round(x.pt[0]), round(x.pt[1]), x.size/2) for x in features], [x.angle for x in features], descriptors


def image_open(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def plot2d(data):
    fig, ax = matplotlib.pyplot.subplots()
    for x, y in data:
        ax.plot(x, y, marker=".", linewidth=0, color='black')
    ax.set_facecolor('white')
    ax.set_xlim([data[:, 0].min(), data[:, 0].max()])
    ax.set_ylim([data[:, 1].min(), data[:, 1].max()])
    ax.set_xlabel('Argument')
    ax.set_ylabel('Value')
    ax.set_title('Graph')
    fig.canvas.set_window_title('View')
    fig.set_facecolor('white')
    matplotlib.pyplot.show()


def plot_ex_3(show):
    image_files = os.listdir('./b52')
    descriptors = []
    for file_name in image_files:
        file_name = os.path.join('.', 'b52', file_name)
        image = image_open(file_name)
        _, _, desc = SIFT(image)
        if show == 'MDS':
            descriptors.extend(desc[:50])
        else:
            descriptors.extend(desc)

    # Y_TSNE = sklearn.manifold.TSNE(n_components=2).fit_transform(descriptors)
    # Y_LLE = sklearn.manifold.LocallyLinearEmbedding(n_components=2).fit_transform(descriptors)
    # Y_MDS = sklearn.manifold.MDS(n_components=2).fit_transform(descriptors)
    # Y_Isomap = sklearn.manifold.Isomap(n_components=2).fit_transform(descriptors)

    # fig, ax = matplotlib.pyplot.subplots(nrows=2, ncols=2)
    #
    # data = [Y_TSNE, Y_LLE, Y_MDS, Y_Isomap]
    # title = ['TSNE', 'LLE', 'MDS', 'Isomap']
    # for index in range(len(ax)):
    #     for i in range(4):
    #         ax[index].plot(data[index][:][0], data[index][:][1], color='black')
    #     ax[index].set_facecolor('white')
    #     ax[index].set_xlim([data[index][:, 0].min(), data[index][:, 0].max()])
    #     ax[index].set_ylim([data[index][:, 1].min(), data[index][:, 1].max()])
    #     ax[index].set_xlabel('Y_x')
    #     ax[index].set_ylabel('Y_y')
    #     ax[index].set_title(title[index])
    # fig.canvas.set_window_title('Dimensional decrease')
    # fig.set_facecolor('white')
    #
    # matplotlib.pyplot.show()

    if show == 'TSNE':
        Y_TSNE = sklearn.manifold.TSNE(n_components=2).fit_transform(descriptors)
        plot2d(Y_TSNE)
    elif show == 'LLE':
        Y_LLE = sklearn.manifold.LocallyLinearEmbedding(n_components=2).fit_transform(descriptors)
        plot2d(Y_LLE)
    elif show == 'MDS':
        Y_MDS = sklearn.manifold.MDS(n_components=2).fit_transform(descriptors)
        plot2d(Y_MDS)
    else:
        Y_Isomap = sklearn.manifold.Isomap(n_components=2).fit_transform(descriptors)
        plot2d(Y_Isomap)


if __name__ == '__main__':
    plot_ex_3('Isomap')


