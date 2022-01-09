from numpy import random
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from wnnlib import KDTree


def test_data_encoding(tree, data):
    i = 0
    cdata = []
    for x in data:
        print('----------------------------------')
        print(i)
        print(x)
        c = tree.encode(point=x)
        print(c)
        cdata.append(c)
        i += 1

    i = 0
    acc_error2 = 0
    for c in cdata:
        print('----------------------------------')
        print(i)
        x = data[i, :]
        print(x)
        print(c)
        d = tree.decode(code=c)
        print(d)
        acc_error2 += np.transpose(x - d) * (x - d)
        i += 1
    print(np.sqrt(acc_error2 / i))

    # Plot four different levels of the KD tree
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.set_aspect(aspect=1)
    ax1.scatter(data[:, 0], data[:, 1], s=5, marker=".")

    patches, colors = tree.draw_rectangle()
    collection = PatchCollection(patches, cmap=plt.cm.get_cmap('gray'), alpha=0.75)
    collection.set_array(np.asarray(colors))
    collection.set_edgecolor('k')
    ax2.add_collection(collection)
    cbar = plt.colorbar(collection)
    cbar.set_label('depth', rotation=90)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    # ax2.set_aspect(aspect=1)

    # circle = plt.Circle(x, radius=1, ec='k', fc='g')
    # ax.add_patch(circle)
    #
    # patches = tree.draw_sector()
    # colors = 255 * np.random.rand(len(patches))
    # collection = PatchCollection(patches, ec='k', fc='k', lw=0.1, cmap=plt.cm.get_cmap('prism'))
    # collection.set_array(np.array(colors))
    # ax.add_collection(collection)
    # ax.scatter(data[:, 1]*np.cos(np.deg2rad(data[:, 0])), data[:, 1]*np.sin(np.deg2rad(data[:, 0])), s=9)

    # ax.set_xlim(0, 20)
    # ax.set_ylim(-60, 60)

    plt.show()


if __name__ == '__main__':
    """Example usage"""

    # tree = KDTree(max_depth=16, learning_rate=0.1, min_splitting_volume=0.00001, min_bounds=[0, -60], max_bounds=[20, 60])
    tree = KDTree(max_depth=16, learning_rate=0.001, min_splitting_volume=0.01, min_bounds=[0, 0],
                  max_bounds=[10, 10])

    # Create a set of structured random points in two dimensions
    np.random.seed(0)

    # data = np.random.multivariate_normal(mean=[10, 0], cov=[[5, 0], [0, 30]], size=10)
    data = np.random.multivariate_normal(mean=[5, 5], cov=[[1, 0.5], [0.5, 1]], size=1000)

    test_data_encoding(tree, data)

    data = np.random.uniform(low=0, high=10, size=[100, 2])

    test_data_encoding(tree, data)



