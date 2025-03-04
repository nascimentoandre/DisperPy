import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

def plot_3D(X, labels=[]):
    if len(labels) == 0:
        labels = X[:, 2]
    ax = plt.axes(projection="3d")
    ax.scatter3D(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap="jet")
    plt.show()

def plot(X, T, U, T0, U0):
    x = X[:, 0]
    y = X[:, 1]
    z = X[:, 2]

    peaks, _ = find_peaks(z)
    xmax = x[peaks]
    ymax = y[peaks]

    x = np.unique(x)[::-1]
    y = np.unique(y)[::-1]
    xmesh, ymesh = np.meshgrid(x, y)
    zmesh = z.reshape(len(x), len(y)).T

    plt.xscale("log")
    plt.ylim([2.0, 5.0])
    plt.contourf(xmesh, ymesh, zmesh, levels=40, cmap="jet")
    plt.scatter(xmax, ymax, c='pink', s=6)
    plt.scatter(T0, U0, c='r')
    plt.scatter(T, U, c='k')
    plt.xlabel("Period (s)")
    plt.ylabel("Group velocity (km/s)")
    plt.show()

def plot_one_curve(X, T, U):
    x = X[:, 0]
    y = X[:, 1]
    z = X[:, 2]

    peaks, _ = find_peaks(z)
    xmax = x[peaks]
    ymax = y[peaks]

    x = np.unique(x)[::-1]
    y = np.unique(y)[::-1]
    print(x.shape, y.shape)
    xmesh, ymesh = np.meshgrid(x, y)
    zmesh = z.reshape(len(x), len(y)).T

    plt.xscale("log")
    plt.ylim([2.0, 5.0])
    plt.contourf(xmesh, ymesh, zmesh, levels=40, cmap="jet")
    plt.scatter(xmax, ymax, c='pink', s=6)
    plt.scatter(T, U, c='k')
    plt.xlabel("Period (s)")
    plt.ylabel("Group velocity (km/s)")
    plt.show()

def save_figure(X, T0, U0, T, U, fname):
    x = X[:, 0]
    y = X[:, 1]
    z = X[:, 2]

    peaks, _ = find_peaks(z)
    xmax = x[peaks]
    ymax = y[peaks]

    x = np.unique(x)[::-1]
    y = np.unique(y)[::-1]
    xmesh, ymesh = np.meshgrid(x, y)
    try:
        zmesh = z.reshape(len(x), len(y)).T

        plt.xscale("log")
        plt.ylim([2.0, 5.0])
        plt.contourf(xmesh, ymesh, zmesh, levels=40, cmap="jet")
        plt.plot(T0, U0, '--', c='r', label="raw", linewidth=3)
        plt.plot(T, U, c='k', label="clean", linewidth=3)

        plt.xlabel("Period (s)")
        plt.ylabel("Group velocity (km/s)")
        plt.legend()

        plt.savefig("%s"%fname)
        plt.clf()
        plt.close()
    except:
        print("An issue has occurred while plotting %s, skipping."%fname)
        return
