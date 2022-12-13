import time
from math import pi
import pyLasaDataset as lasa
import matplotlib.pyplot as plt
from dynamic_obstacle_avoidance.visualization import plot_obstacles


from dynamic_obstacle_avoidance.containers import ObstacleContainer


from vartools.dynamical_systems import LinearSystem


def reference_trajectory():
    path_index = 0
    path = lasa.DataSet.LShape.demos[path_index].pos
    attractor_pos = path[:, -1]
    x_lim = [-35.0, 5.0]
    y_lim = [-5.0, 45.0]
    fig, ax = plt.subplots(figsize=(10, 8))
    obstacle_environment = ObstacleContainer()
    plot_obstacles(
        ax=ax,
        obstacle_container=obstacle_environment,
        x_lim=x_lim,
        y_lim=y_lim,
        showLabel=False,
    )
    ax.plot(path[0, :], path[1, :], color='black', linewidth='3.5')
    ax.plot(path[0, 0], path[1, 0], color='black',
            marker="d", markersize=10.0, label="Initial State")
    ax.plot(path[0, -1], path[1, -1], color='black',
            marker="*", markersize=14.0, label="Attractor")
    ax.grid()
    ax.legend()
    ax.set_title("L Shape")
    plt.show()

    input("Enter key to continue...")


if (__name__) == "__main__":
    plt.close("all")
    plt.ion()
    reference_trajectory()
