"""
@ File Name     :   drawtoolbox.py
@ Time          :   2022/10/17
@ Author        :   Cheng Kaiyue
@ Version       :   1.0
@ Contact       :   chengky18@icloud.com
@ Description   :   None
@ Function List :   func1() -- func desc1
@ Class List    :   Class1 -- class1 desc1
@ Details       :   None
"""

from matplotlib import pyplot as plt
import numpy as np


def draw_3d(draw_data, is_norm, target_path):

    train_data, train_label, valid_data, valid_label, best_model = (
        draw_data[0],
        draw_data[1],
        draw_data[2],
        draw_data[3],
        draw_data[4],
    )

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    x1_train = train_data[:, 1]
    x3_train = train_data[:, 2]
    z_train = train_label[:, 0]
    ax.scatter3D(x1_train, x3_train, z_train, cmap="Blue")

    x1_valid = valid_data[:, 1]
    x3_valid = valid_data[:, 2]
    z_valid = valid_label[:, 0]
    ax.scatter3D(x1_valid, x3_valid, z_valid, cmap="Red")

    if is_norm:
        xx1_model = np.arange(0, 1, 0.01)
        xx3_model = np.arange(0, 1, 0.01)
    else:
        xx1_model = np.arange(np.min(x1_train) * 0.9, np.max(x1_train) * 1.1, 0.01)
        xx3_model = np.arange(np.min(x3_train) * 0.9, np.max(x3_train) * 1.1, 0.01)
    x1_model, x3_model = np.meshgrid(xx1_model, xx3_model)
    model_data = np.hstack(
        (np.array([x1_model.flatten()]).T, np.array([x3_model.flatten()]).T)
    )
    model_data = np.insert(model_data, 0, values=1, axis=1)
    z_model = best_model.forward(model_data)
    z_model = z_model.reshape(x1_model.shape)
    ax.plot_surface(x1_model, x3_model, z_model, cmap="rainbow", alpha=0.8)

    plt.title("Data and Boundary")
    plt.savefig(target_path, dpi=800)
    plt.show()


def draw_loss(x_list, loss_list, args_list, save_path, title):
    for i in range(len(loss_list)):
        plt.plot(x_list[i], loss_list[i])

    plt.legend((args_list), loc="upper right")
    plt.xlabel("iter")
    plt.ylabel("loss")
    plt.title(title)
    plt.savefig(save_path)
    plt.show()


def draw_loss_single(x_list, loss_list, save_path):
    plt.plot(x_list, loss_list)
    plt.xlabel("iter")
    plt.ylabel("loss")
    plt.title("Loss Curve")
    plt.savefig(save_path)
    plt.show()
