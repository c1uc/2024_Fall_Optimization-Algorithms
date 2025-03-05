import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

sgd_lrs = [0.005, 0.0025, 0.001]
svrg_lr = 0.025

# logistic regression
setting = "logistic"
SGD_PATH = "./outputs/20241028-203438_SGD_MNIST_logistic"
SVRG_PATH = "./outputs/20241028-210002_SVRG_MNIST_logistic"
train_loss_optima = 0.21524

# # nn one layer
# setting = "one_layer_nn"
# SGD_PATH = "./outputs/20241028-200737_SGD_MNIST_nn_one_layer"
# SVRG_PATH = "./outputs/20241028-210002_SVRG_MNIST_nn_one_layer"
# train_loss_optima = 0.029126

svrg_values = np.load(SVRG_PATH + f"_lr{svrg_lr}/" + "train_stats.npz")
# optima_values = np.load(OPTIMA_PATH + "train_stats.npz")
# train_loss_optima = optima_values["train_loss"][-1]

def smooth_curve(values, factor=0.995):
    smoothed_values = []
    last = values[0]
    for value in values:
        smoothed_values.append(last * factor + value * (1 - factor))
        last = smoothed_values[-1]
    return smoothed_values

def plot_acc_and_loss(axs, vals, label):
    acc_x = vals["train_acc"][:, 0] / 60000
    acc_y = smooth_curve(vals["train_acc"][:, 1])
    axs[0].plot(acc_x, acc_y, label=label)

    loss_x = vals["train_loss"][:, 0] / 60000
    loss_y = smooth_curve(vals["train_loss"][:, 1] - train_loss_optima)
    axs[1].plot(loss_x, loss_y, label=label)


fig, ax = plt.subplots(1, 2, figsize=(12, 6))
for lr in sgd_lrs:
    sgd_values = np.load(SGD_PATH + f"_lr{lr}/" + "train_stats.npz")
    plot_acc_and_loss(ax, sgd_values, f"SGD: lr={lr}")

plot_acc_and_loss(ax, svrg_values, f"SVRG: lr={svrg_lr}")

ax[0].set_title("Training Accuracy")
ax[0].set_xlabel("# grad / # data")
ax[0].set_ylabel("Accuracy")
ax[0].set_xlim(0, 100)
ax[0].set_ylim(0, 1)
ax[0].legend()

# Training Loss Residual, y-axis: log scale
ax[1].set_title("Training Loss Residual")
ax[1].set_xlabel("# grad / # data")
ax[1].set_ylabel("Loss Residual")
ax[1].set_xlim(0, 100)
ax[1].legend()
plt.yscale("log")
# plt.tight_layout()

plt.suptitle("Training Stats: " + setting, fontsize=16)
plt.savefig("training_stats_" + setting + ".png")
plt.show()
