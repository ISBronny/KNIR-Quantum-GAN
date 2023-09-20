import itertools
import math
import os
import random
from timeit import default_timer as timer

import cv2 as cv
import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pennylane as qml
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn import datasets

if __name__ == '__main__':

    n_qubits = 5
    q_depth = 2
    n_a_qubits = 1

    seed = 116
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed = seed

    dev = qml.device("lightning.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
    def quantum_circuit(noise, weights):
        weights = weights.reshape(1 + q_depth, n_qubits)

        # Initialise latent vectors
        for i in range(n_qubits):
            qml.RY(noise[i], wires=i)


        for i in range(n_qubits - 1):
            qml.IsingYY(weights[0][i], wires=[i, i+1])

        for layer in range(q_depth):
            for i in range(n_qubits - 1):
                qml.CRY(weights[layer + 1][i], wires=[i, i+1])

            qml.CRY(weights[layer][n_qubits-1], wires=[n_qubits-1, 0])

        return qml.probs(wires=list(range(0, n_qubits)))

    noise = torch.rand(n_qubits) * math.pi
    weights = torch.rand((1 + q_depth) * n_qubits)
    fig, ax = qml.draw_mpl(quantum_circuit)(noise,weights)
    fig.show()

    print("Result: ", quantum_circuit(noise, weights))

    probs = quantum_circuit(noise, weights)
    probsgiven0 = probs[: (2 ** (n_qubits - n_a_qubits))]
    probsgiven0 /= np.sum(list(probs))

    # Post-Processing
    probsgiven = probsgiven0 / torch.max(probsgiven0)

    print("Post-process: ", probsgiven)
