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


class Discriminator(nn.Module):
    def __init__(self, image_size, labels_count):
        super().__init__()
        self.label_emb = nn.Embedding(labels_count, labels_count)
        if image_size == 8:
            self.model = nn.Sequential(
                nn.Linear(64 + labels_count, 64),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3),

                nn.Linear(64, 16),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3),

                nn.Linear(16, 1),
                nn.Sigmoid())

        if image_size == 16:
            self.model = nn.Sequential(
                nn.Linear(256 + labels_count, 400),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3),

                nn.Linear(400, 10),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3),

                nn.Linear(10, 1),
                nn.Sigmoid())

        if image_size == 28:
            self.model = nn.Sequential(
                nn.Linear(784 + labels_count, 800),
                nn.ReLU(),

                nn.Linear(800, 10),
                nn.ReLU(),

                nn.Linear(10, 1),
                nn.Sigmoid())

        self.image_size = image_size
        self.labels_count = labels_count

    def forward(self, x, labels):
        # Reshape fake image
        x = x.view(x.size(0), self.image_size ** 2)

        # One-hot vector to embedding vector
        c = self.label_emb(torch.LongTensor(labels))

        # Concat image & label
        x = torch.cat([x, c], 1)

        # Discriminator out
        out = self.model(x)

        return out


class PatchQuantumGenerator(nn.Module):
    """Quantum generator class for the patch method"""

    def __init__(self, n_generators, q_depth, n_qubits, n_a_qubits, device, partial_measure, q_delta=1):
        """
        Args:
            n_generators (int): Number of sub-generators to be used in the patch method.
            q_delta (float, optional): Spread of the random distribution for parameter initialisation.
        """

        super().__init__()

        self.q_params = nn.ParameterList(
            [
                nn.Parameter(q_delta * torch.rand(q_depth * n_qubits), requires_grad=True)
                for _ in range(n_generators)
            ]
        )
        self.n_generators = n_generators
        self.q_depth = q_depth
        self.n_qubits = n_qubits
        self.n_a_qubits = n_a_qubits
        self.device = device
        self.partial_measure = partial_measure

    def forward(self, x):
        # Size of each sub-generator output
        patch_size = 2 ** (self.n_qubits - self.n_a_qubits)

        # Create a Tensor to 'catch' a batch of images from the for loop. x.size(0) is the batch size.
        images = torch.Tensor(x.size(0), 0).to(self.device)

        # Iterate over all sub-generators
        for params in self.q_params:

            # Create a Tensor to 'catch' a batch of the patches from a single sub-generator
            patches = torch.Tensor(0, patch_size).to(self.device)
            for elem in x:
                q_out = self.partial_measure(elem, params).float().unsqueeze(0)
                patches = torch.cat((patches, q_out))

            # Each batch of patches is concatenated with each other to create a batch of images
            images = torch.cat((images, patches), 1)

        return images


def start_train():
    wandb.login(relogin=True, key=os.getenv("WANDB_API_KEY", "606da00db4db699efabdef0dab836bbacb81e261"))

    seed = 141
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    image_size = 8

    use_gpu = False

    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda")
        print('Using CUDA')
    else:
        device = torch.device("cpu")
        print('Using CPU')

    labels = [0, 1, 5]

    # Quantum variables
    n_label_qubits = len(labels)
    n_a_qubits = 2  # Number of ancillary qubits / N_A
    n_qubits = 4 + n_a_qubits  # Total number of qubits / N
    q_depth = 7  # Depth of the parameterised quantum circuit / D
    n_generators = 4  # Number of subgenerators for the patch method / N_G
    dev = qml.device("lightning.qubit", wires=n_qubits)

    def partial_measure(noise, weights):
        # Non-linear Transform
        probs = quantum_circuit(noise, weights)
        probsgiven0 = probs[: (2 ** (n_qubits - n_a_qubits))]
        probsgiven0 /= torch.sum(probs)

        # Post-Processing
        probsgiven = probsgiven0 / torch.max(probsgiven0)
        return probsgiven

    @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
    def quantum_circuit(noise, weights):
        weights = weights.reshape(q_depth, n_qubits)

        # Initialise latent vectors
        for i in range(n_qubits):
            qml.RY(noise[i], wires=i)

        # Repeated layer
        for i in range(q_depth):
            # Parameterised layer
            for y in range(n_qubits):
                qml.RY(weights[i][y], wires=y)

            # Control Z gates
            for y in range(n_qubits - 1):
                qml.CZ(wires=[y, y + 1])

        return qml.probs(wires=list(range(n_qubits)))

    batch_size = 1
    show_images_count = 16

    lrG = 0.01  # Learning rate for the generator
    lrD = 0.003
    num_epochs = 50
    loss_function = nn.BCELoss()

    # WANDB
    wandb.init(
        # set the wandb project where this run will be logged
        project="KNIR-Quantum-GAN",

        config={
            "image_size": image_size,
            "learning_rate_discr": lrD,
            "learning_rate_gen": lrG,
            "batch_size": batch_size,
            "epochs": num_epochs,
            "labels": labels,
            "quantum_variables": {
                "n_qubits": n_qubits,  # Total number of qubits / N
                "n_a_qubits": n_a_qubits,  # Number of ancillary qubits / N_A
                "q_depth": q_depth,  # Depth of the parameterised quantum circuit / D
                "n_generators": n_generators
            },
            "device": device
        },

        tags=[f'BATCH{batch_size}', f'{device}']
    )

    def get_label_vector(batch_size, batch_labels):
        vector = torch.full((batch_size, len(labels)), 0.01, dtype=torch.float)
        for n, label in enumerate(batch_labels):
            # может быть 0.99999?
            vector[n, labels.index(label)] = 0.99
        return vector

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize(image_size, torchvision.transforms.InterpolationMode.BILINEAR)])

    train_set_buff = torchvision.datasets.MNIST(root="./quantum_gans", train=True, download=True, transform=transform)
    train_set = list(filter(lambda i: i[1] in labels, train_set_buff))

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True)

    images = torchvision.utils.make_grid(list(map(lambda im: im[0], train_set[:16])))
    wandb.log({"Data": [wandb.Image(images)]})

    discriminator = Discriminator(image_size=image_size, labels_count=len(labels)).to(device=device)
    generator = PatchQuantumGenerator(n_generators, q_depth, n_qubits, n_a_qubits, device, partial_measure).to(device=device)

    optimizer_discriminator = torch.optim.SGD(discriminator.parameters(), lr=lrD)
    optimizer_generator = torch.optim.SGD(generator.parameters(), lr=lrG)

    wandb.watch(discriminator, log_freq=10)
    wandb.watch(generator, log_freq=10)

    one_labels = torch.full((batch_size,), 1.0, dtype=torch.float, device=device)
    zeros_labels = torch.full((batch_size,), 0.0, dtype=torch.float, device=device)

    fixed_noise = torch.rand(show_images_count, n_qubits - n_label_qubits, device=device)
    fixed_fake_labels = [random.choice(labels) for _ in range(show_images_count)]
    fixed_labels_and_noise = torch.cat((fixed_noise, get_label_vector(show_images_count, fixed_fake_labels)), 1) * math.pi / 2
    # fixed_labels_and_noise = torch.rand(show_images_count, n_qubits, device=device) * math.pi / 2

    METRICS_GEN_LOSS = "Generator Loss"
    wandb.define_metric(METRICS_GEN_LOSS)

    METRICS_GEN_TRAIN_TIME = "Generator Train Time"
    wandb.define_metric(METRICS_GEN_TRAIN_TIME, summary='mean')

    METRICS_DISC_LOSS = "Discriminator Loss"
    wandb.define_metric(METRICS_DISC_LOSS)

    METRICS_DISC_TRAIN_TIME = "Discriminator Train Time"
    wandb.define_metric(METRICS_DISC_TRAIN_TIME, summary='mean')

    METRICS_FAKE_DATA_GENERATION_TIME = "Fake Data Generation Time"
    wandb.define_metric(METRICS_FAKE_DATA_GENERATION_TIME, summary='mean')

    counter = 0

    for epoch in range(num_epochs):
        for n, (real_samples, real_labels) in enumerate(train_loader):

            size = min(batch_size, real_samples.shape[0])

            # Данные для тренировки дискриминатора
            start = timer()

            real_samples = real_samples.reshape(-1, image_size * image_size).to(device=device)

            noise = torch.rand(size, n_qubits - n_label_qubits, device=device)
            fake_labels = [random.choice(labels) for _ in range(size)]
            labels_and_noise = torch.cat((noise, get_label_vector(size, fake_labels)), 1) * math.pi / 2
            # labels_and_noise = torch.rand(size, n_qubits, device=device) * math.pi / 2
            generated_samples = generator(labels_and_noise)

            end = timer()
            fake_data_time = end - start

            # Обучение дискриминатора
            start = timer()

            discriminator.zero_grad()
            outD_real = discriminator(real_samples, [labels.index(label) for label in real_labels]).view(-1)
            outD_fake = discriminator(generated_samples.detach(),  [labels.index(label) for label in fake_labels]).view(-1)

            errD_real = loss_function(outD_real, one_labels)
            errD_fake = loss_function(outD_fake, zeros_labels)

            errD_real.backward()
            errD_fake.backward()

            loss_discriminator = errD_real + errD_fake
            optimizer_discriminator.step()

            end = timer()
            disc_train_time = end - start

            # Обучение генератора
            start = timer()

            generator.zero_grad()
            outD_fake = discriminator(generated_samples,  [labels.index(label) for label in fake_labels]).view(-1)
            errG = loss_function(outD_fake, one_labels)
            errG.backward()
            optimizer_generator.step()

            loss_generator = errG

            end = timer()
            gen_train_time = end - start

            wandb.log({
                f'{METRICS_GEN_LOSS}': loss_generator,
                f'{METRICS_DISC_LOSS}': loss_discriminator,

                f'{METRICS_GEN_TRAIN_TIME}': gen_train_time,
                f'{METRICS_DISC_TRAIN_TIME}': disc_train_time,
                f'{METRICS_FAKE_DATA_GENERATION_TIME}': fake_data_time,
            })

            counter += 1

            if counter % 20 == 0:
                # Show generated images
                generated_samples = generator(fixed_labels_and_noise).view(show_images_count, 1, image_size,image_size).cpu().detach()

                images = wandb.Image(
                    generated_samples,
                    caption=f'Epoch {epoch}. Batch {n + 1}.\n{fixed_fake_labels}'
                )

                # wandb.JoinedTable(columns=fixed_fake_labels, data=[[wandb.Image(im)] for im in torch.split(generated_samples, 1)])

                wandb.log({"Generated": images})

                torch.save(discriminator.state_dict(), f"./saves/{wandb.run.name}_disc_{epoch}_{n}.pt")
                torch.save(generator.state_dict(), f"./saves/{wandb.run.name}_gen_{epoch}_{n}.pt")

    noise = torch.rand(16, n_qubits, device=device) * math.pi / 2
    generated_samples = generator(noise).view(16, 1, image_size, image_size).cpu().detach()

    images = wandb.Image(
        generated_samples,
        caption="Results"
    )

    wandb.log({"Results": images})

    wandb.finish()
