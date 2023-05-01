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
    def __init__(self, image_size=28):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, 16),
            nn.ReLU(),

            nn.Linear(16, 1),
            nn.Sigmoid(),


            # nn.Linear(256, 400),
            # nn.ReLU(),
            #
            # nn.Linear(400, 10),
            # nn.ReLU(),
            #
            # nn.Linear(10, 1),
            # nn.Sigmoid(),


            # nn.Linear(784, 800),
            # nn.ReLU(),
            #
            # nn.Linear(800, 10),
            # nn.ReLU(),
            #
            # nn.Linear(10, 1),
            # nn.Sigmoid(),
        )
        self.image_size = image_size

    def forward(self, x):
        x = x.view(x.size(0), self.image_size ** 2)
        output = self.model(x)
        return output


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


class DigitsDataset(Dataset):
    """Pytorch dataloader for the Optical Recognition of Handwritten Digits Data Set"""

    def __init__(self, csv_file, image_size=28, labels=None, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if labels is None:
            labels = [0]
        self.image_size = image_size
        self.csv_file = csv_file
        self.transform = transform
        self.df = self.filter_by_label(labels)

    def filter_by_label(self, label):
        # Use pandas to return a dataframe of only zeros
        df = pd.read_csv(self.csv_file)
        df = df.loc[df['label'].isin(label)]
        return df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.df.iloc[idx, :-1] / 16

        image = np.array(image)
        image = image.astype(np.float32).reshape(28, 28)

        if self.transform:
            image = self.transform(image)

        # Return image and label
        return image, 0


def start_train():
    wandb.login(relogin=True, key=os.getenv("WANDB_API_KEY", "606da00db4db699efabdef0dab836bbacb81e261"))

    seed = 42
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

    # Quantum variables
    n_qubits = 6  # Total number of qubits / N
    n_a_qubits = 2  # Number of ancillary qubits / N_A
    q_depth = 5  # Depth of the parameterised quantum circuit / D
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
    labels = [0]

    lrG = 0.3  # Learning rate for the generator
    lrD = 0.01
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


    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(image_size, torchvision.transforms.InterpolationMode.BILINEAR)])

    train_set_buff = torchvision.datasets.MNIST(root="./quantum_gans", train=True, download=True, transform=transform)
    train_set = list(filter(lambda i: i[1] in labels, train_set_buff))

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True)

    images = torchvision.utils.make_grid(list(map(lambda im: im[0], train_set[:16])))
    wandb.log({"Data":[wandb.Image(images)]})

    discriminator = Discriminator(image_size=image_size).to(device=device)
    generator = PatchQuantumGenerator(n_generators, q_depth, n_qubits, n_a_qubits, device, partial_measure).to(
        device=device)

    optimizer_discriminator = torch.optim.SGD(discriminator.parameters(), lr=lrD)
    optimizer_generator = torch.optim.SGD(generator.parameters(), lr=lrG)

    wandb.watch(discriminator, log_freq=10)
    wandb.watch(generator, log_freq=10)

    real_labels = torch.full((batch_size,), 1.0, dtype=torch.float, device=device)
    generated_labels = torch.full((batch_size,), 0.0, dtype=torch.float, device=device)

    fixed_noise = torch.rand(show_images_count, n_qubits, device=device) * math.pi / 2

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
        for n, (real_samples, mnist_labels) in enumerate(train_loader):

            size = min(batch_size, real_samples.shape[0])

            # Данные для тренировки дискриминатора
            start = timer()

            real_samples = real_samples.reshape(-1, image_size * image_size).to(device=device)

            noise = torch.rand(size, n_qubits, device=device) * math.pi / 2
            generated_samples = generator(noise)

            end = timer()
            fake_data_time = end - start

            # Обучение дискриминатора
            start = timer()

            discriminator.zero_grad()
            outD_real = discriminator(real_samples).view(-1)
            outD_fake = discriminator(generated_samples.detach()).view(-1)

            errD_real = loss_function(outD_real, real_labels)
            errD_fake = loss_function(outD_fake, generated_labels)

            errD_real.backward()
            errD_fake.backward()

            loss_discriminator = errD_real + errD_fake
            optimizer_discriminator.step()

            end = timer()
            disc_train_time = end - start

            # Обучение генератора
            start = timer()

            generator.zero_grad()
            outD_fake = discriminator(generated_samples).view(-1)
            errG = loss_function(outD_fake, real_labels)
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
                generated_samples = generator(fixed_noise).view(show_images_count, 1, image_size, image_size).cpu().detach()

                images = wandb.Image(
                    generated_samples,
                    caption=f'Epoch {epoch}. Batch {n + 1}'
                )

                wandb.log({"Generated": images})

    noise = torch.rand(16, n_qubits, device=device) * math.pi / 2
    generated_samples = generator(noise).view(16, 1, image_size, image_size).cpu().detach()

    images = wandb.Image(
        generated_samples,
        caption="Results"
    )

    wandb.log({"Results": images})

    wandb.finish()
