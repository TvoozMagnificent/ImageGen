
T = 60
E = 5

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from random import random

def train(datasets):
    # Trains a model on the given datasets
    # The model is a simple feedforward neural network with 2 hidden layers, 10, 20, 20, 1

    model = nn.Sequential(
        nn.Linear(T, 30),
        nn.ReLU(),
        nn.Linear(30, 10),
        nn.ReLU(),
        nn.Linear(10, 1)
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_loader = DataLoader(datasets['train'], batch_size=32, shuffle=True)
    model.train()

    for epoch in range(1000):
        val_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            val_loss += loss.item()
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}, val_loss: {val_loss:.15f}')

        # if val_loss < 0.00001:
        #     break

    model.eval()
    return model

def randomize(sequence):
    return [_ + random() * E * 2 - E for _ in sequence]

def generate_data(sequence):
    # Generate data from sequence. Input is the some consecutive 10 elements, output is the next element.
    X, y = [], []
    for i in range(len(sequence) - T):
        for _ in range(10):
            X.append(randomize(sequence[i:i+T]))
            y.append([sequence[i+T]])
    return np.array(X), np.array(y)

def clamp(x):
    return x + random() * E * 2 - E
    # return max(0, min(round(x), 255))
    # return 0 if x < 128 else 255
    # return max(0, min(round(x), N-1)) + random() * E * 2 - E
    # return max(0, min(x, 10000)) + random()

def extend_sequence(sequence, n):
    # Extends the sequence by n elements
    X, y = generate_data(sequence)
    model = train({'train': TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))})
    for _ in range(n):
        X_input = torch.tensor(sequence[-T:], dtype=torch.float32).unsqueeze(0)  # Shape (1,10)
        y_pred = model(X_input).detach().item()
        sequence.append(clamp(y_pred))
    return sequence

import imageio as iio

# read an image
img = iio.imread("new15.png")
sequence = img[:, :].flatten().tolist()
extended = extend_sequence(sequence[:21*100], 21*100)
iio.imwrite("gen46.png", np.array(extended).reshape(200, 21))
