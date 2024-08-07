# -*- coding: utf-8 -*-
"""
@author: Brahim Mefgouda
@email: brahim.mefgouda@ku.ac.ae 
        brahim.mefgouda@ieee.org
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class Encoder1(nn.Module):
    def __init__(self):
        super(Encoder1, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(32, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 4)
        )

    def forward(self, x):
        return self.layers(x)


class Encoder2(nn.Module):
    def __init__(self):
        super(Encoder2, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 256)
        )

    def forward(self, x):
        return self.layers(x)

    def average_flipped_bits_and_accuracy(self, X, Y):
        if len(X) != len(Y):
            raise ValueError("Both datasets must have the same number of vectors.")
        
        total_flipped_bits = 0
        total_flipped_bits_ten_perc_cent = 0
        threshold = 1.6  # 10% of 16
        accuracy = 0
        num_pairs = len(X)
        
        for x_vec, y_vec in zip(X, Y):
            if len(x_vec) != len(y_vec):
                raise ValueError("Vectors within each pair must have the same length.")
            
            flipped_bits = sum(x != y for x, y in zip(x_vec, y_vec))
            total_flipped_bits += flipped_bits
            
            if flipped_bits >= threshold:
                total_flipped_bits_ten_perc_cent += 1
            else:
                accuracy += 1
        
        average_flipped = total_flipped_bits / num_pairs
        accuracy = (accuracy / num_pairs) * 100
        total_flipped_bits_ten_perc_cent /= num_pairs
        
        return average_flipped, total_flipped_bits_ten_perc_cent, accuracy


class Decoder1(nn.Module):
    def __init__(self):
        super(Decoder1, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, 8), nn.ReLU(),
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 1024), nn.ReLU(),
            nn.Linear(1024, 16)  # Assuming output size is 16
        )

    def forward(self, x):
        return self.layers(x)


class Decoder2(nn.Module):
    def __init__(self):
        super(Decoder2, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 1024), nn.ReLU(),
            nn.Linear(1024, 32)
        )

    def forward(self, x):
        return self.layers(x)


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


class FrozenEncoder1(nn.Module):
    def __init__(self, pretrained_encoder):
        super(FrozenEncoder1, self).__init__()
        self.encoder = pretrained_encoder
        freeze_model(self.encoder)

    def forward(self, x):
        return self.encoder(x)


class FreezedEncoder2(nn.Module):
    def __init__(self, Encoder2):
        super(FreezedEncoder2, self).__init__()
        self.Encoder2 = Encoder2
        freeze_model(self.Encoder2)
        
        self.new_layers = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 4), nn.ReLU()
        )

    def forward(self, x):
        with torch.no_grad():
            x = self.Encoder2(x)
        return self.new_layers(x)

# Note: Attack models would go here if defined.
