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
# Load and prepare data
import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

import LPUFAuthnetDefinition as  ZadiDefinition


# Load data
LinkOfDataset = 'Datasets/CRP_FPGA_01 - Copy.csv'
data = pd.read_csv(LinkOfDataset, encoding='utf-8')




def average_flipped_bits_and_accuracy(X, Y):
    if X.shape != Y.shape:
        raise ValueError("Both tensors must have the same shape.")
    
    # Convert boolean tensor to float
    flipped_bits = (X != Y).float().sum(dim=1)
    threshold = 1  # Adjust this threshold as needed
    
    total_flipped_bits = flipped_bits.sum().item()
    num_pairs = X.shape[0]
    average_flipped = total_flipped_bits / num_pairs
    
    accuracy = (flipped_bits < threshold).float().mean().item() * 100
    total_flipped_bits_ten_perc_cent = (flipped_bits >= threshold).float().mean().item()
    
    return average_flipped, total_flipped_bits_ten_perc_cent, accuracy




def AccurcyOfFirstTraining():
    
    ChallengeLen = 32
    ResponseLen = 32
    TrainData= data
    
    XTrainData = TrainData.iloc[:, :ChallengeLen]
    YTrainData = TrainData.iloc[:, ChallengeLen:ChallengeLen]
    

    encoder1 = ZadiDefinition.Encoder1()
    encoder2 = ZadiDefinition.Encoder2()
    decoder2 = ZadiDefinition.Decoder2 ()
    
    
    checkpoint = torch.load('Trained models/best_model.pth')

    encoder1.load_state_dict(checkpoint['encoder1_state_dict'])
    encoder2.load_state_dict(checkpoint['encoder2_state_dict'])
    decoder2.load_state_dict(checkpoint['decoder2_state_dict'])
        
    encoder1.eval()
    encoder2.eval()
    decoder2.eval()

    
    XTrainData_tensor = torch.FloatTensor(XTrainData.values)
    
    with torch.no_grad():
        encoded_output = encoder1(XTrainData_tensor)
        encoded_output = encoder2(encoded_output)
        encoded_output = decoder2(encoded_output)

    encoded_output = encoded_output.squeeze().round()
    
    avg_flipped, flipped_ten_perc, acc = average_flipped_bits_and_accuracy(encoded_output, XTrainData_tensor)

    print(f"Average flipped bits: {avg_flipped:.4f}")
    print(f"Flipped bits > threshold: {flipped_ten_perc:.4f}")
    print(f"Accuracy: {acc:.2f}%")





def AccuracyOfEnhancedTraining():
    ChallengeLen = 32
    ResponseLen = 16  # Assuming the response length is 16
    TrainData = data
    
    XTrainData = TrainData.iloc[:, :ChallengeLen]
    YTrainData = TrainData.iloc[:, ChallengeLen:ChallengeLen+ResponseLen]
    
    

    # Load original encoders
    original_encoder1 = ZadiDefinition.Encoder1()
    original_encoder2 = ZadiDefinition.Encoder2()
    
    #original_encoder1.load_state_dict(torch.load("Encoder1.pth"))
    #original_encoder2.load_state_dict(torch.load("Encoder2.pth"))
    
    
       
    checkpoint = torch.load('Trained models/best_model.pth')
    original_encoder1.load_state_dict(checkpoint['encoder1_state_dict'])
    original_encoder2.load_state_dict(checkpoint['encoder2_state_dict'])

    # Create enhanced encoders
    enhanced_encoder1 = ZadiDefinition.FrozenEncoder1(original_encoder1)
    enhanced_encoder2 = ZadiDefinition.FreezedEncoder2(original_encoder2)
    decoder1 = ZadiDefinition.Decoder1()
    
    # Load the trained weights for enhanced encoders and decoder
    checkpoint2 = torch.load('Trained models/best_model2.pth')

    enhanced_encoder2.load_state_dict(checkpoint2['encoder2_state_dict'])
    decoder1.load_state_dict(checkpoint2['decoder2_state_dict'])
    
    enhanced_encoder1.eval()
    enhanced_encoder2.eval()
    decoder1.eval()
    
    XTrainData_tensor = torch.FloatTensor(XTrainData.values)
    YTrainData_tensor = torch.FloatTensor(YTrainData.values)
    
    with torch.no_grad():
        encoded_output = enhanced_encoder1(XTrainData_tensor)
        encoded_output = enhanced_encoder2(encoded_output)
        encoded_output = decoder1(encoded_output)
    
    encoded_output = encoded_output.round()
    avg_flipped, flipped_ten_perc, acc = average_flipped_bits_and_accuracy(encoded_output, YTrainData_tensor)
    print(f"Average flipped bits: {avg_flipped:.4f}")
    print(f"Flipped bits > threshold: {flipped_ten_perc:.4f}")
    print(f"Accuracy: {acc:.2f}%")



def main():
    print("Running First Training Accuracy Calculation...")
    AccurcyOfFirstTraining()

    print("Running Enhanced Training Accuracy Calculation...")
    AccuracyOfEnhancedTraining()


if __name__ == "__main__":
    main()
