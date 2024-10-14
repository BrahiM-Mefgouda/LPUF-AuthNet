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
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random
import LPUFAuthnetDefinition as   LPUFAuthnetDefinition 
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(tp, fp, fn, tn, filename="MatrixOfConfusion.pdf"):
    """
    Plot a confusion matrix and save it as a PDF file.
    
    :param tp: True Positives
    :param fp: False Positives
    :param fn: False Negatives
    :param tn: True Negatives
    :param filename: Name of the output file (default: "MatrixOfConfusion.pdf")
    """
    cm = np.array([[tp, fp], [fn, tn]])
    
    plt.figure(figsize=(8, 6))
    
    # Determine the appropriate format based on the data type
    fmt = '.0f' if cm.dtype == int else '.2f'
    
    ax = sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', annot_kws={"size": 30})
    
    # Set labels and ticks
    labels = ['Correct LC', 'Fake LC']
    plt.xticks([0.5, 1.5], labels, fontsize=24, fontweight='bold')
    plt.yticks([0.5, 1.5], labels, fontsize=24, fontweight='bold')
    plt.ylabel('Actual', fontsize=24, fontweight='bold')
    plt.xlabel('Predicted', fontsize=24, fontweight='bold')
    
    # Add black border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_edgecolor('black')
    
    # Calculate and display accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    plt.title(f'Accuracy: {accuracy:.2%}', fontsize=24, fontweight='bold')
    
    # Save and close
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.close()  # Close the figure to free up memory

# Load data
LinkOfDataset = 'CRP_FPGA_01 - Copy.csv'
data = pd.read_csv(LinkOfDataset, encoding='utf-8')


def modify_dataset(data, num_bits_to_change=12):
    """
    Modify the dataset by flipping a specified number of bits in each row,
    while preserving the original DataFrame structure.
    
    :param data: pandas DataFrame containing the dataset
    :param num_bits_to_change: number of bits to flip in each row (default: 12)
    :return: modified pandas DataFrame
    """
    def flip_bits(row, num_bits):
        # Convert row to a list of integers
        bit_list = list(map(int, row.values.astype(str)))
        
        # Randomly select indices to flip
        indices_to_flip = random.sample(range(len(bit_list)), num_bits)
        
        # Flip the selected bits
        for index in indices_to_flip:
            bit_list[index] = 1 - bit_list[index]
        
        # Convert back to a pandas Series with the original index
        return pd.Series(bit_list, index=row.index)

    # Apply the bit-flipping function to each row
    modified_data = data.apply(flip_bits, axis=1, args=(num_bits_to_change,))
    
    # Ensure the modified data has the same dtypes as the original
    modified_data = modified_data.astype(data.dtypes)
    
    return modified_data



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
    

    encoder1 = LPUFAuthnetDefinition.Encoder1()
    encoder2 = LPUFAuthnetDefinition.Encoder2()
    decoder2 = LPUFAuthnetDefinition.Decoder2 ()
    
    
    checkpoint = torch.load('best_model.pth')

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
    original_encoder1 = LPUFAuthnetDefinition.Encoder1()
    original_encoder2 = LPUFAuthnetDefinition.Encoder2()
    
    #original_encoder1.load_state_dict(torch.load("Encoder1.pth"))
    #original_encoder2.load_state_dict(torch.load("Encoder2.pth"))
    
    
       
    checkpoint = torch.load('best_model.pth')
    original_encoder1.load_state_dict(checkpoint['encoder1_state_dict'])
    original_encoder2.load_state_dict(checkpoint['encoder2_state_dict'])

    # Create enhanced encoders
    enhanced_encoder1 = LPUFAuthnetDefinition.FrozenEncoder1(original_encoder1)
    enhanced_encoder2 = LPUFAuthnetDefinition.FreezedEncoder2(original_encoder2)
    decoder1 = LPUFAuthnetDefinition.Decoder1()
    
    # Load the trained weights for enhanced encoders and decoder
    checkpoint2 = torch.load('best_model2.pth')

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




def AuthenticationAccurcyTest(): 
    ChallengeLen = 32
    ResponseLen = 32
    TrainData= data
    XTrainData = TrainData.iloc[:, :ChallengeLen]
    YTrainData = TrainData.iloc[:, ChallengeLen:ChallengeLen]

    checkpoint = torch.load('best_model.pth')

    encoder1 = LPUFAuthnetDefinition.Encoder1()
    encoder2 = LPUFAuthnetDefinition.Encoder2()
    decoder2 = LPUFAuthnetDefinition.Decoder2 ()
    
    
    checkpoint = torch.load('best_model.pth')

    encoder1.load_state_dict(checkpoint['encoder1_state_dict'])
    encoder2.load_state_dict(checkpoint['encoder2_state_dict'])
    decoder2.load_state_dict(checkpoint['decoder2_state_dict'])
        
    encoder1.eval()
    encoder2.eval()
    decoder2.eval()
    
    original_encoder1 = LPUFAuthnetDefinition.Encoder1()
    original_encoder2 = LPUFAuthnetDefinition.Encoder2()


    checkpoint = torch.load('best_model.pth')
    original_encoder1.load_state_dict(checkpoint['encoder1_state_dict'])
    original_encoder2.load_state_dict(checkpoint['encoder2_state_dict'])

    # Create enhanced encoders
    enhanced_encoder1 = LPUFAuthnetDefinition.FrozenEncoder1(original_encoder1)
    enhanced_encoder2 = LPUFAuthnetDefinition.FreezedEncoder2(original_encoder2)
    decoder1 = LPUFAuthnetDefinition.Decoder1()
    
    # Load the trained weights for enhanced encoders and decoder
    checkpoint2 = torch.load('best_model2.pth')

    enhanced_encoder2.load_state_dict(checkpoint2['encoder2_state_dict'])
    decoder1.load_state_dict(checkpoint2['decoder2_state_dict'])
    
    enhanced_encoder1.eval()
    enhanced_encoder2.eval()
    decoder1.eval()
    
    TrainData= data
    
    XTrainData = TrainData.iloc[:, :ChallengeLen]
    YTrainData = TrainData.iloc[:, ChallengeLen:ChallengeLen]
    
    XTrainData_tensor = torch.FloatTensor(XTrainData.values)

    with torch.no_grad():
        encoded_output = encoder1(XTrainData_tensor)
        encoded_output = encoder2(encoded_output)
        encoded_output = decoder2(encoded_output)
    encoded_output  = encoded_output.round()

    # Calculate reconstruction errors
    mse_loss = nn.MSELoss()

    reconstruction_error1 = mse_loss(encoded_output, XTrainData_tensor)
    CorrctCorrct = average_flipped_bits_and_accuracy(encoded_output, XTrainData_tensor)[2] 
    CorrctFake= 100- CorrctCorrct
    
    FakeData= modify_dataset(TrainData, num_bits_to_change=18)
    TrainData= FakeData
    
    XTrainData = TrainData.iloc[:, :ChallengeLen]
    YTrainData = TrainData.iloc[:, ChallengeLen:ChallengeLen]
    
    XTrainData_tensor = torch.FloatTensor(XTrainData.values)

    with torch.no_grad():
        encoded_output = encoder1(XTrainData_tensor)
        encoded_output = encoder2(encoded_output)
        encoded_output = decoder2(encoded_output)
    encoded_output  = encoded_output.round()

    # Calculate reconstruction errors

    FakeCorrect= average_flipped_bits_and_accuracy(encoded_output, XTrainData_tensor)[2] 
    
    FakeFake= 100- FakeCorrect
    
    plot_confusion_matrix(CorrctCorrct, CorrctFake, FakeCorrect, FakeFake)

    



def main():
    print("Running First Training Accuracy Calculation...")
    AuthenticationAccurcyTest()


if __name__ == "__main__":
    main()
