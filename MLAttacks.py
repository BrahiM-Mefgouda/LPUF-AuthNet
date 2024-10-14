# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 14:15:17 2024

@author: KU500935
"""

from torch.utils.data import DataLoader, TensorDataset
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
import  LPUFAuthnetDefinition
import ZadiGeneratingDatasetForMLattack
import os
from sklearn import svm
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, RegressorMixin
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data

data = pd.read_csv('CRP_FPGA_01 - copy.csv')
X = data.iloc[:, :32].values  # Challenges
y = data.iloc[:, 32:].values  # Responses

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


def GeneratingLatentSpaceDataset():
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
        LatentChallenge = enhanced_encoder1(XTrainData_tensor)
        LatentResponse= enhanced_encoder2(LatentChallenge)
        
    X_np = LatentChallenge.numpy()
    Y_np = LatentResponse.numpy()
    latent_df = pd.DataFrame(np.hstack([X_np, Y_np]))
    
    latent_df.to_csv('MLAttackDataset.csv', index=False)
    print("Dataset created and saved as 'MLAttackDataset.csv'")





# SVM
class TorchLogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TorchLogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))

class SklearnLogisticRegression(BaseEstimator, RegressorMixin):
    def __init__(self, input_dim=4, output_dim=4, lr=0.01, epochs=100, batch_size=32):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None

    def fit(self, X, y):
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y)
        
        self.model = TorchLogisticRegression(self.input_dim, self.output_dim)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCELoss()
        
        for epoch in range(self.epochs):
            for i in range(0, len(X), self.batch_size):
                batch_X = X[i:i+self.batch_size]
                batch_y = y[i:i+self.batch_size]
                
                optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
            
        
        return self

    def predict(self, X):
        X = torch.FloatTensor(X)
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X)
        return predictions.numpy()

class TorchSVR(nn.Module):
    def __init__(self, input_dim=4, output_dim=4, epsilon=0.1):
        super(TorchSVR, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.epsilon = epsilon

    def forward(self, x):
        return self.linear(x)

    def loss(self, predictions, targets):
        diff = predictions - targets
        loss = torch.mean(torch.max(torch.zeros_like(diff), torch.abs(diff) - self.epsilon))
        return loss

def calculate_accuracy(y_true, y_pred, tolerance=1):
    within_tolerance = np.abs(y_true - y_pred) <= tolerance
    accuracy = np.mean(within_tolerance) * 100  # Convert to percentage
    return accuracy


class SklearnSVR(BaseEstimator, RegressorMixin):
    def __init__(self, input_dim=4, output_dim=3, epsilon=0.1, lr=0.001, epochs=100, batch_size=32):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.epsilon = epsilon
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None

    def fit(self, X, y):
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y)
        
        self.model = TorchSVR(self.input_dim, self.output_dim, self.epsilon)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        for epoch in range(self.epochs):
            for i in range(0, len(X), self.batch_size):
                batch_X = X[i:i+self.batch_size]
                batch_y = y[i:i+self.batch_size]
                
                optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = self.model.loss(predictions, batch_y)
                loss.backward()
                optimizer.step()
            
         #   if (epoch + 1) % 10 == 0:
        #        print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}")
        
        return self

    def predict(self, X):
        X = torch.FloatTensor(X)
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X)
        return predictions.numpy()




# NN 
def calculate_accuracy(y_true, y_pred, tolerance=1):
    """
    Calculate accuracy for 4-dimensional float outputs.
    
    Args:
    y_true (numpy.ndarray): True values, shape (n_samples, 4)
    y_pred (numpy.ndarray): Predicted values, shape (n_samples, 4)
    tolerance (float): Tolerance for considering a prediction correct
    
    Returns:
    float: Accuracy as a percentage
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Check if shapes match
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of y_true and y_pred must match")
    
    # Check if we have 4-dimensional outputs
    if y_true.shape[1] != 4:
        raise ValueError("Expected 4-dimensional outputs")
    
    # Calculate absolute differences
    abs_diff = np.abs(y_true - y_pred)
    
    # Check if each dimension is within tolerance
    within_tolerance = abs_diff <= tolerance
    
    # Calculate accuracy for each sample (all 4 dimensions must be correct)
    sample_accuracy = np.all(within_tolerance, axis=1)
    
    # Calculate overall accuracy
    accuracy = np.mean(sample_accuracy) * 100
    
    return accuracy

# Define separate encoder and decoder
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32,16), nn.ReLU(),
            nn.Linear(16,8),nn.ReLU(),
            nn.Linear(8,4))
        
    def forward(self, x):
        return self.layers(x)
    def average_flipped_bits_and_accuracy(self, X, Y):
        if len(X) != len(Y):
            raise ValueError("Both datasets must have the same number of vectors.")
        total_flipped_bits = 0
        total_flipped_bits_ten_perc_cent = 0
        threshold =0# 10% of 16
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

def save_accuracies(AccuracySVM, AccuracyNN, filename='accuracy_results2.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump({
            'SVR': AccuracySVM,
            'Neural Network': AccuracyNN
        }, f)
    print(f"Accuracy results saved to '{filename}'")

def load_accuracies(filename='accuracy_results2'):
    with open(filename, 'rb') as f:
        loaded_results = pickle.load(f)
    return loaded_results['SVR'], loaded_results['Neural Network']

########################################################################################################
#DataSet



def AttackVPUF():
    data = pd.read_csv('MLAttackDataset.csv')
    
    AccVPUF= []
    X = data.iloc[:, :4].values  # Challenges
    y = data.iloc[:, 4:].values  # Responses
    TrainData = data
    
    XTrainData = TrainData.iloc[:, :4]
    YTrainData = TrainData.iloc[:, 4:]
    
    # Load original encoders
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
    
    # Convert numpy arrays to PyTorch tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    total_accuracy = 0

    with torch.no_grad():  # Add this line to prevent gradient computation
        for i in range(0,len(X)-1):

            encoded_output = enhanced_encoder2(X_tensor[i,:].unsqueeze(0))
            acc = calculate_accuracy(encoded_output, y_tensor[i,:].unsqueeze(0))
            print (i)
            total_accuracy += acc
            AccVPUF.append(acc)
            
    return  AccVPUF


def Attack() : 
    data = pd.read_csv('MLAttackDataset.csv')
    
    X= data.iloc[: , :4].values  # Challenges
    y= data.iloc[:  , 4:].values  # Responses
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the model
    svr = SklearnSVR(input_dim=4, output_dim=4)
    svr.fit(X, y)
    
    # Make predictions
    y_pred = svr.predict(X_test)
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    #print(f"Mean Squared Error: {mse}")
    # Calculate accuracy
    accuracy = calculate_accuracy(y_test, y_pred, tolerance=1)
    print(f"Accuracy (within 0.1 tolerance): {accuracy:.4f}")
    #GeneratingDataSet()  
    data = pd.read_csv('MLAttackDataset.csv')
    
    NeuralNetworkModel = NN()
    NeuralNetworkModel_optimizer = optim.Adam(NeuralNetworkModel.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    AccurcySVM = []
    AccuracyNN  = []
    
    for i in range(1,data.shape[0]): 
        #Stolen 
        X_stolen= data.iloc[: i, :4].values  # Challenges
        y_stolen= data.iloc[: i , 4:].values  # Responses
        svr = SklearnSVR(input_dim=4, output_dim=4)
        svr.fit(X_stolen, y_stolen)
        
        X= data.iloc[: , :4].values  # Challenges
        y= data.iloc[:  , 4:].values  # Responses
        # Make predictions
        y_pred = svr.predict(X)
        
    
        # Evaluate the model
        mse = mean_squared_error( y, y_pred)
        #print(f"SVM Mean Squared Error: {mse}")
        
        # Calculate accuracy
        accuracy = calculate_accuracy( y, y_pred, tolerance=1)
        AccurcySVM.append(accuracy)
    
        if ( i % 10==0):
            print( i)
            print(f"SVM Accuracy (within 0.1 tolerance): {accuracy:.4f}")
        
    
        # Neural Network part
        X_stolen_tensor = torch.FloatTensor(X_stolen)
        y_stolen_tensor = torch.FloatTensor(y_stolen)
        
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        train_dataset = TensorDataset(X_stolen_tensor, y_stolen_tensor)
        train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
        
        for epoch in range(100):  # Reduced number of epochs for demonstration
            total_loss = 0
            total_acc = 0
            for inputs, labels in train_loader:
                NeuralNetworkModel_optimizer.zero_grad()
                outputs = NeuralNetworkModel(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                NeuralNetworkModel_optimizer.step()
            
        # Evaluate Neural Network on the test set
        X_test_tensor = torch.FloatTensor(X)
        y_test_tensor = torch.FloatTensor(y)
        with torch.no_grad():
            nn_pred = NeuralNetworkModel(X_tensor)
        nn_accuracy = calculate_accuracy(y_tensor, nn_pred,1)
        AccuracyNN.append(nn_accuracy)
        if ( i % 10==0):
            print(f"Neural Network Accuracy: {nn_accuracy:.4f}")
    
    
    save_accuracies(AccurcySVM, AccuracyNN)

def load_accuracies(filename='accuracy_results2.pkl'):
    with open(filename, 'rb') as f:
        loaded_results = pickle.load(f)
        
    return loaded_results['SVR'], loaded_results['Neural Network'],AttackVPUF()


def PlotAcc(loaded_AccuracySVM, loaded_AccuracyNN, loaded_AccuracyVPUF):    
    #loaded_AccuracyNN= sorted(loaded_AccuracyNN)#remove
    
    
    
    time= [x  for x in range(1, len(loaded_AccuracySVM) + 1)]

    # Create the plot with a white background
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
    ax.set_facecolor('white')
    
    
    
    print(f"Lengths: SVM={len(loaded_AccuracySVM)}, NN={len(loaded_AccuracyNN)}, VPUF={len(loaded_AccuracyVPUF)}")
    
    ax.plot(time, loaded_AccuracySVM, label='SVM', color='blue', linewidth=2)           #remove
    ax.plot(time, loaded_AccuracyNN, label='Neural Networks', color='red', linewidth=2) #remove
    ax.plot(time, loaded_AccuracyVPUF, label='LPUF-AuthNet', color='green', linewidth=3)

    # Customize the plot
    #ax.set_title('Accuracy Comparison: VPUF, SVM, and Neural Network', fontsize=16, fontweight='bold', color='black')
    ax.set_xlabel('Number of collected LC-LR', fontsize=25, color='black', fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=25, color='black', fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    
    # Set y-axis to percentage scale
    ax.set_ylim(0, 105)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7, color='lightgray', which='both')
    ax.set_axisbelow(True)  # Place gridlines behind the plot elements
    
    # Add dark lines at X=0 and Y=0
    ax.axhline(y=0, color='black', linewidth=3)
    ax.axvline(x=0, color='black', linewidth=1)
    
    # Improve tick label readability
    ax.tick_params(axis='both', which='major', labelsize=20, colors='black')
    

    for spine in ax.spines.values():
        spine.set_color('black')


    legend = ax.legend(fontsize=20, loc='upper right', bbox_to_anchor=(0.98, 0.7), 
                       frameon=True, fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(1)
    for text in legend.get_texts():
        text.set_fontweight('bold')


    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig('accuracy_comparison_dark_axes.pdf', dpi=300, bbox_inches='tight', facecolor='white')
    
    # Display the plot
    plt.show()
    
    print("Plot saved as 'accuracy_comparison_dark_axes.png'")


#GeneratingLatentSpaceDataset()
#Attack()
loaded_AccuracySVM, loaded_AccuracyNN, loaded_AccuracyVPUF= load_accuracies()
PlotAcc(loaded_AccuracySVM, loaded_AccuracyNN,loaded_AccuracyVPUF)

