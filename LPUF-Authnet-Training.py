import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import ZadiDefinition as Definition

class CRPTrainer:
    def __init__(self, csv_file, challenge_size=32, batch_size=100, learning_rate=0.001):
        self.challenge_size = challenge_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.data = self.load_data(csv_file)
        self.train_loader = self.prepare_data()
        self.criterion = nn.MSELoss()

    def load_data(self, csv_file):
        """Load data from CSV file."""
        return pd.read_csv(csv_file, encoding='utf-8')

    def prepare_data(self):
        """Prepare data for training."""
        X = self.data.iloc[:, :self.challenge_size].values
        y = self.data.iloc[:, self.challenge_size:].values
        X_train = torch.tensor(X, dtype=torch.float32)
        y_train = torch.tensor(y, dtype=torch.float32)
        train_dataset = TensorDataset(X_train, X_train)
        return DataLoader(train_dataset, batch_size=self.batch_size)

    @staticmethod
    def freeze_model(model):
        """Freeze model parameters."""
        for param in model.parameters():
            param.requires_grad = False

    @staticmethod
    def average_flipped_bits_and_accuracy(X, Y):
        """Calculate average flipped bits and accuracy."""
        if len(X) != len(Y):
            raise ValueError("Both datasets must have the same number of vectors.")

        total_flipped_bits = 0
        total_flipped_bits_ten_perc_cent = 0
        threshold = 1
        accuracy = 0
        num_pairs = len(X)
        average_flipped = []

        for x_vec, y_vec in zip(X, Y):
            if len(x_vec) != len(y_vec):
                raise ValueError("Vectors within each pair must have the same length.")
            flipped_bits = sum(x != y for x, y in zip(x_vec, y_vec))
            total_flipped_bits += flipped_bits
            average_flipped.append(flipped_bits)
            if flipped_bits >= threshold:
                total_flipped_bits_ten_perc_cent += 1
            else:
                accuracy += 1

        average_flipped = total_flipped_bits / num_pairs
        accuracy = (accuracy / num_pairs) * 100
        total_flipped_bits_ten_perc_cent /= num_pairs

        return average_flipped, total_flipped_bits_ten_perc_cent, accuracy

    def train_first_phase(self, num_epochs=5000):
        """Train the first phase of the model."""
        encoder1 = Definition.Encoder1()
        encoder2 = Definition.Encoder2()
        decoder2 = Definition.Decoder2()

        encoder1_optimizer = optim.Adam(encoder1.parameters(), lr=self.learning_rate)
        encoder2_optimizer = optim.Adam(encoder2.parameters(), lr=self.learning_rate)
        decoder2_optimizer = optim.Adam(decoder2.parameters(), lr=self.learning_rate)

        start_time = time.time()
        best_acc = 0
        best_epoch = 0
        best_loss = float('inf')
        losses = []
        accuracies = []

        for epoch in range(num_epochs):
            epoch_loss, epoch_acc, batches = self._train_epoch(encoder1, encoder2, decoder2,
                                                               encoder1_optimizer, encoder2_optimizer, decoder2_optimizer)

            avg_loss = epoch_loss / batches
            avg_acc = epoch_acc / batches

            losses.append(avg_loss)
            accuracies.append(avg_acc)

            print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}, Acc: {avg_acc:.2f}% | BestLoss={best_loss}  BestAcc={best_acc}')

            if avg_acc >= best_acc and avg_loss < best_loss:
                best_acc = avg_acc
                best_epoch = epoch + 1
                best_loss = avg_loss
                self._save_model(epoch, encoder1, encoder2, decoder2, encoder1_optimizer, encoder2_optimizer, decoder2_optimizer, avg_loss, best_acc, 'best_model.pth')

        self._print_training_summary(start_time, best_acc, best_epoch)
        self._plot_training_results(losses, accuracies)

    def _train_epoch(self, encoder1, encoder2, decoder2, encoder1_optimizer, encoder2_optimizer, decoder2_optimizer):
        """Train a single epoch."""
        epoch_loss = 0
        epoch_acc = 0
        batches = 0

        for data in self.train_loader:
            inputs, labels = data

            encoder1_optimizer.zero_grad()
            encoder2_optimizer.zero_grad()
            decoder2_optimizer.zero_grad()

            activation1 = encoder1(inputs)
            activation2 = encoder2(activation1)
            activation3 = decoder2(activation2)

            loss = self.criterion(activation3, inputs)
            loss.backward()

            encoder2_optimizer.step()
            decoder2_optimizer.step()
            encoder1_optimizer.step()

            _, _, acc = self.average_flipped_bits_and_accuracy(activation3.round(), labels)
            epoch_loss += loss.item()
            epoch_acc += acc
            batches += 1

        return epoch_loss, epoch_acc, batches

    def _save_model(self, epoch, encoder1, encoder2, decoder2, encoder1_optimizer, encoder2_optimizer, decoder2_optimizer, loss, accuracy, filename):
        """Save the model."""
        torch.save({
            'epoch': epoch,
            'encoder1_state_dict': encoder1.state_dict(),
            'encoder2_state_dict': encoder2.state_dict(),
            'decoder2_state_dict': decoder2.state_dict(),
            'encoder1_optimizer_state_dict': encoder1_optimizer.state_dict(),
            'encoder2_optimizer_state_dict': encoder2_optimizer.state_dict(),
            'decoder2_optimizer_state_dict': decoder2_optimizer.state_dict(),
            'loss': loss,
            'accuracy': accuracy,
        }, filename)

    def _print_training_summary(self, start_time, best_acc, best_epoch):
        """Print training summary."""
        training_time = time.time() - start_time
        print(f"Autoencoder training took {training_time:.2f} seconds")
        print(f"Best accuracy: {best_acc:.2f}% achieved at epoch {best_epoch}")

    def _plot_training_results(self, losses, accuracies):
        """Plot training results."""
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.subplot(1, 2, 2)
        plt.plot(accuracies)
        plt.title('Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')

        plt.tight_layout()
        plt.savefig('training_plots.png')
        plt.show()

    def train_second_phase(self, num_epochs=10000):
        """Train the second phase of the model."""
        # Load the best model from the first phase
        checkpoint = torch.load('best_model.pth')
        
        original_encoder1 = Definition.Encoder1()
        original_encoder2 = Definition.Encoder2()
        original_encoder1.load_state_dict(checkpoint['encoder1_state_dict'])
        original_encoder2.load_state_dict(checkpoint['encoder2_state_dict'])

        encoder1 = Definition.FrozenEncoder1(original_encoder1)
        encoder2 = Definition.FreezedEncoder2(original_encoder2)
        decoder1 = Definition.Decoder1()

        encoder2_optimizer = optim.Adam(encoder2.new_layers.parameters(), lr=self.learning_rate)
        decoder1_optimizer = optim.Adam(decoder1.parameters(), lr=self.learning_rate)

        train_losses = []
        val_losses = []
        best_acc = 0
        best_loss = float('inf')

        for epoch in range(num_epochs):
            train_loss, epoch_accuracy = self._train_second_phase_epoch(encoder1, encoder2, decoder1, encoder2_optimizer, decoder1_optimizer)

            avg_loss = train_loss / len(self.train_loader)
            avg_acc = epoch_accuracy / len(self.train_loader)

            train_losses.append(avg_loss)
            val_losses.append(avg_loss)  # Placeholder for validation loss

            print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}, Acc: {avg_acc:.2f}% | BestLoss={best_loss}  BestAcc={best_acc}')

            if avg_acc >= best_acc and avg_loss < best_loss:
                best_acc = avg_acc
                best_loss = avg_loss
                self._save_second_phase_model(epoch, encoder2, decoder1, encoder2_optimizer, decoder1_optimizer, avg_loss, best_acc, 'best_model2.pth')

        self._plot_second_phase_results(train_losses, val_losses)

    def _train_second_phase_epoch(self, encoder1, encoder2, decoder1, encoder2_optimizer, decoder1_optimizer):
        """Train a single epoch in the second phase."""
        train_loss = 0
        epoch_accuracy = 0

        for data in self.train_loader:
            inputs, targets = data
            encoder2_optimizer.zero_grad()
            decoder1_optimizer.zero_grad()

            encoded1 = encoder1(inputs)
            encoded2 = encoder2(encoded1)
            outputs = decoder1(encoded2)

            loss = self.criterion(outputs, targets)
            loss.backward()

            encoder2_optimizer.step()
            decoder1_optimizer.step()

            train_loss += loss.item()

            _, _, acc = self.average_flipped_bits_and_accuracy(outputs.round(), targets)
            epoch_accuracy += acc

        return train_loss, epoch_accuracy

    def _save_second_phase_model(self, epoch, encoder2, decoder1, encoder2_optimizer, decoder1_optimizer, loss, accuracy, filename):
        """Save the model for the second phase."""
        torch.save({
            'epoch': epoch,
            'encoder2_state_dict': encoder2.state_dict(),
            'decoder1_state_dict': decoder1.state_dict(),
            'Freezedencoder2_optimizer_state_dict': encoder2_optimizer.state_dict(),
            'decoder1_optimizer_state_dict': decoder1_optimizer.state_dict(),
            'loss': loss,
            'accuracy': accuracy,
        }, filename)

    def _plot_second_phase_results(self, train_losses, val_losses):
        """Plot results for the second phase."""
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('second_phase_training_plots.png')
        plt.show()

# Usage
if __name__ == "__main__":
    trainer = CRPTrainer('CRP_FPGA_01 - Copy.csv')
    trainer.train_first_phase()
    trainer.train_second_phase()