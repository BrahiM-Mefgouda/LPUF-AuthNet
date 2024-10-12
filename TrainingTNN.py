import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import time
import matplotlib.pyplot as plt
import LPUFAuthnetDefinition as Models



class PUFTrainer:
    def __init__(self, data_path, batch_size=100):
        self.data_path = data_path
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_data()

    def load_data(self):
        data = pd.read_csv(self.data_path, encoding='utf-8')
        X = data.iloc[:, :32].values
        y = data.iloc[:, 32:].values
        self.X_train = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.y_train = torch.tensor(y, dtype=torch.float32).to(self.device)
        self.train_dataset = TensorDataset(self.X_train, self.X_train)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size)

    @staticmethod
    def freeze_model(model):
        for param in model.parameters():
            param.requires_grad = False

    @staticmethod
    def average_flipped_bits_and_accuracy(X, Y):
        if len(X) != len(Y):
            raise ValueError("Both datasets must have the same number of vectors.")

        total_flipped_bits = 0
        total_flipped_bits_ten_perc_cent = 0
        threshold = 1
        Accuracy = 0
        num_pairs = len(X)
        AverageFlipped = []

        for x_vec, y_vec in zip(X, Y):
            if len(x_vec) != len(y_vec):
                raise ValueError("Vectors within each pair must have the same length.")
            flipped_bits = sum(x != y for x, y in zip(x_vec, y_vec))
            total_flipped_bits += flipped_bits
            AverageFlipped.append(flipped_bits)
            if flipped_bits >= threshold:
                total_flipped_bits_ten_perc_cent += 1
            else:
                Accuracy += 1

        average_flipped = total_flipped_bits / num_pairs
        Accuracy = (Accuracy / num_pairs) * 100
        total_flipped_bits_ten_perc_cent /= num_pairs

        return average_flipped, total_flipped_bits_ten_perc_cent, Accuracy

    def train_first_stage(self, num_epochs):
        encoder1 = Models.Encoder1().to(self.device)
        encoder2 = Models.Encoder2().to(self.device)
        decoder2 = Models.Decoder2().to(self.device)

        encoder1_optimizer = optim.Adam(encoder1.parameters(), lr=0.001)
        encoder2_optimizer = optim.Adam(encoder2.parameters(), lr=0.001)
        decoder2_optimizer = optim.Adam(decoder2.parameters(), lr=0.001)

        criterion = nn.MSELoss()

        best_acc = 0
        best_epoch = 0
        best_loss = float('inf')
        losses = []
        accuracies = []

        start_time = time.time()

        for epoch in range(num_epochs):
            epoch_loss = 0
            epoch_acc = 0
            batches = 0

            for data in self.train_loader:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                encoder1_optimizer.zero_grad()
                encoder2_optimizer.zero_grad()
                decoder2_optimizer.zero_grad()

                activation1 = encoder1(inputs)
                activation2 = encoder2(activation1)
                activation3 = decoder2(activation2)

                loss = criterion(activation3, inputs)
                loss.backward()

                encoder2_optimizer.step()
                decoder2_optimizer.step()
                encoder1_optimizer.step()

                A, _, acc = self.average_flipped_bits_and_accuracy(activation3.round().cpu(), labels.cpu())
                epoch_loss += loss.item()
                epoch_acc += acc
                batches += 1

            avg_loss = epoch_loss / batches
            avg_acc = epoch_acc / batches

            losses.append(avg_loss)
            accuracies.append(avg_acc)

            print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}, Acc: {avg_acc:.2f}% | BestLoss={best_loss:.4f}  BestAcc={best_acc:.2f}')

            if avg_acc >= best_acc and avg_loss < best_loss:
                best_acc = avg_acc
                best_epoch = epoch + 1
                best_loss = avg_loss

                print(f"New best model saved with accuracy: {best_acc:.2f}%")

                torch.save({
                    'epoch': epoch,
                    'encoder1_state_dict': encoder1.state_dict(),
                    'encoder2_state_dict': encoder2.state_dict(),
                    'decoder2_state_dict': decoder2.state_dict(),
                    'encoder1_optimizer_state_dict': encoder1_optimizer.state_dict(),
                    'encoder2_optimizer_state_dict': encoder2_optimizer.state_dict(),
                    'decoder2_optimizer_state_dict': decoder2_optimizer.state_dict(),
                    'loss': avg_loss,
                    'accuracy': best_acc,
                }, 'best_model.pth')

        training_time = time.time() - start_time
        print(f"Autoencoder training took {training_time:.2f} seconds")
        print(f"Best accuracy: {best_acc:.2f}% achieved at epoch {best_epoch}")

        self.plot_training_results(losses, accuracies)

    def train_second_stage(self, num_epochs):
        original_encoder1 = Models.Encoder1().to(self.device)
        original_encoder2 = Models.Encoder2().to(self.device)

        checkpoint = torch.load('best_model.pth')
        original_encoder1.load_state_dict(checkpoint['encoder1_state_dict'])
        original_encoder2.load_state_dict(checkpoint['encoder2_state_dict'])

        encoder1 = Models.FrozenEncoder1(original_encoder1).to(self.device)
        encoder2 = Models.FreezedEncoder2(original_encoder2).to(self.device)
        decoder1 = Models.Decoder1().to(self.device)

        encoder2_optimizer = optim.Adam(encoder2.new_layers.parameters(), lr=0.001)
        decoder1_optimizer = optim.Adam(decoder1.parameters(), lr=0.001)

        criterion = nn.MSELoss()

        train_dataset = TensorDataset(self.X_train, self.y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        best_acc = 0
        best_epoch = 0
        best_loss = float('inf')
        losses = []
        accuracies = []

        for epoch in range(num_epochs):
            train_loss = 0
            epoch_accuracy = 0
            num_batches = 0

            for data in train_loader:
                inputs, targets = data
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                encoder2_optimizer.zero_grad()
                decoder1_optimizer.zero_grad()

                encoded1 = encoder1(inputs)
                encoded2 = encoder2(encoded1)
                outputs = decoder1(encoded2)

                loss = criterion(outputs, targets)
                loss.backward()

                encoder2_optimizer.step()
                decoder1_optimizer.step()

                train_loss += loss.item()

                A, _, acc = self.average_flipped_bits_and_accuracy(outputs.round().cpu(), targets.cpu())
                epoch_accuracy += acc
                num_batches += 1

            avg_loss = train_loss / len(train_loader)
            avg_acc = epoch_accuracy / num_batches

            losses.append(avg_loss)
            accuracies.append(avg_acc)

            print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}, Acc: {avg_acc:.2f}% | BestLoss={best_loss:.4f}  BestAcc={best_acc:.2f}')

            if avg_acc >= best_acc and avg_loss < best_loss:
                best_acc = avg_acc
                best_epoch = epoch + 1
                best_loss = avg_loss

                print(f"New best model saved with accuracy: {best_acc:.2f}%")

                torch.save({
                    'epoch': epoch,
                    'encoder2_state_dict': encoder2.state_dict(),
                    'decoder2_state_dict': decoder1.state_dict(),
                    'Freezedencoder2_optimizer_state_dict': encoder2_optimizer.state_dict(),
                    'decoder1_optimizer_state_dict': decoder1_optimizer.state_dict(),
                    'loss': avg_loss,
                    'accuracy': best_acc,
                }, 'best_model2.pth')

        print(f"Best accuracy: {best_acc:.2f}% achieved at epoch {best_epoch}")
        self.plot_training_results(losses, accuracies)

    @staticmethod
    def plot_training_results(losses, accuracies):
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

def main():
    num_epochs= 2^37
    trainer = PUFTrainer('CRP_FPGA_01 - Copy.csv')
    trainer.train_first_stage(num_epochs)
    trainer.train_second_stage(num_epochs)

if __name__ == "__main__":
    main()
