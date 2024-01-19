import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from model.basicNN import BasicNN
from tqdm import tqdm
import argparse
import os

class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-5):
        """
        Early stops the training if validation loss doesn't improve after a given patience.
        :param patience: (int) How long to wait after last time validation loss improved.
        :param min_delta: (float) Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class CustomDataset(Dataset):
    def __init__(self, data_path, start, end):
        selected_cols = ['value_period' + str(i) for i in range(start, end + 1)]
        data = pd.read_csv(data_path)
        self.data = data[selected_cols]
        self.feature = self.data.iloc[:, :-1]
        self.target = self.data.iloc[:, -1]

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, idx):
        feature = torch.tensor(self.feature.iloc[idx], dtype=torch.float32)
        target = torch.tensor(self.target.iloc[idx], dtype=torch.float32)
        return feature, target

def train(epoch, dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for X, y in tqdm(dataloader, desc=f"Training Epoch {epoch+1}", leave=False):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X).squeeze()
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test(epoch, dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in tqdm(dataloader, desc=f"Testing Epoch {epoch+1}", leave=False):
            X, y = X.to(device), y.to(device)
            pred = model(X).squeeze()
            test_loss += loss_fn(pred, y).item()
        test_loss /= num_batches
    return test_loss

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Generate final data set')
    parser.add_argument('--data_path', type=str, default="", help='path to save the data')
    parser.add_argument('--learning_rate', type=float, default="", help='learning rate')
    parser.add_argument('--batch_size', type=int, default="", help='batch size')
    parser.add_argument('--num_epochs', type=int, default="", help='number of epochs')
    parser.add_argument('--num_simulation', type=int, default="", help='number of epochs')
    parser.add_argument('--file_name', type=str, default="", help='the file name to save the data')
    args = parser.parse_args()

    learning_rate = args.learning_rate
    batch_size = args.batch_size
    epochs = args.num_epochs
    num_simulations = args.num_simulation
    train_ratio = 0.8

    directory_path = f'{args.data_path}/{args.file_name}'

    for idx in range(num_simulations):

        data_dir = os.path.join(directory_path, f'result_{idx}')
        observed_data_path = os.path.join(data_dir, f'observed_data_{idx}.csv')
        output_path = os.path.join(data_dir, f'predicted_transaction_{idx}.csv')
        
        # df
        full_dataset = CustomDataset(observed_data_path, 6, 10)

        # Split the dataset into training and testing sets
        train_size = int(train_ratio * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

        # data loader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        # Device configuration
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using {device} device")

        # Model
        model = BasicNN(4, 128).to(device)
        print(model)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        early_stopping = EarlyStopping(patience=5, min_delta=1e-5)

        for t in range(epochs):
            train(t, train_loader, model, criterion, optimizer)
            t_loss = test(t, test_loader, model, criterion)
            early_stopping(t_loss)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            print(f"Epoch {t+1} \t Validation Loss: {t_loss:.4f}")

        # fit the model
        predict_data = pd.read_csv(observed_data_path)
        pred_selected_cols = ['value_period' + str(i) for i in range(7, 11)]
        numpy_data = predict_data[pred_selected_cols].to_numpy()
        predict_data_tensor = torch.tensor(numpy_data, dtype=torch.float32).to(device)
        model_pred = model(predict_data_tensor)

        # Convert the tensor to a NumPy array
        numpy_array = model_pred.detach().numpy()
        df = pd.DataFrame(numpy_array)
        df.columns = ['predicted_transaction']
        df.to_csv(output_path)

    print("Done!\n")