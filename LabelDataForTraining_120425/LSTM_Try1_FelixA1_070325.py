import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ----- Dataset Definition -----
class SEMGDataset(Dataset):
    def __init__(self, csv_file, sequence_length):
        """
        Args:
            csv_file (str): Path to the CSV file with sEMG data and occasional label rows.
            sequence_length (int): Number of sEMG rows to use as the input sequence.
        """
        self.df = pd.read_csv(csv_file, delimiter=';')
        
        # Define column names for the labels.
        self.node_names = ['W', 
              'F1(0)', 'F1(1)', 'F1(2)', 
              'F2(0)', 'F2(1)', 'F2(2)', 'F2(3)', 
              'F3(0)', 'F3(1)', 'F3(2)', 'F3(3)', 
              'F4(0)', 'F4(1)', 'F4(2)', 'F4(3)', 
              'F5(0)', 'F5(1)', 'F5(2)', 'F5(3)']
        self.label_cols = []
        for node in self.node_names:
            self.label_cols += [f"{node}_X", f"{node}_Y", f"{node}_Z"]
        
        # Assume sEMG channels are in columns 1 to 64 (with column 0 being time).
        self.emg_cols = self.df.columns[1:65].tolist()
        
        # Determine which rows have labels by checking one of the label columns (using W_X as reference).
        self.label_rows = self.df[~self.df['W_X'].isna()].index.tolist()
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.label_rows)
    
    def __getitem__(self, idx):
        label_idx = self.label_rows[idx]
        start_idx = label_idx - self.sequence_length
        if start_idx < 0:
            pad_count = -start_idx
            pad_data = self.df.iloc[0][self.emg_cols].values.reshape(1, -1)
            seq_data = self.df.iloc[0:label_idx][self.emg_cols].values
            seq_data = np.vstack([np.repeat(pad_data, pad_count, axis=0), seq_data])
        else:
            seq_data = self.df.iloc[start_idx:label_idx][self.emg_cols].values
        
        seq_data = seq_data.astype(np.float32)
        label_data = self.df.iloc[label_idx][self.label_cols].values.astype(np.float32)
        
        emg_tensor = torch.tensor(seq_data)           # shape: (sequence_length, 64)
        label_tensor = torch.tensor(label_data)         # shape: (60,)
        return emg_tensor, label_tensor

# ----- LSTM Model Definition -----
class sEMGLSTM(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, num_layers=2, output_dim=60, dropout=0.5):
        super(sEMGLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, 
                            hidden_size=hidden_dim, 
                            num_layers=num_layers, 
                            batch_first=True, 
                            dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        out, (hn, cn) = self.lstm(x)  # out: (batch, sequence_length, hidden_dim)
        last_time_step = out[:, -1, :]  # Use the last time step's output
        last_time_step = self.dropout(last_time_step)
        out = self.fc(last_time_step)   # (batch, output_dim)
        return out

# ----- Training Setup -----
csv_file = 'CombinedDatasets_Felix070325_ABE.csv'
sequence_length = 40  
batch_size = 4096
learning_rate = 0.0008
num_epochs = 200

# Create the full dataset (which now contains independent sEMG sequences as defined by __getitem__).
full_dataset = SEMGDataset(csv_file, sequence_length)

# Instead of a temporal split, we randomly split the dataset into training (80%) and test (20%).
total_samples = len(full_dataset)
train_size = int(0.8 * total_samples)
test_size = total_samples - train_size

train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Set up device for GPU usage.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Move the model to GPU.
model = sEMGLSTM(input_dim=64, hidden_dim=128, num_layers=2, output_dim=60, dropout=0.5).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# For saving the best model (lowest test MSE).
best_test_mse = float('inf')
model_save_path = 'bestLSTMmodel_FelixA1_070325Try3.pth'

# ----- Training Loop with Evaluation on Test Set and Model Saving -----
best_test_mse = float('inf')
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    train_preds = []
    train_labels = []
    
    for batch in train_loader:
        emg_seq, labels = batch  # emg_seq: (batch, sequence_length, 64), labels: (batch, 60)
        # Move batch data to GPU.
        emg_seq = emg_seq.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(emg_seq)
        loss = criterion(outputs, labels)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        
        train_preds.append(outputs.detach().cpu().numpy())
        train_labels.append(labels.detach().cpu().numpy())
    
    train_preds = np.concatenate(train_preds, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    
    train_mse = mean_squared_error(train_labels, train_preds)
    train_rmse = np.sqrt(train_mse)
    train_mae = np.mean(np.abs(train_labels - train_preds))
    train_r2 = r2_score(train_labels, train_preds)
    
    # Evaluate on the test set.
    model.eval()
    test_preds = []
    test_labels = []
    with torch.no_grad():
        for batch in test_loader:
            emg_seq, labels = batch
            emg_seq = emg_seq.to(device)
            labels = labels.to(device)
            outputs = model(emg_seq)
            test_preds.append(outputs.detach().cpu().numpy())
            test_labels.append(labels.detach().cpu().numpy())
    test_preds = np.concatenate(test_preds, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    
    test_mse = mean_squared_error(test_labels, test_preds)
    test_rmse = np.sqrt(test_mse)
    test_mae = np.mean(np.abs(test_labels - test_preds))
    test_r2 = r2_score(test_labels, test_preds)
    
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"  Train Loss: {epoch_loss/len(train_loader):.4f} | MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}")
    print(f"  Test  Metrics: MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")
    
    # Save the model if the test MSE improves.
    if test_mse < best_test_mse:
        best_test_mse = test_mse
        torch.save(model.state_dict(), model_save_path)
        print(f"  Best model saved at epoch {epoch+1} with Test MSE: {test_mse:.4f}")

# --- To Load the Model Later for Testing ---
# Example:
# model = sEMGLSTM(input_dim=64, hidden_dim=128, num_layers=2, output_dim=60, dropout=0.5)
# model.load_state_dict(torch.load(model_save_path))
# model.to(device)
# model.eval()
