
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import numpy as np

class CNN1D(nn.Module):
    def __init__(self):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 46, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 24)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

def tuso_model(train_data, train_label, val_data, device, batch_size=1032):
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    val_data = scaler.transform(val_data)

    seasonal_mean = np.mean(train_data, axis=0)
    train_data = train_data - seasonal_mean
    val_data = val_data - seasonal_mean

    train_tensor = torch.tensor(train_data, dtype=torch.float32).unsqueeze(1).to(device)
    label_tensor = torch.tensor(train_label, dtype=torch.long).to(device)
    val_tensor = torch.tensor(val_data, dtype=torch.float32).unsqueeze(1).to(device)

    dataset = TensorDataset(train_tensor, label_tensor)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    models = []
    for _ in range(5):
        model = CNN1D().to(device)
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        model.train()
        for epoch in range(10):
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        models.append(model)

    model.eval()
    val_outputs = []
    with torch.no_grad():
        for model in models:
            outputs = model(val_tensor)
            val_outputs.append(outputs)

    val_outputs = torch.mean(torch.stack(val_outputs), dim=0)
    _, y_pred = torch.max(val_outputs, 1)

    y_pred_labels = y_pred.cpu().numpy().tolist()
    return y_pred_labels
