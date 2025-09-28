import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.models import resnet50
import time

def tuso_model(train_data, train_label, val_data, device, batch_size=64, input_size=224):
    train_tensor = torch.tensor(train_data).float().to(device)
    label_tensor = torch.tensor(train_label).long().to(device)
    val_tensor = torch.tensor(val_data).float().to(device)

    train_dataset = TensorDataset(train_tensor, label_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    domain_specific_augmentations = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=15)
    ])

    class SphericalResNet(nn.Module):
        def __init__(self):
            super(SphericalResNet, self).__init__()
            self.base_model = resnet50(weights='DEFAULT')
            self.base_model.fc = nn.Linear(self.base_model.fc.in_features, len(set(train_label)))
            self.fc = nn.Linear(self.base_model.fc.out_features, len(set(train_label)))

        def forward(self, x):
            x = self.base_model(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    model = SphericalResNet().to(device)

    class_weights = torch.tensor([1.0] * len(set(train_label))).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    for epoch in range(10):
        start_time = time.time()
        for inputs, labels in train_loader:
            inputs = domain_specific_augmentations(inputs)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} data loading time: {time.time() - start_time:.4f} seconds")
        print(f"Batch size: {len(inputs)}, Queue length: {len(train_loader)}, Throughput: {len(train_loader)/ (time.time() - start_time):.2f} batches/sec")

    model.eval()
    with torch.no_grad():
        val_preds = torch.argmax(model(val_tensor), dim=1).cpu().numpy()
    
    return val_preds.tolist()
