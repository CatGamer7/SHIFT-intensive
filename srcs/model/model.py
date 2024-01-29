import torch.nn as nn
import torch.nn.functional as F


class LinearModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.fc1 = nn.Linear(784, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class Conv3(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding="same")
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(3, 3), padding="same")
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(3, 3), padding="same")
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.flatten = nn.Flatten()

        self.dense1 = nn.Linear(in_features=512*3*3, out_features=512)
        self.dropout1 = nn.Dropout1d(p=dropout_rate)
        
        self.dense2 = nn.Linear(in_features=512, out_features=128)
        self.dropout2 = nn.Dropout1d(p=dropout_rate)

        self.dense3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        x = self.flatten(x)

        x = F.relu(self.dense1(x))
        x = self.dropout1(x)

        x = F.relu(self.dense2(x))
        x = self.dropout2(x)

        x = F.log_softmax(self.dense3(x), dim=1)

        return x