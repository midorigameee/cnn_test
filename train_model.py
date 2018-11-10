"""
pytorchの基本的な流れ
（参考：http://www.ie110704.net/2017/08/31/pytorch%E3%81%A7%E3%83%8B%E3%83%A5%E3%83%BC%E3%83%A9%E3%83%AB%E3%83%8D%E3%83%83%E3%83%88%E3%83%AF%E3%83%BC%E3%82%AF%E3%80%81rnn%E3%80%81cnn%E3%82%92%E5%AE%9F%E8%A3%85%E3%81%97%E3%81%A6%E3%81%BF/）

データセットを作成
それをTensor化しDataloaderでイテレータにする
モデルを定義する
学習させる

"""


import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms, datasets

import os
import numpy as np


# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("device:", device)

# Hyper parameters
NUM_EPOCHS = 50
NUM_CLASSES = 6
IMAGE_SIZE = 64
HIDDEN_SIZE = 512
BATCH_SIZE = 10
LEARNING_RATE = 0.001


# CNN
#   https://qiita.com/kazetof/items/6a72926b9f8cd44c218e
# conv2dの引数は左から，インプットのチャンネルの数，アウトプットのチャンネルの数，カーネルサイズ
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


"""
(3, 64, 64)
conv1(3, 6, 5)   => (6, 60, 60)
pool1(2, 2)      => (6, 30, 30)
conv2(6, 16, 5)  => (16, 26, 26)
pool2(2, 2)      => (16, 13, 13)
conv3(16, 32, 4) => (32, 10, 10)
pool2(2, 2)      => (32, 5, 5)

32 * 5 * 5 = 800 => 400
400 => 120
120 => 84
84 => 4
"""
class CNN_64(nn.Module):
    def __init__(self):
        super(CNN_64, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 4)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 5 * 5, 400)
        self.fc2 = nn.Linear(400, 120)
        self.fc3 = nn.Linear(120, 84)
        self.fc4 = nn.Linear(84, 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 32 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# MLP
#   http://aidiary.hatenablog.com/entry/20180204/1517705138
class MultiLayerPerceptron(nn.Module):
    def __init__(self, image_size, hidden_size, num_classes):
        super(MultiLayerPerceptron, self).__init__()
        # ハイパーパラメータ
        self.image_size = image_size
        self.input_size = image_size * image_size * 3  # RGB想定なので*3
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        # ネットワーク構造
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax()
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x):
        x = x.view(-1, self.image_size * self.image_size * 3)  # RGB想定なので*3
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.softmax(out)
        return out


# 取り込んだデータに施す処理を指定
data_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])

# データセットの読み込み
train = datasets.ImageFolder(root='..\\training_data_ImageFolder\\train',
                             transform=data_transform)
test = datasets.ImageFolder(root='..\\training_data_ImageFolder\\train',
                             transform=data_transform)

# DataLoader化
train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)

# モデル、損失関数、最適化関数の定義
# model = MultiLayerPerceptron(IMAGE_SIZE, HIDDEN_SIZE, NUM_CLASSES).to(device)
model = CNN_64().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Train the model
total_step = len(train_loader)
for epoch in range(NUM_EPOCHS):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 5 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, NUM_EPOCHS, i+1, total_step, loss.item()))

# Test the model
model.eval()  # ネットワークを推論モードに切り替える
with torch.no_grad():   # 推論中は勾配の保存を止める（メモリのリーク？を防ぐため）
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy: {} %' .format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')