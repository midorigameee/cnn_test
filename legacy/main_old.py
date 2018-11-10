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
import torchvision.transforms as transforms

import os
import cv2
import numpy

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("device:", device)

# Hyper parameters
num_epochs = 5
num_classes = 10
batch_size = 10
learning_rate = 0.001


"""
# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='../data/',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../data/',
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


print("train_dataset.image:\n{}\n" .format(train_dataset.image))
print("train_dataset.landmarks:\n{}\n" .format(train_dataset.landmarks))
# print("train_dataset:\n{}\n" .format(train_dataset))
# print("test_dataset:\n{}\n" .format(test_dataset))
# print("train_loader:\n{}\n" .format(train_loader))
# print("test_loader:\n{}\n" .format(test_loader))
"""


# dataset_dirは以下のような構造を持つディレクトリであること
#
# ../dataset_dir
#   |- classA
#       |- imageA
#       |- imageB
#       ...
#   |- classB
#   ...
#   |- classX
#
# datasetは正解ラベル
def makeDataset(dataset_dir, color_type=cv2.IMREAD_GRAYSCALE):
    # 分類したいクラスをリスト化
    class_list = sort(os.listdir(dataset_dir))
    class_num = len(class_list)

    # data_x, data_yはそれぞれ学習データと教師データ
    #   data_yはone-hotベクトル
    # labelはインデックスに対応するラベル名
    data_x = []
    data_y = []
    label = []

    for class_idx, class_name in enumrate(class_list):
        # クラス内にある全データをリスト化
        class_path = os.path.join(dataset_dir, class_name)
        image_list = os.listdir(class_path)

        # クラス名を保存
        label.append(class_name)

        for image_name in image_list:
            # 画像の読み込み
            image_path = os.path.join(class_path, image_name)
            image_data = cv2.imread(image_path, color_type)

            # 画像をリストに追加
            data_x.append(image_data)

            # 教師データを作成し、リストに追加
            supervisor = np.zeros(class_num)
            supervisor[class_idx] = 1
            data_y.append(supervisor)


    # データセットをlistからnumpyに変換
    data_x = np.array(data_x, dtype="float32")
    data_y = np.array(data_y, dtype="int32")

    return data_x, data_y, labe

"""
# https://qiita.com/sheep96/items/0c2c8216d566f58882aa
class MyDataset(Dataset):
    def __init__(self, dataset_dir, root_dir, transform=None):
        # 学習データのパス設定
        self.dataset_dir = dataset_dir
        self.root_dir = root_dir
        # 画像データへの処理
        self.transform = transform

    def __len__(self):
        return len(self.image_dataframe)

    def __getitem__(self, idx):
        # qaa
        label_list = os.listdir(dataset_dir)


        # dataframeから画像へのパスとラベルを読み出す
        label = self.image_dataframe.iat[idx, LABEL_IDX]
        img_name = os.path.join(self.root_dir, 'classification-of-handwritten-letters',
                'letters2', self.image_dataframe.iat[idx, IMG_IDX])
        # 画像の読み込み
        image = io.imread(img_name)
        # 画像へ処理を加える
        if self.transform:
            image = self.transform(image)

        return image, label
"""


# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


dataset_dir = "../training_data_32"
train_x, train_y, label = makeDataset(dataset_dir)

model = ConvNet(num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # print("images\ntype:{}, size:{}" .format(type(images), images.size()))
        # print("labels\ntype:{}, size:{}" .format(type(labels), labels.size()))
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')