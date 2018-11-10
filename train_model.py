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
import numpy as np
from sklearn.model_selection import train_test_split


# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("device:", device)

# Hyper parameters
NUM_EPOCHS = 50
NUM_CLASSES = 4
IMAGE_SIZE = 32
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

# dataset_dirは以下のような構造を持つディレクトリであること
#
# ../dataset_dir
#   |- classA
#       |- imageA
#       |- imageB
#       ...
#   |- classB
#       |- imageX
#       |- imageY
#       ...
#   ...
#
# 現在は全て画像サイズが同じRGB画像を想定したデータセット作成関数
# グレースケールも無理やりRGB画像として読み込む仕様
def makeDataset(dataset_dir):
    # 分類したいクラスをリスト化
    class_list = os.listdir(dataset_dir)
    class_list.sort()
    class_num = len(class_list)

    # data_x, data_yはそれぞれ学習データと教師データ
    #   data_yはone-hotベクトル
    # labelはインデックスに対応するラベル名
    data_x = []
    data_y = []
    label = []

    for class_idx, class_name in enumerate(class_list):
        # クラス内にある全データをリスト化
        class_path = os.path.join(dataset_dir, class_name)
        image_list = os.listdir(class_path)

        # クラス名を保存
        label.append(class_name)

        for image_name in image_list:
            # 画像の読み込み
            image_path = os.path.join(class_path, image_name)
            image_data = cv2.imread(image_path, cv2.IMREAD_COLOR)

            # (縦, 横, チャンネル)を(チャンネル, 縦, 横)に変換
            # CNNで学習させる場合はこちらの方が使いやすいため
            # 通常のMLPの際は1次元にするためどちらでも問題ない
            image_data = np.transpose(image_data, (2, 0, 1))

            # 画像をリストに追加
            data_x.append(image_data)

            # one-hot表現の教師データを作成し、リストに追加
            supervisor = np.zeros(class_num)
            supervisor[class_idx] = 1
            data_y.append(supervisor)

    # データセットをlistからnumpyに変換
    data_x = np.array(data_x, dtype="float32")
    data_y = np.array(data_y, dtype="int32")

    # 訓練用とテスト用に分割
    train_x, test_x, train_y, test_y = train_test_split(
            data_x, data_y, test_size=1/5, random_state=0)

    return train_x, test_x, train_y, test_y, label


# データセットを作成しTensor化
dataset_dir = "..\\training_data_32"
train_x, test_x, train_y, test_y, label = makeDataset(dataset_dir)
train_x = torch.from_numpy(train_x)     # torch.Tensorでも大丈夫
train_y = torch.from_numpy(train_y)     # torch.LongTensorでも大丈夫
test_x = torch.from_numpy(test_x)
test_y = torch.from_numpy(test_y)

# labelを保存
with open("..\\label.txt", "w") as list:
    for l in label:
        list.write(str(l) + "\n")

# DataLoader化
train = torch.utils.data.TensorDataset(train_x, train_y)
train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
test = torch.utils.data.TensorDataset(test_x, test_y)
test_loader = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)

# モデル、損失関数、最適化関数の定義
# model = MultiLayerPerceptron(IMAGE_SIZE, HIDDEN_SIZE, NUM_CLASSES).to(device)
model = CNN().to(device)
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
        loss = criterion(outputs, torch.max(labels, 1)[1])

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
        correct += (predicted == torch.max(labels, 1)[1]).sum().item()

    print('Test Accuracy: {} %' .format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')