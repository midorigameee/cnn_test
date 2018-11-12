"""
target_dir
    |-imageA
    |-imageB
    ...

上記のような構造を持つディレクトリ内の各画像に対して、学習済みのモデルを用いて認識を行う。
現状
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms, datasets
from torch.autograd import Variable

import os
import numpy as np
import cv2

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("device:", device)

# Hyper parameters
NUM_CLASSES = 6
IMAGE_SIZE = 32
BATCH_SIZE = 10


# CNN
#   https://qiita.com/kazetof/items/6a72926b9f8cd44c218e
# conv2dの引数は左から，インプットのチャンネルの数，アウトプットのチャンネルの数，カーネルサイズ
"""
input(3, 32, 32)
conv1(3, 6, 5)   => (6, 28, 28)
pool1(2, 2)      => (6, 14, 14)
conv2(6, 16, 5)  => (16, 10, 10)
pool2(2, 2)      => (16, 5, 5)

16 * 5 * 5 = 400 => 120
120 => 84
84 => 4
"""
class CNN_32(nn.Module):
    def __init__(self):
        super(CNN_32, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, NUM_CLASSES)
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
input(3, 64, 64)
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
        self.fc4 = nn.Linear(84, NUM_CLASSES)
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

# 対象画像のロード
target_dir = "..\\target_images"
target_list = os.listdir(target_dir)
target_list.sort()

target = []
for target_name in target_list:
    target_path = class_path = os.path.join(target_dir, target_name)
    target_data = cv2.imread(target_path, cv2.IMREAD_COLOR)
    target_data = cv2.resize(target_data, (IMAGE_SIZE, IMAGE_SIZE))
    target_data = np.transpose(target_data, (2, 0, 1))
    target.append(target_data)

# ラベルの作成
label = os.listdir("..\\actress\\train")

# モデルのリストア
model = CNN_32  ().to(device)
# model = CNN_64().to(device)
param = torch.load('model.ckpt') # パラメータの読み込み
model.load_state_dict(param)

# ネットワークを推論モードに切り替える
model.eval()
with torch.no_grad():   # 推論中は勾配の保存を止める（メモリのリーク？を防ぐため）
    for idx, x in enumerate(target):
        x = np.array([x])
        x = torch.Tensor(x)  # torch.from_numpy(x)だとエラー
        x = Variable(x)      # dataとgradを持つデータになる
        x = x.to(device)

        outputs = model(x)
        _, predicted = torch.max(outputs.data, 1)

        answer = predicted[0]

        print("{} : {}" .format(target_list[idx], label[answer]))
