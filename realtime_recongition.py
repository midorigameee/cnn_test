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
FACE_SIZE = 32
BATCH_SIZE = 10

ENTER_KEY = 13 # ENTERキー
ESC_KEY = 27 # ESCキー
CASCADE_PATH = "C:\\workspace_py\\Anaconda3\\envs\\py35\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml"

# 分類器の指定
cascade = cv2.CascadeClassifier(CASCADE_PATH)


def detect_maxsize_faces(target_image):
    target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
    face_list = cascade.detectMultiScale(target_gray, minSize=(200, 200))        

    # 検出した顔に印を付ける
    f_x = 0
    f_y = 0
    f_size = 0
    for (x, y, w, h) in face_list:
        color = (0, 0, 225)
        pen_w = 1

        # 一番大きい検出結果を顔とする
        if w > f_size:
            f_x = x
            f_y = y
            f_size = w

    return len(face_list), f_x, f_y, f_size


def make_thumnail(nail_path):
    nail_list = os.listdir(nail_path)
    blank = np.zeros((60, 128, 3), np.uint8)

    for i, nail in enumerate(nail_list):
        data = os.listdir(os.path.join(nail_path, nail))
        nail_name = os.path.join(nail_path, nail, data[0])
        nail_image = cv2.imread(nail_name)
        nail_image = cv2.resize(nail_image, (128, 128))
        # thumnail = cv2.hconcat([thumnail, nail_image])
        if i == 0:
            thumnail = np.concatenate((nail_image, blank), axis=0)
        else:
            thumnail_temp = np.concatenate((nail_image, blank), axis=0)
            thumnail = np.concatenate((thumnail, thumnail_temp), axis=1)

    return thumnail


"""
resultはモデルから出力された値の
.data(構造体の要素)を想定している。
shapeは(1, class数)でtypeはtensor。
"""
def show_result(thumnail, results, labels):
    x = 0
    y = 128

    # 描画対象区間を初期化
    # これをやらないと文字がどんどん重なる
    # (始点のx, 始点のy)から(終点のx, 終点のx)まで塗りつぶし
    cv2.rectangle(thumnail, (0, 128), (128*6, 128+70), (0,0,0), -1)

    # 各クラスごとに結果を描画する
    for i in range(len(results[0])):
        result_text = str(results[0][i]) 
        result_text = result_text[6:]       # tensorという文字が消えないので

        cv2.putText(thumnail, labels[i], (x,y+20), fontFace=cv2.FONT_HERSHEY_PLAIN,\
            fontScale=1, color=(0,255,255), thickness=2)
        cv2.putText(thumnail, result_text, (x,y+40), fontFace=cv2.FONT_HERSHEY_PLAIN,\
            fontScale=1, color=(255,255,255), thickness=1)

        x += 128

    cv2.imshow("thumnail", thumnail)


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
        # self.softmax = nn.LogSoftmax()
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        return x


def main():
    # ラベルの作成
    label = os.listdir("..\\actress\\train")

    # サムネイルの作成
    thumnail = make_thumnail("..\\actress\\train")

    # モデルのリストア
    model = CNN_32  ().to(device)
    # model = CNN_64().to(device)
    param = torch.load('model.ckpt') # パラメータの読み込み
    model.load_state_dict(param)

    # カメラをキャプチャする
    print("capture_camera")
    cap = cv2.VideoCapture(0) # 0はカメラのデバイス番号

    count = -1

    # ネットワークを推論モードに切り替える
    model.eval()
    with torch.no_grad():   # 推論中は勾配の保存を止める（メモリのリーク？を防ぐため）
        while True:
            # retは画像を取得成功フラグ
            ret, frame = cap.read()

            # フレームから顔の個数と最大値の座標とサイズを取得
            f_num, f_x, f_y, f_size = detect_maxsize_faces(frame)

            # 顔があったら認識を開始する
            if f_num > 0:
                # フレームから顔を抽出
                face_image = frame[f_y:f_y+f_size, f_x:f_x+f_size]

                # 顔画像をリサイズ
                if FACE_SIZE is not None:
                    size = (FACE_SIZE, FACE_SIZE)
                    face_image = cv2.resize(face_image, size)

                # 学習済みモデルに抽出した顔画像を入力
                x = np.transpose(face_image, (2, 0, 1)) # (縦, 横, ch)を(ch, 縦, 横)
                x = np.array([x])
                x = torch.Tensor(x)
                x = Variable(x).to(device)

                # 認識結果
                outputs = model(x)
                _, predicted = torch.max(outputs.data, 1)

                # 結果出力速度の調整
                if count > 50 or count == -1:
                    # 認識結果をサムネイルとともに表示
                    show_result(thumnail, outputs.data, label)
                    count = 0

                # 認識結果をフレームに描画する
                answer = predicted[0]
                cv2.rectangle(frame, (f_x, f_y), (f_x+f_size, f_y+f_size), (0,255,0), thickness=2)
                cv2.putText(frame, label[answer], (f_x,f_y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,\
                    fontScale=2, color=(255,255,255), thickness=3)

            # フレームを画面に表示
            cv2.imshow('recognition', frame)

            count += 1

            # キー入力による処理
            k = cv2.waitKey(1)
            if k == ESC_KEY:        # ESCを押したら終了
                print("Exit...")
                break
            elif k == ENTER_KEY:    # ENTERを押したら顔保存
                print("Save now frame...")
                cv2.imwrite("face_image.jpg", frame)

        # キャプチャを解放する
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()