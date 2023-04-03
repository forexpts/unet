#pytorch
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
#その他
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import random

#transforms
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ])
target_transform = transforms.Compose([transforms.ToTensor(),])

#データセット
class LoadDataSet(Dataset):
    """
    データセットのディレクトリ構造により変更が必要
    基本的に指定したidxの(img, mask)さえ返せればおｋ
    """

    def __init__(self, path, N = 256,transform=transform, target_transform=target_transform):
        """
        Nはdatasetのデータの大きさ, N×Nになる
        transformはデータの事前処理に使う関数 target_transformもある
        """
        self.path = path
        self.N = N
        self.folders = os.listdir(path)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        image_folder = os.path.join(self.path, self.folders[idx], "images/")
        mask_folder = os.path.join(self.path, self.folders[idx], "masks/")
        
        #img取得
        img_path = os.path.join(image_folder, os.listdir(image_folder)[0])
        img = cv2.imread(img_path)
        #img = img.astype(np.float32)
        img = cv2.resize(img, (self.N, self.N)) #numpy (256,256,3) float32

        #mask取得
        mask = self.get_mask(mask_folder) #numpy (256,256,1) bool

        #numpyからtensorに変更
        img = transform(img)
        mask = target_transform(mask)
        #print_type("img", img)
        #print_type("mask", mask)
        return (img, mask)

    def get_mask(self, mask_folder):
        """
        まずN×Nの二値行列を作る
        そこにfor文で細胞のマスク画像を足していき、最終的なmaskを作る
        """
        mask = np.zeros((self.N, self.N, 1), dtype = bool) #numpy (256,256,1) bool
        
        for mask_i in os.listdir(mask_folder):
            mask_i = cv2.imread(os.path.join(mask_folder, mask_i), cv2.IMREAD_GRAYSCALE)
            mask_i = cv2.resize(mask_i, (self.N, self.N))
            mask_i = np.expand_dims(mask_i,axis=-1)
            mask_i = mask_i.astype(bool) #numpy (256, 256, 1) bool :print_type("mask_i", mask_i)
            mask = mask+mask_i
        
        return mask

    def print_folders(self, n=3):
        """
        foldersの中身を確認
        """
        [print(self.folders[i]) for i in range(n)]

#データセット
class TestDataSet(Dataset):
    def __init__(self, path, N = 256, transform=transform):
        """
        Nはdatasetのデータの大きさ, N×Nになる
        transformはデータの事前処理に使う関数 target_transformもある
        """
        self.path = path
        self.N = N
        self.folders = os.listdir(path)
        self.transform = transform

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        image_folder = os.path.join(self.path, self.folders[idx], "images/")
        
        #img取得
        img_path = os.path.join(image_folder, os.listdir(image_folder)[0])
        img = cv2.imread(img_path)
        #img = img.astype(np.float32)
        img = cv2.resize(img, (self.N, self.N)) #numpy (256,256,3) float32
        img = transform(img)
        return img

    def print_folders(self, n=3):
        """
        foldersの中身を確認
        """
        [print(self.folders[i]) for i in range(n)]

#データセットの中身確認
def visualize_dataset(N = 3, dataset = None, nums = []):
    """
    データセットのimageとmaskを図にして確認できる関数
    datasetは表示したいデータセット
    numsは表示したい番号で構成された配列、空だったらN個ランダムでサンプリング
    """
    if not nums:
        nums = random.sample(range(0, dataset.__len__()), N)
    figure, ax = plt.subplots(nrows=len(nums), ncols=2, figsize=(5, 8))

    ax[0, 0].set_title("Image")
    ax[0, 1].set_title("Mask")
    for i, img_num in enumerate(nums):
        img, mask = dataset.__getitem__(img_num)
        img = img.numpy().transpose((1,2,0))
        mean=np.array((0.485, 0.456, 0.406))
        std=np.array((0.229, 0.224, 0.225))
        img  = std * img + mean
        img = img*255
        img = img.astype(np.uint8)
        mask = mask.numpy().transpose((1,2,0))
        print_type("img", img)
        print_type("mask", mask)

        ax[i, 0].text(-0.2, 0.5, f"{img_num}", fontsize=12, ha='right', va='center', transform=ax[i, 0].transAxes)
        ax[i, 0].imshow(img)
        ax[i, 1].imshow(mask)
        ax[i, 0].set_xlabel(img_num)
        ax[i, 0].set_axis_off()
        ax[i, 1].set_axis_off()
    
    plt.tight_layout()
    plt.savefig("visualize_dataset.png")

def print_type(name, arr):
    print(f"{name} , 型:{type(arr)}, 形{arr.shape}, データ型:{arr.dtype}")


if __name__ == '__main__':
    """
    以下はdataset.pyの挙動確認用のスクリプト 
    """
    TRAIN_PATH = "../data/stage1_train"
    train_dataset = LoadDataSet(TRAIN_PATH)
    visualize_dataset(5, train_dataset)


"""
#メモ

画像を読み込む手法は以下が有名
・OpenCV
・Pillow
・scikit-image
読み込みだけならPillowが早いらしいが、回転などの処理を加えるとなるとOpenCVが早いらしい
ただ公式はscikit-imageを推奨？
参考:https://www.kaggle.com/code/vfdev5/pil-vs-opencv/notebook
"""
