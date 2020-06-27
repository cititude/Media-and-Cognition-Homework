import torch
import torchvision
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
import json
import tqdm
import random

class_id_list = list()
for classdir in os.listdir(os.path.join("../image_exp/Classification/Data", "Train")):
    class_id_list.append(classdir)

ndata=14463  # number of data
val_id=[]  # validation data indexes, 1000 in aggregate
with open("val_id.txt") as f:
    for line in f.readlines():
        val_id.append(int(line))
class TrafficDataset(Dataset):
    def __init__(self, data_path, mode="train", resize=224,maxdata=99999):
        super(TrafficDataset, self).__init__()
        self.data_path = data_path
        self.mode = mode
        self.resize = resize
        self.data = []
        self.labels = []
        self.maxdata=maxdata
        self.train_transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                                    transforms.Resize((self.resize, self.resize)),
                                                    # transforms.RandomRotation(10),   # data augmentation
                                                    # transforms.ColorJitter(0.2,0.2,0.2)
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(
                                                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                                    ])
        self.test_transform=transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                                    transforms.Resize((self.resize, self.resize)),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(
                                                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                                    ])
        if mode == "train" or mode=="val":
            cnt=0
            data_path = os.path.join(data_path, "Train")
            for classdir in tqdm.tqdm(os.listdir(data_path)):
                class_data_path = os.path.join(data_path, classdir)
                for img_name in os.listdir(class_data_path):
                    if cnt==self.maxdata:
                        break
                    if (mode=="train") ^ (cnt not in val_id):
                            cnt=cnt+1
                            continue
                    img_path = os.path.join(class_data_path, img_name)
                    try:
                        if mode=="train":
                            img = self.train_transform(img_path)
                        else:
                            img=self.test_transform(img_path)
                        self.data.append(img)
                        self.labels.append(class_id_list.index(classdir))
                    except(OSError, IOError):
                        print("Cannot open file "+img_path)
                    cnt=cnt+1
        elif mode == "test":
            json_path = os.path.join(data_path, "test.json")
            self.names=[]
            with open(json_path, "r") as f:
                json_data = json.load(f)
                data_path = os.path.join(data_path, "Test")
                cnt=0
                for img_name in tqdm.tqdm(os.listdir(data_path)):
                    img_path = os.path.join(data_path, img_name)
                    try:
                        img = self.test_transform(img_path)
                        self.data.append(img)
                        self.labels.append(
                            class_id_list.index(json_data[img_name]))
                        self.names.append(img_name)
                    except(OSError, IOError):
                        print("Cannot open file "+img_path)
                    cnt=cnt+1
                    if cnt==self.maxdata:
                        break

    def __getitem__(self, idx):
        if self.mode=="train":
            return self.data[idx],self.labels[idx]
        elif self.mode=="val":
            return self.data[idx], self.labels[idx]
        else:
            return self.data[idx], self.names[idx]
    def __len__(self):
        return len(self.data)
