from torch.utils import data
import os
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random
import torch
import json

class_id_list = list()
for classdir in os.listdir(os.path.join("../image_exp/Classification/Data", "Train")):
    class_id_list.append(classdir)
testclass_id_list=list()
for classdir in os.listdir("../image_exp/Classification/DataFewShot/Train"):
    testclass_id_list.append(classdir)
class Traffic_MetaDataset(Dataset):
    def __init__(self, data_path='../image_exp/Classification/Data', mode="train", resize=224):
        super(Traffic_MetaDataset, self).__init__()
        self.data_path = data_path
        self.mode = mode
        self.resize = resize
        self.data = []
        self.labels = []
        self.names=[]
        self.train_transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                                   transforms.Resize(
            (self.resize, self.resize)),
            transforms.ToTensor(),
            transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        if mode == "train":
            data_path = os.path.join(data_path, "Train")
            for classdir in tqdm(os.listdir(data_path)):
                self.labels.append(class_id_list.index(classdir))
                class_data_path = os.path.join(data_path, classdir)
                data=[]
                for img_name in os.listdir(class_data_path):
                    img_path = os.path.join(class_data_path, img_name)
                    try:
                        img = self.train_transform(img_path)
                        data.append(img)
                    except(OSError, IOError):
                        print("Cannot open file "+img_path)
                self.data.append(data)
        elif mode == "test":
            json_path = os.path.join(data_path,"train.json")
            self.names=[]
            with open(json_path, "r") as f:
                json_data = json.load(f)
                data_path = os.path.join(data_path, "Train")
                for classdir in tqdm(os.listdir(data_path)):
                    class_data_path=os.path.join(data_path,classdir)
                    data=[]
                    for img_name in tqdm(os.listdir(class_data_path)):
                        img_path = os.path.join(class_data_path, img_name)
                        try:
                            img = self.train_transform(img_path)
                            data.append(img)
                            self.labels.append(
                                testclass_id_list.index(json_data[img_name]))
                            self.names.append(img_name)
                        except(OSError, IOError):
                            print("Cannot open file "+img_path)
                    self.data.append(data)
    def __getitem__(self, idx):
        return self.data[idx],self.labels[idx]
    def __len__(self):
        return len(self.data)



class TrafficDataset(Dataset):
    def __init__(self,data,label):
        super(TrafficDataset,self).__init__()
        self.data=data
        self.label=label
    def __getitem__(self,idx):
        return self.data[idx],self.label[idx]
    def __len__(self):
        return len(self.data)

class TestDataset(Dataset):
    def __init__(self, data_path='../image_exp/Classification/DataFewShot/Test', resize=224):
        super(TestDataset, self).__init__()
        self.data_path = data_path
        self.resize = resize
        self.data = []
        self.img_names=[]
        self.train_transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                                   transforms.Resize(
            (self.resize, self.resize)),
            transforms.ToTensor(),
            transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        for img_name in os.listdir(data_path):
            img_path = os.path.join(data_path, img_name)
            try:
                img = self.train_transform(img_path)
                self.data.append(img)
                self.img_names.append(img_name)
            except(OSError, IOError):
                print("Cannot open file "+img_path)
    def __getitem__(self,idx):
        return self.data[idx],self.img_names[idx]
    def __len__(self):
        return len(self.data)


def FewShot(dataset,N=11,support=1,query=5):
    while True:
        idxs=random.sample(range(len(dataset)),N)
        data=[dataset.data[i] for i in range(len(dataset)) if i in idxs ]
        label=[dataset.labels[i] for i in range(len(dataset)) if i in idxs ]
        traindata=[]
        trainlabel=[]
        valdata=[]
        vallabel=[]
        for j in range(N):
            use_idxs=random.sample(range(len(data[j])),support+query)
            tdj=[data[j][k] for k in range(len(data[j])) if k in use_idxs[:support] ]
            vdj=[data[j][k]  for k in range(len(data[j])) if k in use_idxs[support:]]
            tlj=torch.zeros(support).long()
            vlj=torch.zeros(query).long()
            tlj[:]=j
            vlj[:]=j
            traindata.extend(tdj)
            trainlabel.extend(tlj)
            valdata.extend(vdj)
            vallabel.extend(vlj)
        yield TrafficDataset(traindata,trainlabel),TrafficDataset(valdata,vallabel)

