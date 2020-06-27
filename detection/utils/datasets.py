import glob
import math
import os
import random
import shutil
import time
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import torch
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm
import PIL.Image as Image
from torchvision import transforms
from utils.utils import *
import json

def preprocess(img_path,inp_dim=416):
    # input an image_path and turn it into a inp_dim*inp*dim shape normalized tensor
    # return the transformed image and the width and height scale
    # Only consider the case that the input has same height and width
    img=Image.open(img_path).convert('RGB')
    h,w=img.size
    resize=transforms.Resize((inp_dim,inp_dim))
    norm=transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    transform=transforms.Compose([
        resize,
        transforms.ToTensor(),
        norm
    ])
    img=transform(img)
    return img,inp_dim/h


label_names=['pl30', 'pl5', 'w57', 'io', 'p5', 'ip', 'i2', 'i5', 'i4', 'pl80', 'p11', 'pl60', 'p26', 'pl40', 'p23', 'pl50', 'pn', 'pne', 'po']
class TrafficDataset(Dataset):
    datalen=3690
    val_id=random.sample(range(datalen),64)
    def __init__(self,inp_dim=416,data_dir="../image_exp/Detection/",mode='train',image_weights=False,maxdata=np.inf):
        super(TrafficDataset,self).__init__()
        self.data_dir=data_dir
        self.mode=mode
        self.inp_dim=inp_dim
        self.image_weights=image_weights
        annotationfile=os.path.join(self.data_dir,"train_annotations.json")
        # with open(annotationfile,"r") as f:
        #     jsonfile=json.load(f)
        #     self.label_names=jsonfile["types"]  # all labels (map: id->str)
        self.label_names=label_names
        self.images=[]
        self.labels=[]
        self.scales=[]
        self.file_paths=[]
        self.origin_size=2048
        if mode=="train" or mode=="val":
            with open(annotationfile,"r") as f:
                jsonfile=json.load(f)
                imgdata=jsonfile['imgs']
                cnt=0
                for imgid in tqdm(imgdata):
                    if(cnt==maxdata):
                        break
                    if (cnt not in TrafficDataset.val_id)^ (mode=="train"):
                        cnt=cnt+1
                        continue
                    cnt=cnt+1
                    img=imgdata[imgid]
                    imgpath=os.path.join(data_dir,img["path"])
                    objects=img["objects"]
                    processed_img,scale=preprocess(imgpath,inp_dim=self.inp_dim)
                    self.images.append(processed_img)
                    self.scales.append(scale)
                    self.file_paths.append(imgpath)
                    label=[]
                    emptylabel=1
                    for obj in objects:
                        labelid=self.label_names.index(obj["category"])
                        bbox=obj["bbox"]
                        xmin=bbox["xmin"]*scale
                        xmax=bbox["xmax"]*scale
                        ymin=bbox["ymin"]*scale
                        ymax=bbox["ymax"]*scale
                        if emptylabel:
                            label=torch.FloatTensor([[0,labelid,(xmin+xmax)/2/self.inp_dim,(ymin+ymax)/2/self.inp_dim,(xmax-xmin)/self.inp_dim,(ymax-ymin)/self.inp_dim]])
                            emptylabel=0
                        else:
                            label=torch.cat((label,torch.FloatTensor([[0,labelid,(xmin+xmax)/2/self.inp_dim,(ymin+ymax)/2/self.inp_dim,(xmax-xmin)/self.inp_dim,(ymax-ymin)/self.inp_dim]])),dim=0)
                    self.labels.append(label)
        else:
            for _,_, filenames in os.walk(os.path.join(data_dir,'test')):
                cnt=0
                for filename in tqdm(filenames):
                    if(cnt==maxdata):
                        break
                    cnt=cnt+1
                    imgpath=os.path.join(data_dir,'test',filename)
                    processed_img,scale=preprocess(imgpath,inp_dim=self.inp_dim)
                    self.file_paths.append(imgpath)
                    self.images.append(processed_img)
                    self.scales.append(scale)

    def __len__(self):
        return len(self.images)

    def __getitem__(self,i):
        if self.mode=='train' or self.mode=="val":
            return self.images[i],self.labels[i],self.file_paths[i],[(self.origin_size,self.origin_size),(self.inp_dim/self.origin_size,self.inp_dim/self.origin_size,0)]
        else:
            return self.images[i],None,self.file_paths[i],[(self.origin_size,self.origin_size),(self.inp_dim/self.origin_size,self.inp_dim/self.origin_size,0)]

    def collate_fn(self,batch):
        imgs,targets,files,scales=list(zip(*batch))
        if self.mode=="test":
            return torch.stack(imgs,0),None,files,scales
        else:
            targets=[boxes for boxes in targets if boxes is not None]
            for i,boxes in enumerate(targets):
                boxes[:,0]=i
            targets=torch.cat(targets,0)
            return torch.stack(imgs,0),targets,files,scales





