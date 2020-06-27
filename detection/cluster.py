import sys
import numpy as np 
import scipy
import json
from scipy.cluster.vq import vq,kmeans,whiten
import matplotlib.pyplot as plt

def get_all_boxes():
    annotationfile="../image_exp/Detection/train_annotations.json"
    boxes=[]
    with open(annotationfile,"r") as f:
        jsonfile=json.load(f)
        imgdata=jsonfile['imgs']
        for imgid in imgdata:
            img=imgdata[imgid]
            objects=img["objects"]
            for obj in objects:
                bbox=obj["bbox"]
                xmin=float(bbox["xmin"])
                xmax=float(bbox["xmax"])
                ymin=float(bbox["ymin"])
                ymax=float(bbox["ymax"])
                boxes.append((xmax-xmin,ymax-ymin))
    return boxes

boxes=get_all_boxes()
print(len(boxes))
centroid=kmeans(boxes,9)[0]
centroid=[list(x) for x in centroid]
dist=[x**2+y**2 for (x,y) in centroid]
order=np.array(dist).argsort()
centroid=np.array(centroid)[order]*1216/2048
print(centroid,file=open("./bbox_prior.txt","w"))
str0=" ,".join([str(int(x))+","+str(int(y)) for (x,y) in centroid])
print(str0,file=open("./bbox_prior.txt","a"))