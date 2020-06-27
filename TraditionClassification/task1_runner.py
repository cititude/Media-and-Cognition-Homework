# -*- coding=utf-8 -*-
import glob
import platform
import time
from PIL import Image
from skimage.feature import hog
import numpy as np
import os
import joblib
from sklearn.svm import LinearSVC
import shutil
import sys
import re
import cv2
from PIL import ImageStat
from PIL import ImageEnhance
from skimage.feature import local_binary_pattern
import warnings
warnings.filterwarnings("ignore", category=Warning)
label_map =  {'i2': 'i2',
             'i4': 'i4',
             'i5': 'i5',
             'io': 'io',
             'ip': 'ip',
             'p11': 'p11',
             'p23': 'p23',
             'p26': 'p26',
             'p5': 'p5',
             'pl30': 'pl30',
             'pl40': 'pl40',
             'pl5': 'pl5',
             'pl50': 'pl50',
             'pl60': 'pl60',
             'pl80': 'pl80',
             'pn': 'pn',
             'pne': 'pne',
             'po': 'po',
             'w57': 'w57',
             }
# 训练集图片的位置
train_image_path = r'C:\Users\tybus\Downloads\Classification\Data\Train'
# 测试集图片的位置
test_image_path = r'C:\Users\tybus\Downloads\Classification\Data\Test'
# 训练集标签的位置
train_label_path = os.path.join('train.json')
# 测试集标签的位置
test_label_path = os.path.join('test.json')


temp_pathtrain=os.path.join('temptrain.txt')
temp_pathtest=os.path.join('temptest.txt')
image_height = 50
image_width = 50

train_feat_path = 'trainfeat/'
test_feat_path = 'testfeat/'
model_path = 'model/'

def merge(path):
    for classdir in os.listdir(path):
        datapath = os.path.join(path, classdir)
        for file in os.listdir(datapath):
            shutil.move(datapath+'/'+file,path)
        shutil.rmtree(datapath) 
def divide(dir):
    files = os.listdir(dir)
    files.sort()
    for file in files:
       label = file.split('_')
       name=label[0]
       if not os.path.exists(dir+'/'+name):
          os.mkdir(dir+'/'+name)
       shutil.move(dir+'/'+file,dir+'/'+name)
def brightA(im):
    x=im.size[0]
    y=im.size[1] 
    crop1 = im.crop((0, 0, int(0.2*x), int(0.2*y)))  
    crop2 = im.crop((int(0.3*x), int(0.3*y), int(0.7*x),int(0.7*y)))  
    crop3=  im.crop((int(0.8*x), int(0.8*y), x,y))      
    stat1 = ImageStat.Stat(crop1)
    stat2=  ImageStat.Stat(crop2)
    stat3= ImageStat.Stat(crop3)
    if stat1.rms[0]>180 or stat3.rms[0]>180:
        return stat2.rms[0]
    else:
        return 0.15*stat1.rms[0]+0.7*stat2.rms[0]+0.15*stat3.rms[0]

def get_image_list(filePath, nameList):
    print('read image from ',filePath)
    img_list = []
    for name in nameList:
        temp = Image.open(os.path.join(filePath,name)) 
        temp=temp.resize((image_width,image_height),Image.ANTIALIAS)
        a=brightA(temp)
        if(a<100):
            temp=ImageEnhance.Brightness(temp).enhance(-0.0375*a+4.2)
            temp=ImageEnhance.Contrast(temp).enhance(2.8)       
        temp = cv2.cvtColor(np.asarray(temp),cv2.COLOR_RGB2BGR)  
        img_list.append(temp.copy())
    return img_list


def change(filepath1,filepath2):
    f1 = open(filepath1,'r+')
    f2 = open(filepath2,'w+')
    ss=f1.readline()
    ss=ss.replace('": "',' ')
    ss=ss.replace('", "','\n')
    ss=ss.replace('{"','')
    ss=ss.replace('"}','')
    f2.write(ss)
    f1.close()
    f2.close()



def get_feat(image_list, name_list, label_list, savePath):
    i = 0
    for image in image_list:
        try:
            image = np.reshape(image, (image_width, image_height, 3))
        except:
            print('size error：',name_list[i])
            continue

        gray = rgb2gray(image) / 255.0
        fd = hog(gray, orientations=14,block_norm='L1', pixels_per_cell=[6, 6], cells_per_block=[3, 3], visualize=False,
                 transform_sqrt=True)
        temp1= hog(gray, orientations=14,block_norm='L1', pixels_per_cell=[12, 12], cells_per_block=[3, 3], visualize=False,
                 transform_sqrt=True)
        temp2= hog(gray, orientations=14,block_norm='L1', pixels_per_cell=[18, 18], cells_per_block=[3, 3], visualize=False,
                 transform_sqrt=True)
        lbp=local_binary_pattern(gray, 8, 1, "uniform")
        max_bins = int(lbp.max() + 1)
        temp3, _ = np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins))
        fd=np.append(fd,temp1)
        fd=np.append(fd,temp2)
        fd=np.append(fd,temp3)
        fd = np.concatenate((fd, [label_list[i]]))
        fd_name = name_list[i] + '.feat'
        fd_path = os.path.join(savePath, fd_name)
        joblib.dump(fd, fd_path)
        i += 1
    print("Test features are extracted and saved.")


def rgb2gray(im):
    gray = im[:, :, 0] * 0.2989 + im[:, :, 1] * 0.5870 + im[:, :, 2] * 0.1140
    return gray


def get_name_label(file_path):
    print("read label from ",file_path)
    name_list = []
    label_list = []
    with open(file_path) as f:
        for line in f.readlines():
            line.replace('{"','')
            line.replace('"}','')
            if len(line)>=3: 
                name_list.append(line.split(' ')[0])
                label_list.append(line.split(' ')[1].replace('\n','').replace('\r',''))
    return name_list, label_list


def extra_feat():
    merge(train_image_path)
    change(train_label_path,temp_pathtrain)
    change(test_label_path,temp_pathtest)
    train_name, train_label = get_name_label(temp_pathtrain)
    test_name, test_label = get_name_label(temp_pathtest)   
    train_image = get_image_list(train_image_path, train_name)
    test_image = get_image_list(test_image_path, test_name)
    get_feat(train_image, train_name, train_label, train_feat_path)
    get_feat(test_image, test_name, test_label, test_feat_path)
    divide(train_image_path)


def mkdir():
    if not os.path.exists(train_feat_path):
        os.mkdir(train_feat_path)
    if not os.path.exists(test_feat_path):
        os.mkdir(test_feat_path)


def train_and_test():
    t0 = time.time()
    features = []
    labels = []
    correct_number = 0
    total = 0
    for feat_path in glob.glob(os.path.join(train_feat_path, '*.feat')):
        data = joblib.load(feat_path)
        features.append(data[:-1])
        labels.append(data[-1])
    print("Training a Linear LinearSVM Classifier.")
    clf = LinearSVC(tol=0.1,class_weight='balanced')
    clf.fit(features, labels)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    joblib.dump(clf, model_path + 'model')
    #clf = joblib.load(model_path+'model')
    # exit()
    result_list = []
    result_list.append("{")
    for feat_path in glob.glob(os.path.join(test_feat_path, '*.feat')):
        total += 1
        t=len(glob.glob(os.path.join(test_feat_path, '*.feat')))
        if platform.system() == 'Windows':
            symbol = '\\'
        else:
            symbol = '/'
        image_name = feat_path.split(symbol)[1].split('.feat')[0]
        data_test = joblib.load(feat_path)
        data_test_feat = data_test[:-1].reshape((1, -1)).astype(np.float64)
        result = clf.predict(data_test_feat)
        if total<t:
            result_list.append('"'+image_name +'": ' + '"'+label_map[(result[0])] +'"'+ ', ')
        if total==t:
            result_list.append('"'+image_name +'":' + '"'+label_map[(result[0])] +'"')
        if (result[0]) == (data_test[-1]):
            correct_number += 1
    result_list.append('}')
    write_to_txt(result_list)
    rate = float(correct_number) / total
    t1 = time.time()
    print('accuracy： %f' % rate)
    print('time : %f' % (t1 - t0))

def write_to_txt(list):
    with open('pred.json', 'w') as f:
        f.writelines(list)
    print('results are saved in pred.json')

def mainA():   #训练+测试
    mkdir()  

    shutil.rmtree(train_feat_path)
    shutil.rmtree(test_feat_path)
    mkdir()
    extra_feat()  
    train_and_test()  
    shutil.rmtree(train_feat_path)
    shutil.rmtree(test_feat_path)
    os.remove('./temptrain.txt')
    os.remove('./temptest.txt')

def mainB():  #加载模型直接测试（确保本地有model文件夹且存放有model）
    mkdir() 
    shutil.rmtree(test_feat_path)
    shutil.rmtree(train_feat_path)
    mkdir()
    change(test_label_path,temp_pathtest)
    test_name, test_label = get_name_label(temp_pathtest)
    test_image = get_image_list(test_image_path, test_name)
    get_feat(test_image, test_name, test_label, test_feat_path)
    clf = joblib.load(model_path+'model')
    t0 = time.time()
    correct_number = 0
    total = 0
    result_list = []
    result_list.append("{")
    for feat_path in glob.glob(os.path.join(test_feat_path, '*.feat')):
        total += 1
        t=len(glob.glob(os.path.join(test_feat_path, '*.feat')))
        if platform.system() == 'Windows':
            symbol = '\\'
        else:
            symbol = '/'
        image_name = feat_path.split(symbol)[1].split('.feat')[0]
        data_test = joblib.load(feat_path)
        data_test_feat = data_test[:-1].reshape((1, -1)).astype(np.float64)
        result = clf.predict(data_test_feat)
        if total<t:
            result_list.append('"'+image_name +'": ' + '"'+label_map[(result[0])] +'"'+ ', ')
        if total==t:
            result_list.append('"'+image_name +'":' + '"'+label_map[(result[0])] +'"')
        if (result[0]) == (data_test[-1]):
            correct_number += 1
    result_list.append('}')
    write_to_txt(result_list)
    rate = float(correct_number) / total
    t1 = time.time()
    print('accuracy： %f' % rate)
    print('time : %f' % (t1 - t0))
    shutil.rmtree(test_feat_path)
    shutil.rmtree(train_feat_path)
    os.remove('./temptest.txt')

mainB()   #选择执行方式

