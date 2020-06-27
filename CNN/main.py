import torch
import torch.nn as nn
import torchvision
import argparse
import numpy as np
import torch.optim as optim
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from datasets import *
from utils import *
import random
from tqdm import tqdm
from models import *
import os
from tensorboardX import SummaryWriter
from terminaltables import  AsciiTable
import torch.optim.lr_scheduler as lr_scheduler
import math
from cam import *


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", default="../image_exp/Classification/Data")
parser.add_argument("--batch_size", default=32)
parser.add_argument("--record_file", default="result.txt")
parser.add_argument("--optimizer", default="SGD", type=str)
parser.add_argument("--lr", default=0.1, type=float)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument('--weight_decay', '--wd', default=1e-4,
                    type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument("--print_freq", default=20, type=int)
parser.add_argument("--model_name", default="resnet")
parser.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--weights_path",default=None)
parser.add_argument("--record_dir",default="./result")
parser.add_argument("--epochs",default=300,type=int)
parser.add_argument("--only_test",action="store_true",help="only test the current model")
parser.add_argument("--fix_seed",action="store_true",help='fix all seed to zero')
parser.add_argument("--cam",action="store_true",help="use cam to visualize")
parser.add_argument("--maxdata",default=99999,type=int)
parser.add_argument("--pred_file",default="result/pred.json")

args = parser.parse_args()
batch_size = args.batch_size
weights_path=args.weights_path
device=args.device if torch.cuda.is_available() else "cpu"
record_dir=args.record_dir
record_file = os.path.join(record_dir,args.record_file)
data_path=args.data_path

tb_writer=None

def train(trainloader, testloader, model, optimizer, batch_size=batch_size, epochs=300, record_file=record_file):

    # lr_scheduler
    start_epoch=0
    lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.95 + 0.05  # cosine lr schedule
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = start_epoch-1  
    
    train_batches_num = len(trainloader)
    sys.stdout=Logger(record_file)
    for epoch in range(epochs):
        start_epoch=start_epoch+1
        acc1 = counter()
        Losses = counter()
        for i, (data, label) in tqdm(enumerate(trainloader)):
            model.train()
            data = data.to(device)
            label = label.to(device)
            classification_result = model(data)
            loss = 0
            acc1.update(topk_accuracy(
                classification_result, label, 1), label.size(0))
            loss_temp = CELoss(
                classification_result, label)
            Losses.update(loss_temp.data.item(), label.size(0))
            loss += loss_temp
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % args.print_freq == 0:
                string = ('Epoch: [{0}][{1}/{2}]\t'
                          'Losses/Loss {Losses.value:.4f} '
                          '({Losses.ave:.4f})\t'
                          'Prec@1 {top1.value:.3f} ({top1.ave:.3f})\t'.format(
                              epoch, i+1, train_batches_num, top1=acc1, Losses=Losses))
                print(string)
        
        scheduler.step()
        string = ('Epoch: [{0}]\t'
                  'Losses/Loss {Losses.value:.4f} '
                  '({Losses.ave:.4f})\t'
                  'Prec@1 {top1.value:.3f} ({top1.ave:.3f})\t'.format(
                      epoch,  top1=acc1, Losses=Losses))
        print(string)
        with torch.no_grad():
            val_loss,val_acc1=validate(testloader, model)
            chkpt = {'epoch': epoch,'model': model.module.state_dict() if hasattr(model, 'module') else model.state_dict()}
            torch.save(chkpt, os.path.join(record_dir,"model_{}.pt".format(epoch)))

        # Tensorboard
        if tb_writer:
            tags = ['train/loss', 'train/acc1', 'val/loss','val/acc1']
            values=[Losses.ave,acc1.ave,val_loss,val_acc1]
            for x, tag in zip(values, tags):
                tb_writer.add_scalar(tag, x, epoch)

    tb_writer.close()
    
def validate(testloader, model, batch_size=batch_size, record_file=record_file):
    acc1 = counter()
    test_batches_num = len(testloader)
    Losses = counter()
    model.eval()
    for i, (data, label) in enumerate(testloader):
        data = data.to(device)
        label = label.to(device)
        classification_result = model(data)
        loss = 0
        acc1.update(topk_accuracy(
            classification_result, label, 1), label.size(0))
        loss_temp = CELoss(
            classification_result, label)
        Losses.update(loss_temp.data.item(), label.size(0))
        loss += loss_temp
        if (i+1) % args.print_freq == 0:
            string = ('Epoch: [0][{0}/{1}]\t'
                      'Losses/Loss {Losses.value:.4f} '
                      '({Losses.ave:.4f})\t'
                      'Prec@1 {top1.value:.3f} ({top1.ave:.3f})\t'.format(
                          i+1, test_batches_num, top1=acc1, Losses=Losses))
            print(string)
    string = ('TEST:\t'
              'Losses/Loss {Losses.value:.4f} '
              '({Losses.ave:.4f})\t'
              'Prec@1 {top1.value:.3f} ({top1.ave:.3f})\t'.format(
                  top1=acc1, Losses=Losses))
    print(string)
    return Losses.ave,acc1.ave


# run on the test dataset and save the results in pred_file
def test(testloader,model,pred_file="./pred.json"):
    test_batches_num = len(testloader)
    model.eval()
    result={}
    for i, (data, names) in tqdm(enumerate(testloader),desc="predicting"):
        data = data.to(device)
        classification_result = model(data)
        _,top1=classification_result.topk(1,dim=1,largest=True)
        for j in range(len(top1)):
            result[names[j]]=class_id_list[top1[j].item()]
    with open(pred_file,"w") as f:
        json.dump(result,fp=f)

def main():
    print(AsciiTable([[key,vars(args)[key]] for key in vars(args)]).table)
    if args.fix_seed:
        init_seeds()
    global record_file
    if not os.path.exists(record_dir):
        os.mkdir(record_dir)
    global CELoss
    CELoss = nn.CrossEntropyLoss().to(device)
    torch.backends.cudnn.benchmark = True
    if args.model_name == "resnet":
        classifier = models.resnet101(num_classes=19, pretrained=False).to(device)
    elif args.model_name == "densenet":
        classifier = models.densenet121(
            num_classes=19, pretrained=False).to(device)
    else:
        classifier=SEDenseNet(growth_rate=32,num_classes=19).to(device)
    if weights_path is not None:
        chkpt=torch.load(weights_path, map_location=device)
        classifier.load_state_dict(chkpt["model"])


    if args.cam:
        cam=Cam(model=classifier)
        test_data = TrafficDataset(data_path=args.data_path, mode="test",maxdata=args.maxdata)
        test_dataloader=DataLoader(test_data,batch_size=1,shuffle=False)
        classifier.eval()
        for i,(img,name) in tqdm(enumerate(test_dataloader)):
            img=img.to(device)
            cam.forward_pass(img,name)
        return
    # save the test result in pred.json
    if args.only_test:    
        test_data = TrafficDataset(data_path=args.data_path, mode="test",maxdata=args.maxdata)
        test_dataloader=DataLoader(test_data,batch_size=batch_size,shuffle=False)
        test(test_dataloader,classifier,pred_file=os.path.join(record_dir,args.pred_file))
        return

    train_data = TrafficDataset(data_path=args.data_path, mode="train",maxdata=args.maxdata)

    val_data=TrafficDataset(data_path=args.data_path, mode="val",maxdata=args.maxdata)

    train_dataloader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False)

    if args.optimizer == 'Adam':
        optimizer = optim.Adam([{'params': filter(lambda p: p.requires_grad, classifier.parameters())}, ],
                               lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay, amsgrad=False)
    else:
        optimizer = optim.SGD([{'params': filter(lambda p: p.requires_grad, classifier.parameters())}, ],
                              lr=args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay,
                              nesterov=False)
    tb_writer = SummaryWriter(os.path.join(record_dir,"runs"))
    train(trainloader=train_dataloader, testloader=val_dataloader,
          model=classifier, optimizer=optimizer)


if __name__ == "__main__":
    main()
