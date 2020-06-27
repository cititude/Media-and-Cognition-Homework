import os
import argparse
import tqdm
import json
import re
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tensorboardX import SummaryWriter
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import math

from models import *
from datasets import *
from utils import *



def Variable_(tensor, *args_, **kwargs):
    '''
    Make variable cuda depending on the arguments
    '''
    # Unroll list or tuple
    if type(tensor) in (list, tuple):
        return [Variable_(t, *args_, **kwargs) for t in tensor]
    # Unroll dictionary
    if isinstance(tensor, dict):
        return {key: Variable_(v, *args_, **kwargs) for key, v in list(tensor.items())}
    # Normal tensor
    variable = Variable(tensor, *args_, **kwargs)
    if args.cuda:
        variable = variable.cuda()
    return variable


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", default="../image_exp/Classification/Data")
parser.add_argument("--batch_size", default=32)
parser.add_argument("--record_file", default="result.txt")
parser.add_argument("--optimizer", default="SGD", type=str)
parser.add_argument("--lr", default=0.1, type=float)   # now not used
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument('--weight_decay', '--wd', default=1e-4,
                    type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument("--print_freq", default=20, type=int)
parser.add_argument("--model_name", default="resnet")
parser.add_argument("--device",default="0" if torch.cuda.is_available() else "cpu")
parser.add_argument("--pred",action="store_true",help="pred and save the result in pred.json")
parser.add_argument("--weights_path",default="./sedense.pt")
parser.add_argument("--record_dir",default="./results")
parser.add_argument("--epochs",default=300,type=int)
parser.add_argument("--meta_lr",default=0.1,type=float)
parser.add_argument("--only_test",action="store_true",help="test on testdata(without training)")
parser.add_argument("--use_mixup",action="store_true")
parser.add_argument("--fix_seed",action="store_true")
parser.add_argument("--meta_batch",default=1,type=int)
parser.add_argument("--finetune",action="store_true")
parser.add_argument("--iterations",type=int,default=30)

args = parser.parse_args()
record_file = args.record_file
batch_size = args.batch_size
weights_path=args.weights_path
pred=args.pred
record_dir=args.record_dir
device=args.device
lr=args.lr

epochs=args.epochs
iterations=args.iterations
meta_lr=args.meta_lr
device=select_device(device)

def train_model(model,traindataset,optimizer):

    # lr_scheduler
    start_iter=0
    lf = lambda x: (((1 + math.cos(x * math.pi / iterations)) / 2) ** 1.0) * 0.95 + 0.05  # cosine lr schedule
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = start_iter-1  
    for iteration in tqdm(range(iterations)):
        start_iter=start_iter+1
        model.train()
        for idx,(imgs,labels) in enumerate(traindataset):
            imgs=Variable(imgs,requires_grad=False).to(device)
            labels=labels.to(device)
            if args.use_mixup and iteration+5<iterations:
                mixed_x, y_a, y_b, lam=mixup_data(imgs,labels)
                pred=model(mixed_x)
                loss=nn.CrossEntropyLoss()(pred,y_a)*lam+nn.CrossEntropyLoss()(pred,y_b)*(1-lam)
            else:
                pred=model(imgs)
                loss=nn.CrossEntropyLoss()(pred,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
    return loss.item()

def val_model(model,valdataset):
    losses=counter()
    accs=counter()
    torch.cuda.empty_cache()
    model.eval()
    with torch.no_grad():
        for idx,(imgs,labels) in enumerate(valdataset):
            imgs=Variable(imgs,requires_grad=False).to(device)
            labels=labels.to(device)
            pred=model(imgs)
            loss=nn.CrossEntropyLoss()(pred,labels)    
            _,pred_cls=pred.max(1)
            acc=(pred_cls==labels).float().mean()
            losses.update(loss)
            accs.update(acc)
    return losses,accs


def test_model(model,pred_file="pred.json"):
    testdata=TestDataset()
    pred_dict={}
    torch.cuda.empty_cache()
    model.eval()
    with torch.no_grad():
        for idx in range(len(testdata)):
            img,img_name=testdata[idx]
            img=Variable(img,requires_grad=False).to(device)
            pred=model(img)
            _,pred=pred.max(1)
            pred_dict[img_name]=testclass_id_list[pred]
        json.dump(pred_dict,open(pred_file,"w"))
    return None

def train_meta_model(meta_model,dataset,test_data,meta_optimizer):
    fewshotdata=FewShot(dataset)
    last_loss=np.inf
    sys.stdout=Logger(os.path.join(record_dir,"result.txt"))

    # lr_scheduler
    start_iter=0
    lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.95 + 0.05  # cosine lr schedule
    meta_scheduler = lr_scheduler.LambdaLR(meta_optimizer, lr_lambda=lf)
    meta_scheduler.last_epoch = start_iter-1  
    
    # batch_accumulation
    batch_id=1

    for epoch in tqdm(range(epochs)):
        meta_model.train()
        traindataset,valdataset=next(fewshotdata)
        traindataloader=DataLoader(traindataset,batch_size=batch_size,shuffle=True)
        valdataloader=DataLoader(valdataset,batch_size=batch_size,shuffle=False)
        cur_model=meta_model.clone()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,cur_model.parameters()), lr=lr, betas=(0, 0.999))
        train_model(model=cur_model,traindataset=traindataloader,optimizer=optimizer)
        loss,acc=val_model(cur_model,valdataloader)
        string = ('Epoch: [{0}]\t'
                    'Losses/Loss {Losses.value:.4f} '
                    '({Losses.ave:.4f})\t'
                    'Prec@1 {top1.value:.3f} ({top1.ave:.3f})\t'.format(
                        epoch,  top1=acc, Losses=loss))
        print(string)     
        model=meta_model.clone()
        model.zero_grad()
        model_optimizer=torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, betas=(0, 0.999))

        meta_model.point_grad_to(cur_model)

        if batch_id%args.meta_batch==0:
            meta_optimizer.step()
            meta_optimizer.zero_grad()
            batch_id=0      
        
        meta_scheduler.step()  
        batch_id=batch_id+1
        
        last_loss=loss.ave

        sys.stdout.flush()

        if epoch%10==0 or epoch+1==epochs:
            model=meta_model.clone()
            chkpt = {'epoch': epoch,'model': model.module.state_dict() if hasattr(model, 'module') else model.state_dict()}
            torch.save(chkpt, os.path.join(record_dir,"model_{}.pt".format(epoch)))

def main():
    if args.fix_seed:
        init_seeds()
    if not os.path.exists(record_dir):
        os.mkdir(record_dir)
    test_data=Traffic_MetaDataset(data_path="../image_exp/Classification/DataFewShot",mode="test")
    test_data,_=next(FewShot(test_data,N=11,support=1,query=0))
    test_data=DataLoader(test_data,batch_size=batch_size)
    meta_model=Meta_Model(num_classes=11).to(device)
    meta_model.load_pretrained_state_dict(torch.load(weights_path,map_location=device)["model"])
    meta_optimizer=optim.SGD(filter(lambda p: p.requires_grad, meta_model.parameters()),lr=meta_lr)
    dict_name = list(meta_model.state_dict())

    # show model details
    # for i, p in enumerate(dict_name):
    #     print(i, p)

    # fix pretrained parameters in first 3 denseblocks
    # for i,p in enumerate(meta_model.parameters()):
    #     if i <=549:
    #         p.requires_grad = False
    if args.only_test:
        meta_model.load_state_dict(torch.load(weights_path,map_location=device)["model"])
        test_model(meta_model)
        return

    if args.finetune:
        meta_model.train()
        model=meta_model.clone()
        model.load_state_dict(torch.load(weights_path,map_location=device)["model"])
        optimizer=torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, betas=(0, 0.999))
        train_model(model,test_data,optimizer)
        chkpt = {'epoch': -1,'model': model.module.state_dict() if hasattr(model, 'module') else model.state_dict()}
        torch.save(chkpt,"model.pt")
        test_model(model)
        return

    traffic_data=Traffic_MetaDataset(data_path=args.data_path)
    train_meta_model(meta_model=meta_model,dataset=traffic_data,test_data=test_data,meta_optimizer=meta_optimizer)
    cur_model=meta_model.clone()
    optimizer=torch.optim.Adam(filter(lambda p: p.requires_grad,cur_model.parameters()), lr=lr, betas=(0, 0.999))
    train_model(cur_model,test_data,optimizer)
    test_model(cur_model.clone())



if __name__=="__main__":
    main()