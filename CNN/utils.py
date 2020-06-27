import torch
import numpy as np 
import random
import torch.backends.cudnn as cudnn
import sys
import os

class Logger(object):
    def __init__(self,filename="result.txt"):
        self.terminal=sys.stdout
        self.log=open(filename,"a")
    def write(self,message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass

class counter():
    def __init__(self):
        self.value = 0
        self.ave = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.value = val
        self.sum += val * n
        self.count += n
        self.ave = self.sum / self.count


def topk_accuracy(output, labels, k):
    batch_size = labels.size(0)
    _, pred = output.topk(k, 1, True, True)
    pred = pred.t()
    judge = pred.eq(labels.view(1, -1).expand_as(pred))
    acc_top_k = judge[:k].view(-1).float().sum(0)*100.0/batch_size
    return acc_top_k

def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if seed == 0:
        cudnn.deterministic = True
        cudnn.benchmark = False
