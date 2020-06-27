import torch
from torch import nn
from torch.autograd import Variable
from sedense import *

class Reptile(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)

    def point_grad_to(self, target):
        '''
        Set .grad attribute of each parameter to be proportional
        to the difference between self and target
        '''
        for p, target_p in zip(self.parameters(), target.parameters()):
            if p.grad is None:
                if self.is_cuda():
                    p.grad = Variable(torch.zeros(p.size())).cuda()
                else:
                    p.grad = Variable(torch.zeros(p.size()))
            p.grad.data.add_(p.data - target_p.data)

    def is_cuda(self):
        return next(self.parameters()).is_cuda


class Meta_Model(Reptile):
    def __init__(self, num_classes=11):
        Reptile.__init__(self)

        self.num_classes = num_classes

        self.classifier=SEDenseNet(num_classes=num_classes)

    def forward(self, x):
        out = x.view(-1, 3, 224, 224)
        out = self.classifier(out)
        return out

    def predict(self, prob):
        __, argmax = prob.max(1)
        return argmax

    def clone(self):
        clone = Meta_Model(self.num_classes)
        clone.load_state_dict(self.state_dict())
        if self.is_cuda():
            clone.cuda()
        return clone

    def load_pretrained_state_dict(self,pretrained_state_dict):
        self_state_dict=self.classifier.state_dict()
        pretrained_state_dict={k:v for k,v in pretrained_state_dict.items() if (k in self_state_dict and "classifier" not in k)}
        self_state_dict.update(pretrained_state_dict)
        self.classifier.load_state_dict(self_state_dict)
