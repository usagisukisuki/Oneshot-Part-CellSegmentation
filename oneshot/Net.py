import numpy as np
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import timm
import torchvision.models as models
import network

class UNet(nn.Module):
    def __init__(self, output=2):
        super(UNet, self).__init__()
        self.conv11 = nn.Conv2d(3,64,3,padding=1)
        self.conv12 = nn.Conv2d(64,64,3,padding=1)
        self.conv13 = nn.Conv2d(64,64,3,padding=1)

        self.conv21 = nn.Conv2d(64,128,3,padding=1)
        self.conv22 = nn.Conv2d(128,128,3,padding=1)
        self.conv23 = nn.Conv2d(128,128,3,padding=1)

        self.conv31 = nn.Conv2d(128,256,3,padding=1)
        self.conv32 = nn.Conv2d(256,256,3,padding=1)
        self.conv33 = nn.Conv2d(256,256,3,padding=1)

        self.conv41 = nn.Conv2d(256,512,3,padding=1)
        self.conv42 = nn.Conv2d(512,512,3,padding=1)
        self.conv43 = nn.Conv2d(512,512,3,padding=1)

        self.conv51 = nn.Conv2d(512,1024,3,padding=1)
        self.conv52 = nn.Conv2d(1024,1024,3,padding=1)
        self.conv53 = nn.Conv2d(1024,1024,3,padding=1)

        self.deconv1 = nn.ConvTranspose2d(1024+512,512,2,2)
        self.conv61 = nn.Conv2d(512,512,3,padding=1)
        self.conv62 = nn.Conv2d(512,512,3,padding=1)

        self.deconv2 = nn.ConvTranspose2d(512+256,256,2,2)
        self.conv71 = nn.Conv2d(256,256,3,padding=1)
        self.conv72 = nn.Conv2d(256,256,3,padding=1)

        self.deconv3 = nn.ConvTranspose2d(256+128,128,2,2)
        self.conv81 = nn.Conv2d(128,128,3,padding=1)
        self.conv82 = nn.Conv2d(128,128,3,padding=1)

        self.deconv4 = nn.ConvTranspose2d(128+64,64,2,2)
        self.conv91 = nn.Conv2d(64,64,3,padding=1)
        self.conv92 = nn.Conv2d(64,64,3,padding=1)

        self.conv10 = nn.Conv2d(64,output,1,1) 


        self.bn11 = nn.BatchNorm2d(64)
        self.bn12 = nn.BatchNorm2d(64)
        self.bn13 = nn.BatchNorm2d(64)

        self.bn21 = nn.BatchNorm2d(128)
        self.bn22 = nn.BatchNorm2d(128)
        self.bn23 = nn.BatchNorm2d(128)

        self.bn31 = nn.BatchNorm2d(256)
        self.bn32 = nn.BatchNorm2d(256)
        self.bn33 = nn.BatchNorm2d(256)

        self.bn41 = nn.BatchNorm2d(512)
        self.bn42 = nn.BatchNorm2d(512)
        self.bn43 = nn.BatchNorm2d(512)

        self.bn51 = nn.BatchNorm2d(1024)
        self.bn52 = nn.BatchNorm2d(1024)
        self.bn53 = nn.BatchNorm2d(1024)

        self.bn61 = nn.BatchNorm2d(512)
        self.bn62 = nn.BatchNorm2d(512)
        self.bnde1 = nn.BatchNorm2d(512)

        self.bn71 = nn.BatchNorm2d(256)
        self.bn72 = nn.BatchNorm2d(256)
        self.bnde2 = nn.BatchNorm2d(256)

        self.bn81 = nn.BatchNorm2d(128)
        self.bn82 = nn.BatchNorm2d(128)
        self.bnde3 = nn.BatchNorm2d(128)

        self.bn91 = nn.BatchNorm2d(64)
        self.bn92 = nn.BatchNorm2d(64)
        self.bnde4 = nn.BatchNorm2d(64)

        self.pool1 = nn.MaxPool2d(2,2)
        self.pool2 = nn.MaxPool2d(2,2)
        self.pool3 = nn.MaxPool2d(2,2)
        self.pool4 = nn.MaxPool2d(2,2)

        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        # block1
        h = F.relu(self.bn11(self.conv11(x)))
        h = F.relu(self.bn12(self.conv12(h)))
        h = F.relu(self.bn13(self.conv13(h)))
        h1 = self.pool1(h)

        # block2
        h = F.relu(self.bn21(self.conv21(h1)))
        h = F.relu(self.bn22(self.conv22(h)))
        h = F.relu(self.bn23(self.conv23(h)))
        h2 = self.pool2(h)

        # block3
        h = F.relu(self.bn31(self.conv31(h2)))
        h = F.relu(self.bn32(self.conv32(h)))
        h = F.relu(self.bn33(self.conv33(h)))
        h3 = self.pool3(h)

        # block4
        h = F.relu(self.bn41(self.conv41(h3)))
        h = F.relu(self.bn42(self.conv42(h)))
        h = F.relu(self.bn43(self.conv43(h)))
        h4 = self.pool4(h)

        # block5
        h = F.relu(self.bn51(self.conv51(h4)))
        h = F.relu(self.bn52(self.conv52(h)))
        h = F.relu(self.bn53(self.conv53(h)))

        # block 6
        h = torch.cat((h,h4),1)
        h = F.relu(self.bnde1(self.deconv1(h)))
        h = F.relu(self.bn61(self.conv61(h)))
        h = F.relu(self.bn62(self.conv62(h)))

        # block 7
        h = torch.cat((h,h3),1)
        h = F.relu(self.bnde2(self.deconv2(h)))
        h = F.relu(self.bn71(self.conv71(h)))
        h = F.relu(self.bn72(self.conv72(h)))

        # block 8
        h = torch.cat((h,h2),1)
        h = F.relu(self.bnde3(self.deconv3(h)))
        h = F.relu(self.bn81(self.conv81(h)))
        h = F.relu(self.bn82(self.conv82(h)))

        # block 9
        h = torch.cat((h,h1),1)
        h = F.relu(self.bnde4(self.deconv4(h)))
        h = F.relu(self.bn91(self.conv91(h)))
        h1 = F.relu(self.bn92(self.conv92(h)))

        h = self.conv10(h1)
        
        return h
        
        
class CrossAttention(nn.Module):
    def __init__(self, temp=1.0):
        super(CrossAttention, self).__init__()
        self.temp = temp

    def forward(self, h, xs, ys):
        B, C, H, W = h.shape
        Bs, Cs, Hs, Ws = xs.shape
        h = h.view(B, C, H*W)
        xs = xs.view(Bs, Cs, Hs*Ws)
        ys = ys.view(ys.shape[0], ys.shape[1], -1)
        
        # Attention #
        attention = torch.bmm(h.permute(0,2,1), xs) #[b, 196, 2048], [1, 2048, 16] --> [1, 14, 14, 16]
        attention = F.softmax(attention/self.temp, dim=2) # [1, 196, 16]
        xxs = torch.bmm(ys, attention.permute(0, 2, 1))#[b, 2048, 196], [1, 196, 16] --> [1, 2048, 16]
        xxs = xxs.view(B, C, H, W)
        
        return xxs 
        
        
class PromptSeg(nn.Module):
    def __init__(self, output=2, temp=1.0):
        super(PromptSeg, self).__init__()
        self.generator = UNet(output=output)
        
        #model_path = "model/{}/model_bestiou.pth".format(model_name)
        #self.generator.load_state_dict(torch.load(model_path))
        
        self.crossattention = CrossAttention(temp=temp)

    def forward(self, x, xs, ys):
        x = self.generator(x) #[1, 2048, 196]
        xs = self.generator(xs) #[1, 2048, 196]
        
        out = self.crossattention(x, xs, ys)
        
        return out, x, xs
        




