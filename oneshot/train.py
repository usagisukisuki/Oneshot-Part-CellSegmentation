#coding: utf-8
import numpy as np
import os
import argparse
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from PIL import Image

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn

from Net import PromptSeg
from Mydataset import ISBICellDataloader
import utils as ut




### onehot label ###
def _one_hot_encoder(input_tensor):
    tensor_list = []
    for i in range(2):
        temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
        tensor_list.append(temp_prob.unsqueeze(1))
    output_tensor = torch.cat(tensor_list, dim=1)
    return output_tensor.float()


### IoU ###
def IoU(output, target, label):
    output = np.array(output)
    target = np.array(target)
    seg = np.argmax(output,axis=1)
    seg = seg.flatten()
    target = target.flatten() 
    mat = confusion_matrix(target, seg, labels=label)
    iou_den = (mat.sum(axis=1) + mat.sum(axis=0) - np.diag(mat))
    iou = np.array(np.diag(mat) ,dtype=np.float32) / np.array(iou_den, dtype=np.float32)

    return iou


### training ###
def train(epoch):
    sum_loss = 0
    model.train()
    for batch_idx, (inputs, targets, sample_image, sample_label) in enumerate(tqdm(train_loader, leave=False)):
        sample_label = _one_hot_encoder(sample_label)
        inputs = inputs.cuda()
        targets = targets.cuda()
        sample_image = sample_image.cuda()
        sample_label = sample_label.cuda()
        targets = targets.long()

        output, _, _ = model(inputs, sample_image, sample_label)
        loss = criterion(output, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sum_loss += loss.item()
 
    return sum_loss/(batch_idx+1)



### validation ###
def validation(epoch):
    sum_loss = 0
    model.eval()
    predict = []
    answer = []
    with torch.no_grad():
        for batch_idx, (inputs, targets, sample_image, sample_label) in enumerate(tqdm(val_loader, leave=False)):
            sample_label = _one_hot_encoder(sample_label)
            inputs = inputs.cuda()
            targets = targets.cuda()
            sample_image = sample_image.cuda()
            sample_label = sample_label.cuda()
            targets = targets.long()

            output, _, _ = model(inputs, sample_image, sample_label)
            loss = criterion(output, targets)

            sum_loss += loss.item() 

            ### save output ###
            output = F.softmax(output, dim=1)
            inputs = inputs.cpu().numpy()
            output = output.cpu().numpy()
            targets = targets.cpu().numpy()
            
            for i in range(args.batchsize):
                predict.append(output[i])
                answer.append(targets[i])


        ### IoU ###
        iou = IoU(predict, answer, label=[0,1])


    return sum_loss/(batch_idx+1), iou.mean()
    
def adjust_learning_rate(optimizer, epoch, lr):
    epoch = epoch + 1
    if  epoch > 190:#169
        lr = lr * 0.01
    elif epoch > 180:#99
        lr = lr * 0.1
    else:
        if epoch <= 5:
           lr = lr * epoch / 5
        else:
           lr = lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


##### main #####
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='One-shot segmentation')
    parser.add_argument('--batchsize', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--datapath', default='Dataset/ISBI2012')
    parser.add_argument('--out', type=str, default='result')
    parser.add_argument('--gpu', type=str, default=-1)
    parser.add_argument('--K', type=int, default=3)
    parser.add_argument('--cross', type=int, default=0)
    parser.add_argument('--aug', action='store_true')
    parser.add_argument('--temp', type=float, default=1.0)
    args = parser.parse_args()
    gpu_flag = args.gpu

    ### device ###
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    
    ### save ###
    if not os.path.exists("{}_{}".format(args.out, args.temp)):
      	os.mkdir("{}_{}".format(args.out, args.temp))
    if not os.path.exists(os.path.join("{}_{}".format(args.out, args.temp), "model")):
      	os.mkdir(os.path.join("{}_{}".format(args.out, args.temp), "model"))
      	
    PATH_1 = "{}_{}/trainloss.txt".format(args.out, args.temp)
    PATH_2 = "{}_{}/testloss.txt".format(args.out, args.temp)
    PATH_3 = "{}_{}/IoU.txt".format(args.out, args.temp)
    
    with open(PATH_1, mode = 'w') as f:
        pass
    with open(PATH_2, mode = 'w') as f:
        pass
    with open(PATH_3, mode = 'w') as f:
        pass


    ### seed ###
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    ### augmentation ###
    if args.aug:
        train_transform = ut.ExtCompose([ut.ExtRandomCrop((128, 128)),
                                         ut.ExtRandomRotation(degrees=90),
                                         ut.ExtRandomHorizontalFlip(),
                                         ut.ExtToTensor(),
                                         ut.ExtNormalize((0.466, 0.471, 0.380), (0.195, 0.194, 0.192))
                                         ])
                                     
        test_transform = ut.ExtCompose([ut.ExtResize((256, 256)),
                                        ut.ExtToTensor(),
                                        ut.ExtNormalize((0.466, 0.471, 0.380), (0.195, 0.194, 0.192))
                                        ])
                                        
        sample_transform = ut.ExtCompose([ut.ExtToTensor(),
                                          ut.ExtNormalize((0.466, 0.471, 0.380), (0.195, 0.194, 0.192))
                                          ])

    else:
        train_transform = ut.ExtCompose([ut.ExtResize((256, 256)),
                                         ut.ExtToTensor(),
                                         ut.ExtNormalize((0.466, 0.471, 0.380), (0.195, 0.194, 0.192))
                                         ])
        test_transform = ut.ExtCompose([ut.ExtResize((256, 256)),
                                         ut.ExtToTensor(),
                                         ut.ExtNormalize((0.466, 0.471, 0.380), (0.195, 0.194, 0.192))
                                         ])
                                         
        sample_transform = ut.ExtCompose([ut.ExtToTensor(),
                                      ut.ExtNormalize((0.466, 0.471, 0.380), (0.195, 0.194, 0.192))
                                      ])

    ### data loader ###
    data_train = ISBICellDataloader(root = args.datapath, 
                                    dataset_type='train',
                                    cross=args.cross,
                                    K=args.K,
                                    transform1=test_transform,
                                    transform2=sample_transform)
                                   
    data_val = ISBICellDataloader(root = args.datapath,    
                                  dataset_type='test',
                                  cross=args.cross,
                                  K=args.K,
                                  transform1=test_transform,
                                  transform2=sample_transform)
                                   
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=args.batchsize, shuffle=True, drop_last=True, num_workers=1)
    val_loader = torch.utils.data.DataLoader(data_val, batch_size=args.batchsize, shuffle=True, drop_last=True, num_workers=1)



    # networks #
    model = PromptSeg(output=2, temp=args.temp).cuda(device)


    # optimizer # 
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    
    
    # loss function #
    criterion = nn.CrossEntropyLoss()


    ### training & validation ###
    sample = 0
    sample_loss = 10000000
    
    for epoch in range(args.num_epochs):
        adjust_learning_rate(optimizer, epoch, lr=1e-3)
        loss_train = train(epoch)
        loss_test, mm = validation(epoch)

        print("epoch %d / %d" % (epoch+1, args.num_epochs))
        print('train loss: %.4f' % loss_train)
        print('test loss : %.4f' % loss_test)
        print(" Mean IoU : %.4f" % mm)
        print("")

        ### save ###
        with open(PATH_1, mode = 'a') as f:
            f.write("\t%d\t%f\n" % (epoch+1, loss_train))
        with open(PATH_2, mode = 'a') as f:
            f.write("\t%d\t%f\n" % (epoch+1, loss_test))
        with open(PATH_3, mode = 'a') as f:
            f.write("\t%d\t%f\n" % (epoch+1, mm))


        ### model save ###
        PATH1 ="{}_{}/model/model_train.pth".format(args.out, args.temp)
        torch.save(model.state_dict(), PATH1)


        ### best miou ###
        if mm >= sample:
           sample = mm
           PATH1_best ="{}_{}/model/model_bestiou.pth".format(args.out, args.temp)
           torch.save(model.state_dict(), PATH1_best)

        ### best test loss ###
        if loss_train < sample_loss:
           sample_loss = loss_train
           PATH1_best ="{}_{}/model/model_bestloss.pth".format(args.out, args.temp)
           torch.save(model.state_dict(), PATH1_best)
 
