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
    

def Dice(output, target, label):
    output = np.array(output)
    target = np.array(target)
    seg = np.argmax(output,axis=1)
    seg = seg.flatten()
    target = target.flatten()
    mat = confusion_matrix(target, seg, labels=label)
    #sns.heatmap(mat, annot=True, fmt='.0f', cmap='jet')
    #plt.savefig("{}/CM.png".format(args.out))
    mat = np.array(mat).astype(np.float32)
    mat_all = mat.sum()
    diag_all = np.sum(np.diag(mat))
    fp_all = mat.sum(axis=1)
    fn_all = mat.sum(axis=0)
    tp_tn = np.diag(mat)
    precision = np.zeros((2)).astype(np.float32)
    recall = np.zeros((2)).astype(np.float32)
    f2 = np.zeros((2)).astype(np.float32)

    for i in range(2):
        if (fp_all[i] != 0)and(fn_all[i] != 0):  
            precision[i] = float(tp_tn[i]) / float(fp_all[i])
            recall[i] = float(tp_tn[i]) / float(fn_all[i])
            if (precision[i] != 0)and(recall[i] != 0):  
                f2[i] = (2.0*precision[i]*recall[i]) / (precision[i]+recall[i])
            else:       
                f2[i] = 0.0
        else:
            precision[i] = 0.0
            recall[i] = 0.0

    return precision, recall, f2


def Save_image(cll, cll_s, img, img_t, img_s, ano, ano_s, path, idx):
    img = np.argmax(img,axis=1)
    img_t = np.argmax(img_t,axis=1)
    img_s = np.argmax(img_s,axis=1)
    cll = cll[0,0]
    cll_s = cll_s[0,0]
    img = img[0]
    img_t = img_t[0]
    img_s = img_s[0]
    ano = ano[0]
    ano_s = np.argmax(ano_s,axis=1)
    ano_s = ano_s[0]
    dst1 = np.zeros((256,256,3))
    dst2 = np.zeros((256,256,3))
    dst3 = np.zeros((256,256,3))
    dst4 = np.zeros((64,64,3))
    dst5 = np.zeros((64,64,3))

    dst1[img==0] = [0.0,0.0,0.0]
    dst1[img==1] = [1.0,1.0,1.0]

    dst2[ano==0] = [0.0,0.0,0.0]
    dst2[ano==1] = [1.0,1.0,1.0]
    
    dst3[img_t==0] = [0.0,0.0,0.0]
    dst3[img_t==1] = [1.0,1.0,1.0]
    
    dst4[img_s==0] = [0.0,0.0,0.0]
    dst4[img_s==1] = [1.0,1.0,1.0]
    
    dst5[ano_s==0] = [0.0,0.0,0.0]
    dst5[ano_s==1] = [1.0,1.0,1.0]

    plt.imsave(path + "seg/{}.png".format(idx), dst1)
    plt.imsave(path + "ano/{}.png".format(idx), dst2)
    plt.imsave(path + "seg_t/{}.png".format(idx), dst3)
    plt.imsave(path + "seg_s/{}.png".format(idx), dst4)
    plt.imsave(path + "ano_s/{}.png".format(idx), dst5)
    plt.imsave(path + "inputs/{}.png".format(idx), cll, cmap='gray')
    plt.imsave(path + "inputs_s/{}.png".format(idx), cll_s, cmap='gray')
 
    
    plt.close()


def test():
    predict = []
    answer = []
    model_path = "{}_{}/model/model_bestiou.pth".format(args.out, args.temp)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets, sample_image, sample_label) in enumerate(tqdm(test_loader, leave=False)):
            sample_label = _one_hot_encoder(sample_label)
            inputs = inputs.cuda()
            targets = targets.cuda()
            sample_image = sample_image.cuda()
            sample_label = sample_label.cuda()
            targets = targets.long()
                
            output, out_t, out_s = model(inputs, sample_image, sample_label)
            inputs = inputs.cpu().numpy()
            output = output.cpu().numpy()
            out_t = out_t.cpu().numpy()
            out_s = out_s.cpu().numpy()
            targets = targets.cpu().numpy()
            sample_image = sample_image.cpu().numpy()
            sample_label = sample_label.cpu().numpy()
            
            for j in range(output.shape[0]):
                predict.append(output[j])
                answer.append(targets[j])

            Save_image(inputs, sample_image, output, out_t, out_s, targets, sample_label, "{}_{}/image/".format(args.out, args.temp), batch_idx+1)

        iou = IoU(predict, answer, label=[0,1])
        precision, recall, f2 = Dice(predict, answer, label=[0,1])

        miou = (iou[0] + iou[1]) / 2.0
        m0 = iou[0]
        m1 = iou[1]
        mm = miou

        dm = np.mean(f2)
        d0 = f2[0]
        d1 = f2[1]

        print("mIoU = %f ; class 0 = %f ; class 1 = %f" % (mm, m0, m1))
        print("Dice = %f ; class 0 = %f ; class 1 = %f" % (dm, d0, d1))
        with open(PATH, mode = 'a') as f:
            f.write("%f\t%f\t%f\n" % (mm, m0, m1))
        with open(PATH, mode = 'a') as f:
            f.write("%f\t%f\t%f\n" % (dm, d0, d1))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SR')
    parser.add_argument('--datapath', default='Dataset/ISBI2012/Crop')
    parser.add_argument('--out', type=str, default='result')
    parser.add_argument('--gpu', type=str, default=-1)
    parser.add_argument('--K', type=int, default=3)
    parser.add_argument('--cross', type=int, default=0)
    parser.add_argument('--temp', type=float, default=0.01)
    args = parser.parse_args()
    gpu_flag = args.gpu

    ### device ###
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    
    ### save ###
    if not os.path.exists("{}_{}".format(args.out, args.temp)):
      	    os.mkdir("{}_{}".format(args.out, args.temp))
    if not os.path.exists(os.path.join("{}_{}".format(args.out, args.temp), "image")):
      	    os.mkdir(os.path.join("{}_{}".format(args.out, args.temp), "image"))
    if not os.path.exists(os.path.join("{}_{}".format(args.out, args.temp), "image", "inputs")):
      	    os.mkdir(os.path.join("{}_{}".format(args.out, args.temp), "image", "inputs"))
    if not os.path.exists(os.path.join("{}_{}".format(args.out, args.temp), "image", "inputs_s")):
      	    os.mkdir(os.path.join("{}_{}".format(args.out, args.temp), "image", "inputs_s"))
    if not os.path.exists(os.path.join("{}_{}".format(args.out, args.temp), "image", "seg")):
      	    os.mkdir(os.path.join("{}_{}".format(args.out, args.temp), "image", "seg"))
    if not os.path.exists(os.path.join("{}_{}".format(args.out, args.temp), "image", "seg_s")):
      	    os.mkdir(os.path.join("{}_{}".format(args.out, args.temp), "image", "seg_s"))
    if not os.path.exists(os.path.join("{}_{}".format(args.out, args.temp), "image", "seg_t")):
      	    os.mkdir(os.path.join("{}_{}".format(args.out, args.temp), "image", "seg_t"))
    if not os.path.exists(os.path.join("{}_{}".format(args.out, args.temp), "image", "ano")):
      	    os.mkdir(os.path.join("{}_{}".format(args.out, args.temp), "image", "ano"))
    if not os.path.exists(os.path.join("{}_{}".format(args.out, args.temp), "image", "ano_s")):
      	    os.mkdir(os.path.join("{}_{}".format(args.out, args.temp), "image", "ano_s"))
      	
    PATH = "{}_{}/predict.txt".format(args.out, args.temp)
    
    with open(PATH, mode = 'w') as f:
        pass

    test_transform = ut.ExtCompose([ut.ExtResize((256, 256)),
                                    ut.ExtToTensor(),
                                    ut.ExtNormalize((0.466, 0.471, 0.380), (0.195, 0.194, 0.192))
                                    ])
                                    
    sample_transform = ut.ExtCompose([ut.ExtToTensor(),
                                      ut.ExtNormalize((0.466, 0.471, 0.380), (0.195, 0.194, 0.192))
                                      ])

    ### data loader ###
    data_test = ISBICellDataloader(root = args.datapath,    
                                  dataset_type='test',
                                  cross=args.cross,
                                  K=args.K,
                                  transform1=test_transform,
                                  transform2=sample_transform)

    test_loader = torch.utils.data.DataLoader(data_test, batch_size=4, shuffle=False, drop_last=True, num_workers=4)


    # networks #
    model = PromptSeg(output=2, temp=args.temp).cuda(device)       
                                            
    test()



