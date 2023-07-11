# Oneshot-Part-CellSegmentation
This repository is the official PyTorch implementation ''One-shot and Partially-Supervised Cell Image Segmentation Using Small Visual Prompt'' [[paper]](https://openaccess.thecvf.com/content/CVPR2023W/CVMI/papers/Kato_One-Shot_and_Partially-Supervised_Cell_Image_Segmentation_Using_Small_Visual_Prompt_CVPRW_2023_paper.pdf)

## Introduction
<div align="center">
  <img src="figs/git_fig1.png" width="80%">
</div>
Dice loss is widely used for medical image segmentation, and many improvement loss functions based on such loss have been proposed. However, further Dice loss improvements are still possible. In this study, we reconsidered the use of Dice loss and discovered that Dice loss can be rewritten in the loss function using the cosine similarity through a simple equation transformation. Using this knowledge, we present a novel t-vMF Dice loss based on the t-vMF similarity instead of the cosine similarity. Based on the t-vMF similarity, our proposed Dice loss is formulated in a more compact similarity loss function than the original Dice loss. Furthermore, we present an effective algorithm that automatically determines the parameter $\kappa$ for the t-vMF similarity using a validation accuracy, called Adaptive t-vMf Dice loss. Using this algorithm, it is possible to apply more compact similarities for easy classes and wider similarities for difficult classes, and we are able to achieve an adaptive training based on the accuracy of the class. Through experiments conducted on four datasets using a five-fold cross validation, we confirmed that the Dice score coefficient (DSC) was further improved in comparison with the original Dice loss and other loss functions.
<br />
<br />

## Preparation for preprocessing datasets
Please download from [[Dataset]](https://drive.google.com/drive/folders/1q80fDpAM62jPR5p61_4BVzCX4I1KZMqt?usp=drive_link) and extract them under "data", and make them look like this:
```
data
├── CVC-ClinicDB
    ├── datamodel
        ├── train_data_1.npy
        ├── train_label_1.npy
        ├── ...
├── Kvasir-SEG
├── ACDC
`── Synapse

```

## Pre-trained model for TransUNet and FCBFormer
Please download pre-trained models for TransUNet and FCBFormer encoders from [[TransUNet]](https://github.com/Beckschen/TransUNet), and [[FCBFormer]](https://github.com/ESandML/FCBFormer).

## Training
If you prepared the dataset, you can directly run the following code to train the model.
```
sh train.sh
```

## Testing
If you generated the pretrain model, you can run the following code to evaluate the model.
```
sh test.sh
```

## Results and Visualization
<div align="center">
  <img src="figs/git_fig3.png" width="100%">
</div>

## Citation
```
@misc{kato2022adaptive,
      title={Adaptive t-vMF Dice Loss for Multi-class Medical Image Segmentation}, 
      author={Sota Kato and Kazuhiro Hotta},
      year={2022},
      eprint={2207.07842},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```
