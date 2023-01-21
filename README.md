# RINDNet-plusplus

## Usage
### Training
1. Clone this repository to local
```shell
git clone https://github.com/MengyangPu/RINDNet-plusplus.git
```

2. Download the [augmented data]() to the local folder /data

3. run train
```shell
python train_RINDNet_plusplus_80k.py
or
python train_RINDNet_plusplus_edge_80k.py
```
more train files (train_*modelname*_80k.py and train_*modelname_edge*_80k.py) in [/train_tools](train_tools)

4. Note: The imagenet pretrained vgg16 pytorch model for BDCN can be downloaded in [vgg16.pth](link: https://pan.baidu.com/s/10Tgjs7FiAYWjVyVgvEM0mA) code: ab4g.
         The imagenet pretrained vgg16 pytorch model for HED can be downloaded in [5stage-vgg.py36pickle](https://pan.baidu.com/s/1lQbAnNhymhXPYM2wL0cSnA) code: 9po1.
         
         
### Plot edge PR curves of RIND++
We have released the code and data for plotting the edge PR curves of the above edge detectors [here](https://github.com/MengyangPu/RINDNet-plusplus/tree/main/plot-rind-edge-pr-curves).      

### Precomputed Results
If you want to compare your method with RINDNet and other methods, you can download the precomputed results [here](https://pan.baidu.com/s/1SEQdbibqnntb_fJqiw2VVw) (code: ewco).

## Acknowledgments
- The work is partially done while Mengyang was at Stony Brook University.
- Thanks to previous open-sourced repo:<br/>
  [HED-pytorch](https://github.com/xwjabc/hed)<br/>
  [RCF-pytorch](https://github.com/meteorshowers/RCF-pytorch)<br/>
  [BDCN](https://github.com/pkuCactus/BDCN)<br/>
  [DexiNed](https://github.com/xavysp/DexiNed)<br/>
  [DFF](https://github.com/Lavender105/DFF)<br/>
  [pytorch-deeplab-xception](https://github.com/jfzhang95/pytorch-deeplab-xception)<br/>
  [DOOBNet-pytorch](https://github.com/yuzhegao/doob)
