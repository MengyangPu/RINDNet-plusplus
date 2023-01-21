# RINDNet-plusplus
> [RINDNet++: Edge Detection for Discontinuity in Reflectance, Illumination, Normal, and Depth]()             
> Mengyang Pu, Yaping Huang, Qingji Guan, Zhihao Liu, and Haibin Ling                 
> *Under review*

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
