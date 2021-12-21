## 文件目录
* 解压文件至当前目录
```
pytorch-spine-segmentation
└── plot
    ├── Attention U-Net
    ├── DANet
    ├── DeepLabv3+
    ├── DenseASPP
    ├── EMANet
    ├── Inf-Net
    ├── MiniSeg
    ├── PSANet
    ├── PSPNet
    ├── R2UNet
    ├── ResUNet++
    ├── SegNet
    ├── TransUNet
    ├── U-Net
    ├── model_para.json
    ├── plot.py
    └── README.md
```

##画图
```
python plot.py
```
* 结果会保存在Result中
* 同时plot.py会调用```pic/```中的图片，合成完整分割效果大图，```pic/```暂不上传，代码仅供参考，若报错，请注释plot.py320行
