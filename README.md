# MIT summer project 2021


* [环境要求](#环境要求)
* [如何运行](#如何运行)
    * [准备数据](#准备数据)
    * [调整超参](#调整超参)
    * [安装环境](#安装环境)
    * [下载代码](#下载代码)
    * [前处理](#前处理)
    * [训练模型](#训练模型)
    * [tensorboard](#tensorboard)
    * [压缩logs](#压缩logs)
* [Logs](#Logs)
* [Todo](#Todo)
* [写论文需要的内容](#写论文需要的内容)


## 环境要求
* pytorch 1.7
* torchio <= 0.18.45
* python >= 3.6


## 如何运行
### 准备数据
* 将老师给的`mit_ai_2021_course_2_project_1_dataset_train_1`和`mit_ai_2021_course_2_project_1_dataset_train_2`合并
* 不使用`mit_ai_2021_course_2_project_1_dataset_test`中的数据，因为没有标签
* 将数据如下排列：务必检查好source200，label200，共计400个文件
```
pytorch-spine-segmentation
├── dataset
    ├── train
    │    ├── source
    │    │   ├── Case1.nii.gz
    │    │   ├── Case2.nii.gz
    │    │   ├── ...
    │    │   └── Case160.nii.gz
    │    └── label
    │        ├── mask_case1.nii.gz
    │        ├── mask_case2.nii.gz
    │        ├── ...
    │        └── mask_case160.nii.gz
    └── test
        ├── source
        │   ├── Case161.nii.gz
        │   ├── Case162.nii.gz
        │   ├── ...
        │   └── Case200.nii.gz
        └── label
            ├── mask_case161.nii.gz
            ├── mask_case162.nii.gz
            ├── ...
            └── mask_case200.nii.gz
```

### 调整超参
* 如有必要，详见hparam.py

### 安装环境(仅第一次，可保存不用反复安装)
如在服务器上运行，须执行额外包安装:
```bash
pip install torchio==0.18.45 tensorboard
```

### 下载代码(仅第一次)
```bash
cd /mnt/
git clone https://github.com/Vizards8/pytorch-spine-segmentation.git -b v1.1
cd pytorch-spine-segmentation/
```
将`dataset.zip`放在`pytorch-spine-segmentation`根目录下
```bash
unzip dataset.zip
```

### 前处理(仅第一次)
```bash
python preprocess.py
```
运行结束后核对数量，slice_train:source/label各2016,slice_test:source/label各507

### 训练模型(每次)
```bash
cd /mnt/pytorch-spine-segmentation/
nohup python main.py > runlog.txt 2>&1 &
```
查看显卡占用，确保正常运行
```bash
nvidia-smi
```

### tensorboard
```bash
cd /mnt/pytorch-spine-segmentation/
tensorboard --logdir logs
```

### 压缩logs(均可)
```bash
zip -r logs.zip logs/
tar -cvf logs.tar.gz logs/
```

## Logs
* 8.14 多分类标签问题调整完毕，现输出18分类，效果还行
* 8.15 添加validation
* 8.16 尝试colab，但是爆内存，修改切片大小为660*660，能跑，但是gpu限额
* 8.17 添加读数据进度条
* 8.17 添加test_split参数，可调整每次读取的数据量，不必删数据集
* 8.17 添加跑代码过程中的进度条，优化可读性
* 8.18 添加use_queue参数，可将一张原图切成多张小图读取
* 8.18 修改test
* 8.18 添加postprocess，组合为三维MRI
* 8.19 添加SegNet支持
* 8.20 添加PSPNet支持
* 8.23 添加Using Device，方便调试
* 8.25 添加IOU，Dice，FP，FN，待测试
* 8.27 测试GPU服务器性能，综合考虑，优选3090
* 8.27 搞半天label是0-19，20分类¿我TM
* 8.27 use_queue,sampler任然存在问题，but因为用不到，所以不改了
* 8.27 惊魂未定
* 8.28 问题排查，排除笔记本问题，但UNet的评价指标仍然异常
* 8.29 修改数据集划分，1-160位训练集，161-200为测试集，方便后续统计
* 8.31 怀疑：
    - [ ] 数据增强是不是太多了
    - [x] lr是不是太大：修改lr = 0.0001，可行，但发现了后续其他bug
    - [x] 权重初始化是不是不对，应该对的
    - [x] 为什么预测是0，但是loss还在下降
        * 因为你用的sigmoid，他当成独立概率了，有一个为0.99，其他可能还有0.4，这个在降低，loss也在降低
    - [x] 用softmax还是sigmoid，在测，均使用0.25的总数据集 *不测了肯定用softmax*
        * sigmoid，45epoch，loss0.3-0.4，缓慢下降
        * softmax，lr = 0.002，王懿测
        * softmax，lr = 0.0001，去掉/不去掉loss系数，都有个背景色label9，周能测
        * softmax，为什么val_loss这么大？因为还是sigmoid没改，我是伞兵
* 9.1 bug：包括但不限于
    * 我为什么用sigmoid
    * dice loss为什么要加权重
    * 我为什么要对测试集做数据增强
    * lr应该是多少
* 9.2 Tag_v1.0:softmax+无loss权重+0.0005(0.0001不行)+train/valid都有aug，看着还行
    * lr0.0001，小样本实验，loss:0.49/0.75,IOU:0.61/0.25，上下有别的label的边
    * lr0.0005
        * me跑的0.25，0.8/10epoch，50epoch，loss:0.46/0.50,IOU:0.63/0.55，对比下面貌似更好看一点
        * me跑的0.25，50epoch，loss:0.46/0.51,IOU:0.63/0.54
        * hairu跑的all，晚上看结果
            - [x] 12h,30epoch,loss:0.2331/0.2225,IOU:0.563/0.5855(没写错)
            - [x] 1d10h,80epoch,loss:0.1903/0.1975,IOU:0.6291/0.6246
* 9.2 测试：softmax+无loss权重+0.001(0.0005不行)+去掉valid的aug
    * lr0.0005，train/valid相差很大，我不懂，100epoch，loss:0.58/0.81,IOU:0.52/0.16
    * lr0.002，可以，就是收敛有点慢，78epoch，valid_IOU只有0.2
    * lr0.005，还是有点慢，关键颜色很混乱，100epoch，loss:0.44/0.62,IOU:0.4/0.2
    * lr0.01，仍然并不好看，100epoch，loss:0.36/0.52,IOU:0.43/0.20
    * 总结：validation颜色搅在一起，很奇怪，train没改，所以分割得很好也不奇怪
* 9.3 Tag_v1.1:v1.0+test没有aug+metrics去除indices
    * hairu跑的all，明天看结果
* 9.16-17
    * ESPNet:
        * `2018: ESPNet: Efficient Spatial Pyramid of Dilated Convolutions for Semantic Segmentation`, `296 Ciatations`
        * 好复杂啊，实现不了，告辞
    * CaraNet:
        * `CaraNet: Context Axial Reverse Attention Network for Segmentation of Small Medical Objects`, `0 Citations`
        * 添加final layer解决20分类的问题，对应的需要改upsample的scale_factor为1/2
    * ENet:
        * `2016: A Deep Neural Network Architecture for Real-Time Semantic Segmentation`
    * GCN:
        * `2017: Large Kernel Matters——Improve Semantic Segmentation by Global Convolutional Network`
        * 后面loss不降了，dice只有0.26左右，全黑
    * ResUNet:
        * `Road Extraction by Deep Residual U-Net`
    * ResUNetpp:
        * `ResUNet++: An Advanced Architecture for MedicalImage Segmentation`
    * TransUNet:
        * `TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation`
* 9.25-27
    * Res2Net最后三层注释了，因为没用到
    * PSPNet对比实验:
        * ResNet34:
            * /8只有三个upsample
            * conv2d->/2, maxpool->/2, block2->/2 --> batchsize=4.dice=0.85
        * Res2Net101:09250202
            * PSPNet修改三处，有注释
            * maxpool->/2, block2->/2, block3->/2 --> 模型过大:batchsize=1
        * Res2Net101:09260128
            * conv2d->/2, maxpool->/2, block2->/2 --> batchsize=2,dice=0.8
        * Res2Net50:09262300
            * conv2d->/2, maxpool->/2, block2->/2 --> batchsize=2,dice=0.8
        * ResNet101:09261258
            * conv2d->/2, maxpool->/2, block2->/2 --> batchsize=2,dice<0.65
        * ResNet34:09271013
            * conv2d->/2, maxpool->/2, block2->/2 --> batchsize=4,dice=0.8
        * ResNet101:09271605
            * dilation -->1/1/2/4
            * conv2d->/2, maxpool->/2, block2->/2 --> batchsize=2,dice=0.74
* 9.28-30
    * F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
    * assert layers in [50, 101, 152]
    * PraNet:
        * `PraNet: Parallel Reverse Attention Network for Polyp Segmentation`, `69 Citations`
        * Res2Net50:0.746
        * Res2Net101:0.735
    * SegFormer:
        * `SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers`
        * 是我想多了，输出1/4，我也不知道为啥
    * DANet:
        * `2019: Dual Attention Network for Scene Segmentation`, `1342 Citations`
        * ResNet50:0.81
        * ResNet101:0.83
        * upsample直接一步到位
    * CCNet:
        * `2019: CCNet: Criss-cross attention for semantic segmentation`, `593 Citations`
        * from inplace_abn import InPlaceABN, InPlaceABNSync
        * 没有上采样，我不理解;我自己加上了，不知道成不成
        * ResNet101 + recurrence=2:waiting
    * ANNNet:
        * ` Automatic detection of coronavirus disease (COVID-19) using X-ray images and deep convolutional neural networks`, `579 Citations`
        * ResNet101:waiting
        * upsample直接一步到位
    * GFF:
        * `2020: Gated Fully Fusion for Semantic Segmentation`, `15 Citations`
        * 本身成绩一般, Cityscapes test:82.3%, #20
    * Inf-Net:
        * `Inf-Net: Automatic COVID-19 Lung Infection Segmentation from CT Images`, `185 Citations`
        * Res2Net50:0.75
        * Res2Net101:0.81
    * EMANet:
        * `Expectation-Maximization Attention Networks for Semantic Segmentation`, `149 Citations`
        * ResNet101:0.85
    * OCNet:
        * `2018: OCNet: Object Context Network for Scene Parsing`, `287 Citations`
        * ResNet101:waiting
    * ANN:
        * `2019: Asymmetric Non-Local Neural Networks for Semantic Segmentation`, `188 Citations`
        * ResNet101:
    * DenseASPP:
        * `2018: DenseASPP for Semantic Segmentation in Street Scenes`, `341 Citations`
        * DenseASPP169:0.695
        * DenseASPP201:waiting
    * PSANet:
        * `2018: PSANet: Point-wise Spatial Attention Network for Scene Parsing`, `404 Citations`
        * 加了个lib_psa, 这个除法还蛮奇怪的
        * ResNet101:0.85
    * BiSeNetv2:
        * `2020: BiSeNet V2: Bilateral Network with Guided Aggregation for Real-time Semantic Segmentation`, `57 Citations`
        * 不清楚backbone:waiting
        
        
## Todo
- [ ] 切片880*880过大，不合理，需要trick
- [x] 边缘切片没有18个分类，效果肯定不好，是否影响整体模型 *仍然放进去训练*
- [x] RandomElasticDeformation()会报warning，*为了观感,暂时去除*
- [x] 其他模型
- [ ] 其他Loss函数
- [x] 其他评价指标
- [x] 有的切片没有18个分类，导致one-hot存在全0tensor，tp = 0, fn = 0, 所以正确率虚高 *已去除*


## 写论文需要的内容
* number of params
* evaluate
    * IOU
    * Dice
    * pixel acc
* model
    * UNet
    * SegNet
    * ~~FCN~~
    * UNet++
    * PSPNet
    * DeepLabv3