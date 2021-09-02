# MIT summer project 2021


* [环境要求](#环境要求)
* [如何运行](#如何运行)
    * [准备数据](#准备数据)
    * [调整超参](#调整超参)
    * [安装环境](#安装环境)
    * [前处理](#前处理)
    * [训练模型](#训练模型)
    * [压缩logs](#压缩logs)
* [Logs](#Logs)
* [Todo](#Todo)
* [写论文需要的内容](#写论文需要的内容)


## 环境要求
* pytorch 1.7
* torchio <= 0.18.20
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

### 重新拉取
```bash
git clone https://github.com/Vizards8/pytorch-spine-segmentation.git
```

### 安装环境
如在服务器上运行，须执行额外包安装:
```bash
pip install torchio tensorboard
```

### 前处理
```bash
python preprocess.py
```

### 训练模型
```bash
cd ../mnt/pytorch-spine-segmentation/
git reset --hard
git pull
nohup python main.py > runlog.txt 2>&1 &
```
查看显卡占用，确保正常运行
```bash
nvidia-smi
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
* 9.2 Tag:softmax+无loss权重+0.0005(0.0001不行)+train/valid都有aug，看着还行
    * lr0.0001，小样本实验，loss:0.49/0.75,IOU:0.61/0.25，上下有别的label的边
    * lr0.0005，me跑的0.25，0.8/10epoch，loss:,IOU:
    * me跑的0.25，50epoch，loss:0.46/0.51,IOU:0.63/0.54
    * hairu跑的all，晚上看结果
* 9.2 Tag:softmax+无loss权重+0.001(0.0005不行)+去掉valid的aug
    * lr0.0005，train/valid相差很大，我不懂，100epoch，loss:0.58/0.81,IOU:0.52/0.16
    * lr0.002，可以，就是收敛有点慢，78epoch，valid_IOU只有0.2
    * lr0.005，还是有点慢，关键颜色很混乱，100epoch，loss:0.44/0.62,IOU:0.4/0.2
    * lr0.01，仍然并不好看，100epoch，loss:0.36/0.52,IOU:0.43/0.20
    * 总结：validation颜色搅在一起，很奇怪，train没改，所以分割得很好也不奇怪
        
        
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