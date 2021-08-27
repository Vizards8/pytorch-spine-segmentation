# MIT summer project 2021


* [环境要求](#环境要求)
* [如何运行](#如何运行)
    * [准备数据](#准备数据)
    * [调整超参](#调整超参)
    * [训练模型](#训练模型)
* [Logs](#Logs)
* [Todo](#Todo)
* [写论文需要的内容](#写论文需要的内容)


## 环境要求
* pytorch 1.7
* torchio <= 0.18.20
* python >= 3.6


## 如何运行
### 准备数据
* 将老师给的 *mit_ai_2021_course_2_project_1_dataset_train_1* 和 *mit_ai_2021_course_2_project_1_dataset_train_2* 合并
* 不使用 *mit_ai_2021_course_2_project_1_dataset_test* 中的数据，因为没有标签
* 将数据如下排列：务必检查好source200，label200，共计400个文件
```
pytorch-spine-segmentation
├── dataset
    └── train
        ├── source
        │   ├── Case1.nii.gz
        │   ├── Case2.nii.gz
        │   ├── ...
        │   └── Case200.nii.gz
        └── label
            ├── mask_case1.nii.gz
            ├── mask_case2.nii.gz
            ├── ...
            └── mask_case200.nii.gz
```

### 调整超参
* 如有必要，详见hparam.py

### 安装环境
* 如在服务器上运行，须执行额外包安装:
```bash
pip install torchio
pip install tensorboard
```

### 前处理
```bash
python preprocess.py
```

### 训练模型
```bash
python main.py
```

### 查看效果
* 在terminal输入以下内容，并点击网址:
```bash
tensorboard --logdir logs
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
    * FCN
    * UNet++
    * PSPNet
    * DeepLabv3