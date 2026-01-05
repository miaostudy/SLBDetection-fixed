# 环境
```shell
conda create -n slb python=3.11
conda activate slb
pip install -r requirements.txt
```

# 训练
```shell
python -m torch.distributed.launch --nproc_per_node 2 --master_port 9527 train.py \
    --workers 8 \
    --device 0,1 \
    --sync-bn \
    --batch-size 128 \
    --data data/SLBD.yaml \
    --img 640 \
    --cfg cfg/SLBD/lba.yaml \
    --weights '' \
    --name yolov7
```


# 修改记录
## train.py
### 多gpu错误
应该是`--local-rank`，参数写成了`--local_rank`

## SLBD.yaml
改了路径。 目前的数据集结构:
```shell
|data
  |- riseHand_Dataset
    |- images
    |- labels
```
所以SLBD.yaml需要改一下路径。
```shell
train: 'data/riseHand_Dataset/images/train'
val: 'data/riseHand_Dataset/images/val'
test: 'data/riseHand_Dataset/images/val'

# number of classes
nc: 1

# class names
names: ['raise_hand']
```

##  lba.yaml
有一些语法错误。少了逗号、括号啥的。

最后几层的索引错误
```yaml
   [96, 1, Conv, [384, 3, 1]],
   [112, 1, Conv, [768, 3, 1]],
   [128, 1, Conv, [1152, 3, 1]],
   [144, 1, Conv, [1536, 3, 1]],

   [[145,146,147,148], 1, Detect, [nc, anchors]],
```
为啥连配置文件都写不对???

## models/common.py
重复定义了`LBASwinTransformerblock`，只保留一个就行。
`LBASwinTransformerblock`的forward有大问题。


## models/yolo.py
1. 没有引用的`SwinTransformerBlock`, 直接删掉
2. LBASwinTransformerblock也需要加进去
3. MP模块不应该在`if`里处理

## utils/datasets.py
经典weights_only=False

## 性能问题
LBASwinTransformerblock中有两个部分：
1. Shifted Windows：必须需要 Mask。移位后的窗口包含了图像中原本不相邻的部分。**必须使用 Mask 来防止注意力机制混淆这些原本不相关的像素。**

2. LearningBehaviorawareAttention：完全没有使用 **Mask**

这导致：模型在计算注意力时，**将图像左边缘的特征和右边缘的特征进行了融合**，彻底破坏了特征的空间结构。

解决：使用标准的 WindowAttention


# 模型方面的问题
1. LBASwinTransformerblock只是单层的Block。
2. 传入LBASwinTransformer的是[256, [32,32], 4, 8]，即(dim, input_resolution, num_heads, window_size)，那么**shift_size**就是默认的0

核心：所有Transformer的shift_size均为0，这些层只能在固定的 8x8 窗口内提取特征，窗口之间没有任何信息交流。

3. input_resolution是[32,32], [16,16], [8,8]，分辨率太低了，加上窗口又是8、4、2，几乎没有感受野。

