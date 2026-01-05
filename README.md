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
所以SLBD.yaml需要改一下路径。懒得划分数据集了。
```shell
train: data/riseHand_Dataset/images
val: data/riseHand_Dataset/images
test: data/riseHand_Dataset/images
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
重复定义了`LBASwinTransformerblock`，删掉第二个，第二个接收的参数不是配置的格式。


## models/yolo.py
1. 没有引用的`SwinTransformerBlock`, 直接删掉
2. LBASwinTransformerblock也需要加进去
3. MP模块不应该在`if`里处理

## utils/datasets.py
经典weights_only=False
