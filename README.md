# 环境
```shell
conda create -n slb python=3.11
conda activate slb

```



# 修改记录
## train.py
### 多gpu错误
应该是`--local-rank`，参数写成了`--local_rank`
