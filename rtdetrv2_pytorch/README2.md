## 配置环境

```
uv sync
source .venv/bin/activate
```

## 准备预训练模型

预训练模型为 [RTDETRv2-S] (https://github.com/lyuwenyu/storage/releases/download/v0.2/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth)

## 准备数据集

使用 `bproc-container-08.zip`

已上传 OneDrive / Minio

## 训练模型

`python tools/train.py -c configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml -t pretrained_models/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth`

## Tensorboard

`tensorboard --logdir=output --bind_all`