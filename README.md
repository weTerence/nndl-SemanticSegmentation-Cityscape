# 语义分割-Cityscape
本项目实现了FCN和ENet模型，基于Cityscape论文复现了以下两篇论文：[Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038)；[A Deep Neural Network Architecture for Real-Time Semantic Segmentation](https://arxiv.org/pdf/1606.02147)

项目小组成员：靳建华 20210980147，张晓琛 20210980070，马嘉晨 20210980109，付涵 20210980124



## 环境与依赖

python 3.7

pytorch 1.1.1

tensorboardX

tensorborad

Image

tqdm

requests

## 代码结构：

FCN模型的相关代码和预测结果放在"nndl-SemanticSegmentation-Cityscape/[FCN](https://github.com/weTerence/nndl-SemanticSegmentation-Cityscape/tree/master/FCN)/"文件夹下



## Usage
### Train
-----------------
- **Single GPU training**
```
# for example, train fcn32_vgg16_pascal_voc:
python train.py --model fcn32s --backbone vgg16 --dataset pascal_voc --lr 0.0001 --epochs 50
```
- **Multi-GPU training**

```
# for example, train fcn32_vgg16_pascal_voc with 4 GPUs:
export NGPUS=4
python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py --model fcn32s --backbone vgg16 --dataset pascal_voc --lr 0.0001 --epochs 50
```

### Evaluation
-----------------
- **Single GPU evaluating**
```
# for example, evaluate fcn32_vgg16_pascal_voc
python eval.py --model fcn32s --backbone vgg16 --dataset pascal_voc
```
- **Multi-GPU evaluating**
```
# for example, evaluate fcn32_vgg16_pascal_voc with 4 GPUs:
export NGPUS=4
python -m torch.distributed.launch --nproc_per_node=$NGPUS eval.py --model fcn32s --backbone vgg16 --dataset pascal_voc
```
### Demo
```
cd ./scripts
python demo.py --model fcn32s_vgg16_voc --input-pic ./datasets/test.jpg
```

```
.{SEG_ROOT}
├── scripts
│   ├── demo.py
│   ├── eval.py
│   └── train.py
```

```
.{SEG_ROOT}
├── core
│   ├── data
│   │   ├── dataloader
│   │   │   ├── ade.py
│   │   │   ├── cityscapes.py
│   │   │   ├── mscoco.py
│   │   │   ├── pascal_aug.py
│   │   │   ├── pascal_voc.py
│   │   │   ├── sbu_shadow.py
│   │   └── downloader
│   │       ├── ade20k.py
│   │       ├── cityscapes.py
│   │       ├── mscoco.py
│   │       ├── pascal_voc.py
│   │       └── sbu_shadow.py
```

## Overfitting Test
See [TEST](https://github.com/Tramac/Awesome-semantic-segmentation-pytorch/tree/master/tests) for details.

```
.{SEG_ROOT}
├── tests
│   └── test_model.py
```



## References
- [PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding)
- [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)
- [gloun-cv](https://github.com/dmlc/gluon-cv)
- [imagenet](https://github.com/pytorch/examples/tree/master/imagenet)

<!--
[![python-image]][python-url]
[![pytorch-image]][pytorch-url]
[![lic-image]][lic-url]
-->

[python-image]: https://img.shields.io/badge/Python-2.x|3.x-ff69b4.svg
[python-url]: https://www.python.org/
[pytorch-image]: https://img.shields.io/badge/PyTorch-1.1-2BAF2B.svg
[pytorch-url]: https://pytorch.org/
[lic-image]: https://img.shields.io/badge/Apache-2.0-blue.svg
[lic-url]: https://github.com/Tramac/Awesome-semantic-segmentation-pytorch/blob/master/LICENSE
