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

FCN模型的相关代码和预测结果放在"nndl-SemanticSegmentation-Cityscape/FCN/"文件夹下，**具体怎么运行FCN相关代码请参考"nndl-SemanticSegmentation-Cityscape/FCN/README.md"**

ENet模型的相关代码和预测结果放在"nndl-SemanticSegmentation-Cityscape/ENet/"文件夹下，**具体怎么运行FCN相关代码请参考"nndl-SemanticSegmentation-Cityscape/ENet/README.md"**



```
.{SEG_ROOT}
├── tests
│   └── test_model.py
```



## References
code:

- [awesome-semantic-segmentation-pytorch](https://github.com/Tramac/awesome-semantic-segmentation-pytorch)
- [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)

paper:

- [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038)
- [A Deep Neural Network Architecture for Real-Time Semantic Segmentation](https://arxiv.org/pdf/1606.02147)