# 语义分割-Cityscape
本项目实现了FCN和ENet模型，基于Cityscape论文复现了以下两篇论文：[Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038)；[A Deep Neural Network Architecture for Real-Time Semantic Segmentation](https://arxiv.org/pdf/1606.02147)

项目小组成员：靳建华 20210980147，张晓琛 20210980070，马嘉晨 20210980109，付涵 20210980124



## 环境与依赖

| python 3.7             |
| ---------------------- |
| kiwisolver>=1.0.1      |
| matplotlib>=3.0.2      |
| numpy>=1.16.0          |
| Pillow>=6.2.0          |
| pyparsing>=2.3.1       |
| python-dateutil>=2.7.5 |
| pytz>=2018.9           |
| six>=1.12.0            |
| pytorch 1.1.1          |
| tensorboardX           |
| tensorborad            |
| cycler>=0.10.0         |
| Image                  |
| tqdm                   |
| requests               |

## 代码结构：

FCN模型的相关代码和预测结果放在"nndl-SemanticSegmentation-Cityscape/FCN/"文件夹下，**具体怎么运行FCN相关代码请参考"nndl-SemanticSegmentation-Cityscape/FCN/README.md"**

ENet模型的相关代码和预测结果放在"nndl-SemanticSegmentation-Cityscape/ENet/"文件夹下，**具体怎么运行FCN相关代码请参考"nndl-SemanticSegmentation-Cityscape/ENet/README.md"**



## 代码使用

#### 1. Train for FCN

```
python train.py --model fcn32s --backbone vgg16 --dataset pascal_voc --lr 0.0001 --epochs 50
```

此处train.py文件为"nndl-SemanticSegmentation-Cityscape/FCN/scripts/train.py"

#### 2. Evaluation and test for FCN

```
python eval.py --model fcn32s --backbone vgg16 --dataset citys
```

此处train.py文件为"nndl-SemanticSegmentation-Cityscape/FCN/scripts/eval.py"

#### 3. Train for ENet

```
python main.py -m train --save-dir save/folder/ --name model_name --dataset name --dataset-dir path/root_directory/
```

此处main.py文件为"nndl-SemanticSegmentation-Cityscape/ENet/main.py"

#### 4. Evaluation and test for ENet

```
python main.py -m train --resume True --save-dir save/folder/ --name model_name --dataset name --dataset-dir path/root_directory/
```

```
python main.py -m test --save-dir save/folder/ --name model_name --dataset name --dataset-dir path/root_directory/
```

此处main.py文件为"nndl-SemanticSegmentation-Cityscape/ENet/main.py"



```
.{SEG_ROOT}
├── tests
│   └── test_model.py
```



## References
code:

- [awesome-semantic-segmentation-pytorch](https://github.com/Tramac/awesome-semantic-segmentation-pytorch)
- [PyTorch-ENet](https://github.com/davidtvs/PyTorch-ENet)

paper:

- [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038)
- [A Deep Neural Network Architecture for Real-Time Semantic Segmentation](https://arxiv.org/pdf/1606.02147)