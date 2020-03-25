## pytorch_mnist

MNISTをpytorchを用いて分類しました。

## Requirements

torch

torchvision

## Models

### TwoLayerNet

2層の全結合層のみからなるモデル


```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 30, 24, 24]             780
              ReLU-2           [-1, 30, 24, 24]               0
         MaxPool2d-3           [-1, 30, 12, 12]               0
            Linear-4                  [-1, 100]         432,100
              ReLU-5                  [-1, 100]               0
            Linear-6                   [-1, 10]           1,010
================================================================
Total params: 433,890
Trainable params: 433,890
Non-trainable params: 0
----------------------------------------------------------------
```

### SimpleConvNet

1層の畳み込み層と2層の全結合層から成るモデル

## Results

| model | accuracy | cross-entropy-loss |
| ---- | ---- | ---- |
| TwoLayerNet | 0.972 | 0.103 |
| SimpleConvNet | 0.988 | 0.059 |

## References

ゼロから作るDeep Learning――Pythonで学ぶディープラーニングの理論と実装
https://www.oreilly.co.jp/books/9784873117584/
