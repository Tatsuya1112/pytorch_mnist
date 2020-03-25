## pytorch_mnist

MNISTをpytorchを用いて分類しました。

## Requirements

torch

torchvision

## Models

### TwoLayerNet

2層の全結合層のみからなるモデル

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
