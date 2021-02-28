# ResNet20 on CIFAR10
В этих скриптах реализована модель ResNet20, которая обучается с нуля до 90.3% точности на датасете CIFAR10.

## Структура файлов
* В файле [net.py](net.py) реализована сама модель ResNet20
* В файл [train.py](train.py) в функцию `train_epoch`, вынесен код обучения одной эпохи.
* Все эпохи обучаются в jupyter ноутбуке [baseline.ipynb](baseline.ipynb).
* Обученная модель сохранена  в файле [res_net_20.model](res_net_20.model).
* Filter-level pruning в ноутбуке [prune.ipynb](prune.ipynb).
---
Чтобы вычислить ошибку сохраненной модели, достаточно на машине с доступной **cuda** перейти в корень репозитория и выполнить в питоне:
```python
import torch

from net import ResNet20
from loader import testloader
from train import get_accuracy

res_net = ResNet20()
res_net.load_state_dict(torch.load('./res_net_20.model'))
res_net.cuda()
print(f'res_net_20.model has test dataset accuracy: {get_accuracy(res_net, testloader)}')
```
