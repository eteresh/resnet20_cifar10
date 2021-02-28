# ResNet20 on CIFAR10
В этих скриптах реализована модель ResNet20, которая обучается с нуля до 90.3% точности на датасете CIFAR10.

## Структура файлов
* В файле [net.py](net.py) реализована сама модель ResNet20
* В файл [train.py](train.py) в функцию `train_epoch`, вынесен код обучения одной эпохи.
* Все эпохи обучаются в jupyter ноутбуке [baseline.ipynb](baseline.ipynb).
* Обученная модель сохранена  в файле [res_net_20.model](res_net_20.model).
