# numpy-neural-net

Нейронная сеть с вручную выведенным градиентом и разведка данных методом главных компонент, на чистом NumPy, без фреймворков глубокого обучения.

[![Сеть в Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoonzky/numpy-neural-net/blob/main/three_neuron_net.ipynb)
[![PCA в Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoonzky/numpy-neural-net/blob/main/pca_kmeans.ipynb)

## three_neuron_net.ipynb

Сеть из трёх нейронов определяет, лежит ли точка выше прямой y = −x. Keras, TensorFlow и PyTorch по условию запрещены, разрешён только NumPy.

- Класс `SimpleNeuralNetwork` на четырёх скалярных весах: по одному на каждый нейрон первого слоя и два на выходной. Forward без матричного умножения — ровно по схеме из трёх нейронов.
- Градиент выведен вручную на бумаге и запрограммирован по формуле; сигмоида с `np.clip(z, -500, 500)` от переполнения.
- Обучающая выборка 8000 точек, валидирующая 100. Точность считается через таблицу TP, TN, FP, FN.
- Шаг обучения подобран перебором: шесть значений лямбда от 0,001 до 0,5 с выводом train accuracy, val accuracy и итогового лосса.

## pca_kmeans.ipynb

- Три двумерных облака по 400 точек: два нормальных, одно равномерное.
- Расширение до пяти признаков: x3 = x1 + x2, x4 = ln|x1| + 1 + x2, x5 = sin(x1·x2). Логарифм взят от модуля, иначе при отрицательных x1 он не определён.
- PCA снижает размерность обратно до двух, дальше k-means (`n_init='auto'`).
- Число кластеров проверяется по silhouette score перебором k от 2 до 5: оптимум совпадает с тремя исходными облаками.

## Запуск

```
pip install -r requirements.txt
jupyter notebook three_neuron_net.ipynb
```

## Стек

Python, NumPy, scikit-learn, Matplotlib.

Учебные проекты курса «Введение в разработку систем искусственного интеллекта», СПбГЭТУ «ЛЭТИ», 2025.
