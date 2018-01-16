

# 第一专题最后一周笔记

主要实现了深度神经网络的搭建，当然是最基础的全连接神经网络，还可以实现自定义层数和网络神经元的数量。
向前传播需要计算输出的同时，保存相关信息，因为反向传播时需要用到。如下图所示

![](https://github.com/cryer/Coursera_deep_learning/raw/master/image/5.png)

![](https://github.com/cryer/Coursera_deep_learning/raw/master/image/6.png)


正向传播和反向传播的大致代码结构如下，比较简单：

![](https://github.com/cryer/Coursera_deep_learning/raw/master/image/7.png)

![](https://github.com/cryer/Coursera_deep_learning/raw/master/image/8.png)
