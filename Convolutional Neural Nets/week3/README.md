

# 吴恩达深度学习第三周


这一周主要实现了YOLO目标检测，这是一种一步式目标检测算法，所谓一步式就是原图片只向前传播一次，不用第二次，和region proposal不同，
那个称之为两步式，一步式速度更快，但是精度不足，两步式相反，速度慢但是精度高，而YOLO属于精度和速度都不错的算法。
YOLOV2甚至获得了CVPR2017最佳论文鼓励奖。

## 效果展示

再看具体之前，先看实现的效果。

![](https://github.com/cryer/Coursera_deep_learning/raw/master/Convolutional%20Neural%20Nets/week3/out/test.jpg)

![](https://github.com/cryer/Coursera_deep_learning/raw/master/image/54.png)

![](https://github.com/cryer/Coursera_deep_learning/raw/master/image/55.png)


## 关键Slide

给出吴恩达课程上的关键Slide，具体原理不多解释，可以参考课程视频和相关论文。

![](https://github.com/cryer/Coursera_deep_learning/raw/master/image/42.png)

![](https://github.com/cryer/Coursera_deep_learning/raw/master/image/43.png)

![](https://github.com/cryer/Coursera_deep_learning/raw/master/image/44.png)

![](https://github.com/cryer/Coursera_deep_learning/raw/master/image/45.png)

![](https://github.com/cryer/Coursera_deep_learning/raw/master/image/46.png)

![](https://github.com/cryer/Coursera_deep_learning/raw/master/image/47.png)

![](https://github.com/cryer/Coursera_deep_learning/raw/master/image/48.png)

![](https://github.com/cryer/Coursera_deep_learning/raw/master/image/49.png)

![](https://github.com/cryer/Coursera_deep_learning/raw/master/image/50.png)

![](https://github.com/cryer/Coursera_deep_learning/raw/master/image/51.png)

## 编程思路

![](https://github.com/cryer/Coursera_deep_learning/raw/master/image/53.png)

