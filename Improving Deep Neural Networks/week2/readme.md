

# 第二专题第二周笔记

本周的内容相对很重要，主要讲了一些权重优化方法，这些方法可以加速训练，优化结果。

* 指数权重平均，如下图：

![](https://github.com/cryer/Coursera_deep_learning/raw/master/image/9.png)

大致计算方式就如图中所说，指数权重平均主要可以平滑数据，beta的取值不同，平滑曲线不同，如下图：

![](https://github.com/cryer/Coursera_deep_learning/raw/master/image/10.png)

beta0.9时，红线，beta0.98时绿线，beta的含义其实是beta越大曲线就越向右轻微移动，beta越小，曲线越抖动，具体可以看这个式子，1/（1-beta），
这个值代表了取几个iter的平均滑动，就图中例子，就是几天的温度。比如beta0.9，式子的值就是10，也就是取10天的加权平均温度。这样变大平移，变小抖动就
很好理解了。

当然，要得到上面的曲线，其实还需要偏置修正，如下图，具体不多解释，自己计算一次就能看出：

![](https://github.com/cryer/Coursera_deep_learning/raw/master/image/11.png)

* 加入动量，在普通梯度下降中加入动量，其实就是基于指数权重平均，包括下面的RMSprop和Adam都是如此，加入动量可以加速训练，让函数收敛的快一些，而且可以有防止局部最优的效果。

![](https://github.com/cryer/Coursera_deep_learning/raw/master/image/12.png)

* RMSprop方法也就是均方根prop，是一种自适应学习速率的方法，它可以让不同的梯度维度上用不同的学习速率，变化快的维度，学习速率小一点，变化满的梯度学习速度大一点。

![](https://github.com/cryer/Coursera_deep_learning/raw/master/image/13.png)

* Adam，Adam将动量和RMSprop结合起来，效果更佳，我平时基本选择的都是Adam。具体计算如图中所示：

![](https://github.com/cryer/Coursera_deep_learning/raw/master/image/14.png)

![](https://github.com/cryer/Coursera_deep_learning/raw/master/image/15.png)
