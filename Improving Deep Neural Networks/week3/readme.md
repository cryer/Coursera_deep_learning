

# 第二专题 第三周笔记

本周讲到了一个很重要的方法，Batch Norm，批归一层，这个方法现在几乎每一个著名的网络结构里都有，代替了曾经Alex等网络的LRN层，即局部响应归一层。
BN层的主要作用就是加快网络训练速度，因为每次在数据进入下一层时都经过一个归一化过程，包括均值和方差的归一，这使得网络权重的收敛很快，对每一层来说都是如
此，此外，BN层还有轻微的正则化效果，但是并不理想，因此需要额外的正则化，这里Andrew Ng说一般结合BN和Dropout，但是最近的研究其实表明这两者结合起来反而会
降低效果，因为这两者之间有一定的冲突，想具体了解可以去查看相关论文。

BN层还有一个好处，就是使得你的网络对一些超参数，包括学习速率等不是那么敏感，你可以不用
像以前那样慢慢调整超参数，而是可以直接给定一个较大的学习速率，也没有关系。反而可以加快训练。

下面给出Ng课程中的关键Slides。

![](https://github.com/cryer/Coursera_deep_learning/raw/master/image/16.png)

![](https://github.com/cryer/Coursera_deep_learning/raw/master/image/17.png)

![](https://github.com/cryer/Coursera_deep_learning/raw/master/image/18.png)


## TensorFlow

另外本周介绍了深度学习框架Tensorflow的大概使用方法，作业也是熟悉TF的相关用法，直接给出我的代码，不多介绍，这部分是TF的基础，没有难度，第一次接触可能需
要多看看。

