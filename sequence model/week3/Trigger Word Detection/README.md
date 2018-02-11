

# Trigger Word Detection

触发字检测，最后一次作业完成了一个语音识别的小项目，也很有意思，不需要其他语音识别那样庞大的数据就可以完成，

实现识别出关键字时就给出反应，就像siri，小爱同学，你好百度那样，用于唤醒智能设备。

## 模型

模型大概如下：

![](https://github.com/cryer/Coursera_deep_learning/raw/master/sequence%20model/week3/image/model.png)

注意点

* Trigger Word Detection一定不能用Bidirectional RNN，因为这样必须全部输入完成才能进行预测和识别，但是Trigger Word Detection要求及时反应。
* 语音识别任务中，语音数据处理成频谱是最正常的预处理
* 在RNN前加入1D的卷积也很不错，可以实现尺寸的控制
* Dropout层在多层RNN中建议使用
