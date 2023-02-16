## Task04 Grad-CAM

### Grad-CAM 论文十问

1. 论文试图解决什么问题

深度学习的可解释性分析、显著性分析

2. 这是否是一个新的问题

不是，在Grad-CAM之前，有大量对卷积神经网络学习到的特征做可视化的工作，也有CAM类激活热力图的工作

3. 这篇论文要验证一个什么科学假设

卷积神经网络能提取位置信息，并按特定类别展示出来。

4. 有哪些相关研究？如何归类？谁是这一课题在领域内值得关注的研究员？

可视化卷积神经网络学到的特征
CAM类激活热力图

5. 论文中提到的解决方案之关键是什么？

对原生CAM（类激活热力图）改进，计算“特定类别预测分数相对于最后一层卷积层输出特征图每个元素的偏导数”，进而计算特征图每个channel对模型预测为特定类别的影响程度。
无需修改模型，无需重新训练，即可对已有卷积神经网络模型绘制特定类别的Grad-CAM热力图，展示指定类别在原图上重点关注的特征区域，并用图像分类实现弱监督定位。可推广至图像分类、图像描述、视觉问答等多个任务。

6. 论文中的实验是如何设计的？

ImageNet弱监督定位任务
人工评价
图像分类、图像描述、视觉问答、DenseCap等其它视觉任务

7. 用于定量评估的数据集是什么？代码有没有开源？

ImageNet, COCO, ILSVRC13 detection val set
开源地址：https://github.com/ramprs/grad-cam/

8. 论文中的实验及结果有没有很好地支持需要验证的科学假设？

有

9. 这篇论文到底有什么贡献？

无需修改模型，无需重新训练，即可对已有卷积神经网络模型绘制特定类别的Grad-CAM热力图，展示指定类别在原图上重点关注的特征区域，并用图像分类实现弱监督定位。可推广至图像分类、图像描述、视觉问答等多个任务。

10. 下一步呢？有什么工作可以继续深入？

Grad-CAM++
Score-CAM
LayerCAM
等一系列基于CAM的工作
