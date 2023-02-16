## Task 06 LIME

### LIME的基本假设

线性模型特征需要易于理解

特征数量需要足够少，便于人类理解 

$x\in \mathbb{R}^d$表示原模型训练用到的特征，$d$维实数向量

$x'\in \{0, 1\}^{d'}$ 可解释模型训练用到的特征，$d'$维0-1向量

### trade-off

可解释性interpretability与拟合准确度fidelity的trade-off

可解释性好的模型，模型结构与原理更加简单，更易于人类理解，但是可能无法很好地拟合数据

可解释性差的模型，模型结合与原理更加复杂，不易于人类理解，但可能对数据的拟合更加准确

### LIME公式

![LIME公式](https://docs-xy.oss-cn-shanghai.aliyuncs.com/xai07.png)

### 得到可解释模型的步骤

从原始数据中提取d维可解释特征X，转换为d'维的0-1向量，并在这些0-1向量添加扰动（将部分特征的0改为1，1改为0），得到X'，在待测样本中生成邻域数据Z'，将Z'由0-1向量恢复为d维原始向量，使用原模型进行预测。用d'维的0-1向量作为特征，用原模型的预测结果作为标签，训练出一个可解释模型g。

输入d维特征，输出原模型预测结果。通过可解释模型g中的权重，分析哪些特征对模型的预测提供了较大的贡献

![步骤](https://docs-xy.oss-cn-shanghai.aliyuncs.com/xai08.png)

### Sparese Linear Explanations

使用$\pi_x(z) = exp(-D(x,z)^2/\sigma^2)$高斯核为半径，待测样本为中心画圆，计算距离来表示局部不可信度。距离越远，局部不可信度越大。

不可信度计算公式：

![不可信度计算公式](https://docs-xy.oss-cn-shanghai.aliyuncs.com/xai09.png)