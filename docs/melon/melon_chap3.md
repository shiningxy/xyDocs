# 西瓜书第三章 + 南瓜书第三章

## 线性回归

对离散属性，若属性值间存在“序”关系，可以通过连续化将其转化为连续值，例如三值属性“高度”的取值“高”，“中”，“低”可转化为{1.0,0.5,0.0}。

若属性间不存在序关系，假定有$k$个属性值，通常转化为$k$维向量，例如属性“瓜类”的取值“西瓜”，“南瓜”，“黄瓜”可转化为(0,0,1),(0,1,0),(1,0,0).

若将无序属性连续化，则会不恰当地引入序关系，对后续处理如距离计算等造成误导。

> 离散属性的向量化可以使用sklearn中 one-hot 独热编码等方法来实现。

> 对于离散属性的连续化问题，可以使用pandas中的 as_unordered 和 reorder_categories  来实现有序类别和无序类别的互相转化。先用 s.cat.as_ordered() 将数据列转化为有序类别，再利用 reorder_categories 进行具体的相对大小调整。由此完成了离散属性序的建立，进一步可以实现分类变量的比较。

线性回归试图学得

$$f(x_i) = \omega x_i+b 使得f(x_i)\simeq y_i$$

衡量$f(x)$与$y$之间的差别是确定$\omega$和$b$的关键，均方误差是回归任务中最常用的性能指标。使均方误差最小化：

> 均方误差也称平方损失 (square loss)

$$(\omega^*,b^*) = \argmin_{(\omega,b)} \sum_{i=1}^{m}(f(x_i)-y_i)^2 \\
            = \argmin_{(\omega,b)} \sum_{i=1}^{m}(y_i-\omega x_i - b)^2 $$


> $\omega^*$ 和 $b^*$ 表示 $\omega$ 和 $b$ 的解

它对应了常用的欧几里得距离或简称“欧氏距离”。基于均方误差最小化来进行模型求解的方法称为“最小二乘法”

最小二乘法就是试图找到一条直线，使所有样本到直线上的欧氏距离最小。

求解$\omega$ 和 $b$ ，使 $ E_{(\omega,b)}= \sum_{i=1}^{m} (y_i - \omega x_i - b)^2 $ 最小化的过程，称为线性回归模型的最小二乘“参数估计”(parameter estimation)。我们可将$E_{(\omega,b)}$分别对$\omega$和$b$求导，得到

$$\frac{\partial E_{(\omega,b)}}{\partial \omega} = 2 (\omega \sum_{i=1}^{m} x_i^2 - \sum_{i=1}^{m} (y_i - b)x_i)$$


$$\frac{\partial E_{(\omega,b)}}{\partial \omega} = 2 (mb-\sum_{i=1}^{m}(y_i -\omega x_i))$$

令上两式为零可得到$\omega$ 和 $b$最优解的闭式(closed-form)解

$$\omega = \frac{\sum_{i=1}^{m} y_i(x_i-\bar{x})}{\sum_{i=1}^{m}x_i^2-\frac{1}{m}(\sum_{i=1}^{m} x_i)^2}$$

$$b = \frac{1}{m}\sum_{i=1}^{m}(y_i-\omega x_i)$$

其中，$\bar{x}=\frac{1}{m}\sum_{i=1}^{m}x_i$ 为 $x$的均值



