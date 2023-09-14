# 第三章

官方教程文档地址: https://www.yuque.com/u1507140/vslam-hmh/pn5az0nwigy51f25

四元数与三维旋转 (From Krasjet): https://docs-xy.oss-cn-shanghai.aliyuncs.com/quaternion.pdf

四元数可视化: https://eater.net/quaternions

李群和李代数: https://zhuanlan.zhihu.com/p/33156814

欧拉角万向节死锁: https://zhuanlan.zhihu.com/p/344050856

Mahony姿态解算算法笔记: https://www.cnblogs.com/WangHongxi/p/12357230.html

> 
> 笔记暂略 待后补 
> 

```python
import numpy as np
from scipy.spatial.transform import Rotation

# 定义姿态变换参数: 在这里定义姿态变换参数，例如旋转矩阵R或欧拉角yaw/pitch/roll等 
# 以下是一个简单的旋转矩阵示例
# 对每个人体姿势数据进行姿态变换，以单个为例
q0, qx, qy, qz = [0.5834008455276489, -0.7853162288665771, 0.14548823237419128, -0.14749594032764435]

R_input = Rotation.from_quat([qx, qy, qz, q0]).as_matrix()
R_new = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])
R_output = np.dot(R_new, R_input)
q = Rotation.from_matrix(R_output).as_quat()

# 平移向量的修改
translation = [2044.45849609375, 4935.1171875, 1481.2275390625]
translation_new = np.array([0, 0, 1])
translation += translation_new

#输出修改后的数据：
print(q)
print(translation)
```