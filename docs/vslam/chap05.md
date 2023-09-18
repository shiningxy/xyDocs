# 第五章

官方教程文档: https://www.yuque.com/u1507140/vslam-hmh/rcvyw38lhgchkb6g


# 思考题

1.请说说SIFT或SURF的原理，并对比它们与ORB之间的优劣。

* SIFT原理：
    * SIFT算法首先通过高斯滤波器构建多尺度图像金字塔，然后在不同尺度下使用高斯差分来检测潜在的关键点。接着，对每个关键点周围的局部区域计算梯度和梯度方向，生成特征向量。最后，对这些特征向量进行方向和尺度不变性的处理，以获得稳定的特征描述子。

* SURF原理：
    * SURF算法也使用了尺度空间的概念，但它使用了一种称为积分图像的数据结构来加速计算。SURF检测关键点的方式是通过检测图像中的兴趣点，然后在兴趣点周围计算方向和大小不变的特征描述子。

* ORB原理：
    * ORB（Oriented FAST and Rotated BRIEF）是一种更快速的特征提取算法，它结合了FAST关键点检测和BRIEF描述子。FAST用于检测关键点，然后使用BRIEF描述子来描述这些关键点。ORB还具有方向不变性，但它通常在速度和计算资源效率方面优于SIFT和SURF。

优劣比较：

* SIFT和SURF在精度和稳定性上表现良好，对旋转、尺度变化和部分遮挡具有较好的鲁棒性。但它们通常比ORB更慢，尤其是SIFT。
* ORB速度更快，适用于实时性要求较高的应用。然而，它可能对视角变化和光照变化敏感，不如SIFT和SURF稳定。

2.我们发现，OpenCV提供的ORB特征点在图像中分布不够均匀。你是否能够找到或提出让特征点分布更均匀的方法？

* 关键点采样策略：可以尝试修改关键点检测参数，例如阈值或最小距离，以控制关键点的密度。减小最小距离可以增加关键点密度，而增加阈值可以减少关键点数量。

* 区域采样：将图像分成多个区域，并在每个区域内独立地检测关键点。这可以确保在整个图像中有更均匀的特征点分布。

* 动态调整参数：可以根据图像的内容和分布情况动态调整关键点检测参数，以获得更好的均匀性。

* 布局约束：根据应用的需求，可以添加布局约束，例如在特定区域内或特定方向上检测关键点，以确保关键点的均匀分布。


## 代码

SIFT python实现&注释：

```python
import numpy as np
import cv2
import requests

# 下载一个标定板图片

# 图片的URL链接
image_url = "https://docs-xy.oss-cn-shanghai.aliyuncs.com/vslam-chapter1/example.jpg"

# 发送HTTP请求获取图片内容
response = requests.get(image_url)

# 检查是否成功获取图片
if response.status_code == 200:
    # 指定本地文件路径来保存图片
    local_file_path = "example.jpg"

    # 以二进制写入模式打开本地文件
    with open(local_file_path, "wb") as f:
        # 将获取到的图片内容写入本地文件
        f.write(response.content)
    print(f"Image downloaded and saved to {local_file_path}")
else:
    print(f"Failed to download image. Status code: {response.status_code}")


# 读取图像
image = cv2.imread('example.jpg', cv2.IMREAD_GRAYSCALE)

# 初始化SIFT检测器
sift = cv2.SIFT_create()

# 在图像上检测关键点
keypoints = sift.detect(image, None)

# 计算关键点的SIFT描述子
keypoints, descriptors = sift.compute(image, keypoints)

# 在图像上绘制关键点
output_image = cv2.drawKeypoints(image, keypoints, None)

# 保存结果图像
cv2.imwrite('sift_example.jpg', output_image)
```
原始图像

![原始图像](https://docs-xy.oss-cn-shanghai.aliyuncs.com/vslam-chatper5/example.jpg)


SIFT检测结果

![SIFT检测结果](https://docs-xy.oss-cn-shanghai.aliyuncs.com/vslam-chatper5/sift_example.jpg)




ORB python实现&注释：

```python
import numpy as np
import cv2
import requests

# 下载一个标定板图片

# 图片的URL链接
image_url = "https://docs-xy.oss-cn-shanghai.aliyuncs.com/vslam-chapter1/example.jpg"

# 发送HTTP请求获取图片内容
response = requests.get(image_url)

# 检查是否成功获取图片
if response.status_code == 200:
    # 指定本地文件路径来保存图片
    local_file_path = "example.jpg"

    # 以二进制写入模式打开本地文件
    with open(local_file_path, "wb") as f:
        # 将获取到的图片内容写入本地文件
        f.write(response.content)
    print(f"Image downloaded and saved to {local_file_path}")
else:
    print(f"Failed to download image. Status code: {response.status_code}")

# 读取图像
image = cv2.imread('example.jpg', cv2.IMREAD_GRAYSCALE)

# 初始化ORB检测器
orb = cv2.ORB_create()

# 在图像上检测关键点和计算ORB描述子
keypoints, descriptors = orb.detectAndCompute(image, None)

# 在图像上绘制关键点
output_image = cv2.drawKeypoints(image, keypoints, None)

# 保存结果图像
cv2.imwrite('orb_example.jpg', output_image)
```

ORB检测结果

![ORB检测结果](https://docs-xy.oss-cn-shanghai.aliyuncs.com/vslam-chatper5/orb_example.jpg)

