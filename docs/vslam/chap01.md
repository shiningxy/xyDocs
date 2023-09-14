# 第一章

官方教程文档地址: https://www.yuque.com/u1507140/vslam-hmh/ogc8v31hbzb6efy8

仅罗列部分内容或代码实现，详细知识点参考官方教程

## 相机内参标定

OpenCV是一款广泛使用的计算机视觉库，其中包含相机内参标定的相关函数。在OpenCV中，使用calibrateCamera函数进行相机内参标定，该函数使用棋盘格等标定板，通过对标定板拍摄的多幅图像进行处理，得出相机的内参参数。OpenCV还提供了相关的可视化工具，如drawChessboardCorners函数，用于显示标定板的角点，以及projectPoints函数，用于将3D点投影到2D图像平面上。

1. 循环读取图片
2. 使用findChessboardCorners函数检测角点（需提前输入角点数）
3. 使用find4QuadCornerSubpix函数对角点进行亚像素精确化
4. 可用drawChessboardCorners将角点显示
5. 根据角点数和尺寸创建一个理想的棋盘格（用point向量存储所有理论上的角点坐标）
6. 通过calibrateCamera函数由理想坐标和实际图像坐标进行标定，可得到标定结果
7. 由projectPoints函数计算反向投影误差

## 代码

完整代码&注释：
```python
import numpy as np
import cv2
import requests

# 步骤0: 下载一个标定板图片

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

# 步骤1: 循环读取图片
image_paths = ["example.jpg"]  # 替换为你的图像文件路径列表
images = []
for path in image_paths:
    img = cv2.imread(path)
    images.append(img)

# 步骤2: 使用findChessboardCorners函数检测角点
# 提前输入角点数和标定板尺寸
pattern_size = (7, 7)  # 标定板内角点数目
obj_points = []  # 存储标定板上的理论坐标
img_points = []  # 存储实际图像上的角点坐标

for image in images:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    print("findChessboardCorners函数检测到的角点:",corners.shape)
    if ret:
        obj_points.append(np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32))
        obj_points[-1][:, :2] = np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(-1, 2)
        img_points.append(corners)
        # 在图像上绘制角点
        cv2.drawChessboardCorners(image, pattern_size, corners, ret)
        # 保存标定角点后的图像
        cv2.imwrite("calibrated_" + path, image)
    else:
        print(f"Failed to detect corners in image {image}")

# 步骤3: 使用find4QuadCornerSubpix函数对角点进行亚像素精确化
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
for i in range(len(images)):
    cv2.cornerSubPix(gray, img_points[i], (11, 11), (-1, -1), criteria)

# 步骤4: 可用drawChessboardCorners将角点显示
for i in range(len(images)):
    cv2.drawChessboardCorners(images[i], pattern_size, img_points[i], True)

# 步骤5: 根据角点数和尺寸创建一个理想的棋盘格
point_size = pattern_size  # 标定板内角点数目
square_size = 1.0  # 标定板方格的尺寸（单位可以是任意的，但需要与图像坐标对应）
objp = np.zeros((point_size[0] * point_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0 : point_size[0], 0 : point_size[1]].T.reshape(-1, 2) * square_size

# 步骤6: 通过calibrateCamera函数进行相机标定
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

# 步骤7: 由projectPoints函数计算反向投影误差
mean_error = 0
for i in range(len(obj_points)):
    img_points2, _ = cv2.projectPoints(objp, rvecs[i], tvecs[i], mtx, dist)
    print(img_points[i].shape, img_points2.shape)
    error = cv2.norm(img_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
    mean_error += error

print("Mean error: ", mean_error)

# 打印相机内参数（矩阵）和畸变参数
print("Camera Matrix:")
print(mtx)
print("\nDistortion Coefficients:")
print(dist)

# 实时显示标定后的图像
for image in images:
    cv2.imshow("Calibrated Image", image)
    cv2.waitKey(0)

# 销毁所有打开的窗口
cv2.destroyAllWindows()
```

## 代码运行结果

原始图像

![example_image](https://docs-xy.oss-cn-shanghai.aliyuncs.com/vslam-chapter1/example.jpg)

---

灰度图像

![gray_example_image](https://docs-xy.oss-cn-shanghai.aliyuncs.com/vslam-chapter1/gray_example.jpg)

---

标定后图像

![calibrated_example_image](https://docs-xy.oss-cn-shanghai.aliyuncs.com/vslam-chapter1/calibrated_example.jpg)



# 思考：

## 相机内参的物理意义

相机内参是相机标定过程中获得的一组参数，描述了相机的内部特性和成像过程。这些参数包括：

a. 焦距 (focal length)：焦距是指相机的透镜将光线聚焦到成像平面的距离。较长的焦距表示更窄的视野，较短的焦距表示更广的视野。

b. 主点 (principal point)：主点是成像平面上的点，它是光轴与成像平面的交点。主点的位置描述了图像中心的偏移。

c. 像素尺度因子 (pixel scale factor)：它定义了成像平面上的像素尺寸，通常以像素/米或像素/毫米为单位。它与焦距和成像平面尺寸有关。

d. 图像畸变参数 (image distortion parameters)：这些参数描述了透镜畸变，如径向畸变和切向畸变，它们影响图像中的像素位置。

如果一部相机的分辨率变为原来的两倍而其他内参不变，那么内参将保持不变。这意味着焦距、主点、像素尺度因子和图像畸变参数都不会改变。只有图像的分辨率发生了变化，也就是成像平面上的像素数量增加了，但每个像素的物理尺寸和相对位置与之前相同。

## 全局快门 (Global Shutter) 相机和卷帘快门 (Rolling Shutter) 相机的异同以及它们在SLAM中的优缺点

全局快门相机：

> 异同：全局快门相机在捕捉图像时同时曝光整个成像传感器的所有像素。它们在某一瞬间捕捉整个图像，因此不会出现图像扭曲或畸变问题。这使得全局快门相机适用于快速移动的场景和需要高精度时间同步的应用。

> 优点：减少了图像畸变和扭曲，适合高速运动捕捉，提供了更准确的时间信息，有助于SLAM系统的鲁棒性。

> 缺点：相机传感器需要更大的内存缓冲区来一次性保存整个图像，这可能会导致成本上升。另外，在相同分辨率下，全局快门相机通常需要更多的光，因为所有像素同时曝光，这可能在低光条件下表现不佳。

卷帘快门相机：

> 异同：卷帘快门相机在捕捉图像时逐行或逐列曝光成像传感器，因此图像捕捉的时间会有微小的时间差，导致快速运动或快速变化的场景中可能出现图像扭曲或滚动快门效应。

> 优点：相对于全局快门相机，卷帘快门相机通常更经济实惠，因为它们不需要大内存缓冲区。它们适用于大多数一般场景，可以在成本更低的情况下提供图像捕捉功能。

> 缺点：在快速移动场景下，图像可能会出现扭曲或滚动快门效应，这对SLAM系统可能会造成定位和地图构建的困难。

在SLAM中，选择全局快门还是卷帘快门相机取决于具体的应用需求。如果需要高精度的时间同步、快速运动捕捉和最小化图像畸变，全局快门相机可能更合适。然而，如果成本和性能平衡更重要，或者应用场景中没有快速运动，卷帘快门相机可能是一个更经济实惠的选择。综合考虑SLAM系统的需求和可用资源，可以选择合适的相机类型。