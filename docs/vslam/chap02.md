# 第一章

官方教程文档地址: https://www.yuque.com/u1507140/vslam-hmh/csqub9k4nax99i19

仅罗列部分内容或代码实现，详细知识点参考官方教程

## 相机外参标定

相机外参标定是计算相机的外部参数的过程，也称为相机姿态估计。外部参数包括相机的位置（平移向量）和方向（旋转矢量或旋转矩阵）。相机外参标定的主要目标是确定相机在世界坐标系中的位置和朝向，从而将3D世界坐标映射到2D图像坐标。这对于计算机视觉任务如SLAM（Simultaneous Localization and Mapping）、3D重建、目标追踪等非常重要。

以下是相机外参标定的详细步骤和方法：

1. 准备标定板或特征点集合：

使用标定板：通常，相机外参标定的第一步是准备一个标定板，如棋盘格。这个标定板包含已知尺寸的方格，用于计算相机的内参和外参。标定板应该在多个位置和姿态下被拍摄。
使用特征点集合：如果没有标定板，您可以使用已知世界坐标的特征点集合。这些特征点可以是物体的角点、突出特征或已知3D坐标的地标点。

2. 拍摄图像：

使用相机拍摄标定板或特征点集合的图像。在不同位置和角度下拍摄多张图像是很重要的，以获得不同视角的信息。

3. 提取特征点：

对于标定板，使用图像处理技术来检测和提取每个方格角点的图像坐标。
对于特征点集合，这些点的3D坐标已知，可以直接使用。

4. 设置初始猜测：

对于每张图像，需要提供一个初始的外参猜测，通常由相机运动或已知的姿态提供。这将用作优化算法的起始点。

5. 外参标定：

使用外参标定算法来估计相机的外部参数。常见的算法包括：
PnP（Perspective-n-Point）：通过已知的3D-2D点对来估计外参，可以使用cv2.solvePnP函数实现。
PnPRansac：与PnP类似，但使用RANSAC来鲁棒地处理异常值和噪声。可以使用cv2.solvePnPRansac函数实现。
手动外参标定：在特定情况下，可以手动调整外参以获得最佳匹配。
这些算法将尝试最小化3D点与2D投影点之间的重投影误差，从而找到最佳的外部参数。

6. 重复步骤 3-5：

对于每张图像，重复步骤3至步骤5，以估计多个外部参数集合。

7. 合并结果：

将所有图像的外参结果合并，通常使用平均值或其他统计方法来获得最终的相机外参。

8. 评估标定结果：

最后，评估标定的准确性。这可以通过使用标定结果对图像中的物体进行重建，并与已知的3D世界坐标进行比较来完成。

## 重要函数

cv2.solvePnPRansac
> retval, rvec, tvec, inliers = cv2.solvePnPRansac(world_point, exact_corners, mtx, dist)
> 布尔值retval（表示成功或失败） 旋转向量（rvec） 平移向量（tvec） 内点的索引（inliers）
> 该函数用于获取相机外参，solvePnPRansac是OpenCV中用于解决PnP问题（Perspective-n-Point）的函数 它通常与RANSAC一起使用 以估计相机的位姿。
> 参考官方文档: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga50620f0e26e02caa2e9adc07b5fbf24e
> 官方文档Pose computation overview: https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html


cv2.projectPoints
> imgpts, jac = cv2.projectPoints(axis, rvec, tvec, mtx, dist)
> cv2.projectPoints是OpenCV中用于将3D点投影到2D图像坐标的函数。它可以根据相机内参、旋转矢量（rvec）、平移矢量（tvec）、畸变系数等信息，将3D空间中的点投影到图像平面上。
> imgpts：这是一个包含了投影到图像上的2D点坐标的NumPy数组。
> 具体来说，imgpts 是一个形状为 (N, 1, 2) 的数组，其中 N 是输入的3D点的数量。每个2D点坐标表示一个3D点在图像平面上的投影位置。这些坐标可以用于绘制或进行其他后续处理。
> jac：这是一个可选的输出参数，是用于计算投影的雅可比矩阵（Jacobian matrix）。
> 雅可比矩阵是一个矩阵，用于描述输入参数（在这里是 rvec 和 tvec）与输出参数（在这里是 imgpts）之间的关系。
> 雅可比矩阵通常用于优化问题，例如在相机姿态估计或结构光三维重建中。如果不需要使用雅可比矩阵，可以忽略这个输出参数。
> 参考官方文档: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga1019495a2c8d1743ed5cc23fa0daff8c

## 代码

完整代码&注释：
```python
import cv2
import numpy as np

# 在运用第一章相机内参标定的代码后 将相机内参mtx与畸变参数dist保存为npy 
# 读取相机内参 和 畸变参数
mtx = np.load("E:/code/vslam/example_mtx.npy")
dist = np.load("E:/code/vslam/example_dist.npy")

# 标定图像保存路径
photo_path = "E:/code/vslam/example.jpg"


def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
    return img

# 标定图像
def calibration_photo(photo_path):
    # 设置要标定的角点个数
    # x y方向上的角点个数
    x_nums = 7  
    y_nums = 7
    # 设置(生成)标定图在世界坐标中的坐标
    # 生成x_nums*y_nums个坐标，每个坐标包含x,y,z三个元素
    world_point = np.zeros((x_nums * y_nums, 3), np.float32)
    # mgrid[]生成包含两个二维矩阵的矩阵，每个矩阵都有x_nums列,y_nums行 转置矩阵 reshape()重新规划矩阵，但不改变矩阵元素
    world_point[:, :2] = np.mgrid[:x_nums, :y_nums].T.reshape(-1, 2)  
    # 设置世界坐标的坐标
    axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)
    # 设置角点查找限制
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    image = cv2.imread(photo_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 使用findChessboardCorners函数检测角点
    ret, corners = cv2.findChessboardCorners(
        gray,
        (x_nums, y_nums),
    )

    if ret:
        # 使用find4QuadCornerSubpix函数对角点进行亚像素精确化
        exact_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # 获取外参 solvePnPRansac是OpenCV中用于解决PnP问题（Perspective-n-Point）的函数 它通常与RANSAC一起使用 以估计相机的位姿。
        # 布尔值retval（表示成功或失败） 旋转向量（rvec） 平移向量（tvec） 内点的索引（inliers）
        retval, rvec, tvec, inliers = cv2.solvePnPRansac(world_point, exact_corners, mtx, dist)
        print('旋转向量: {} \n平移向量: {}'.format(rvec, tvec))
        # cv2.projectPoints是OpenCV中用于将3D点投影到2D图像坐标的函数。它可以根据相机内参、旋转矢量（rvec）、平移矢量（tvec）、畸变系数等信息，将3D空间中的点投影到图像平面上。
        # imgpts：这是一个包含了投影到图像上的2D点坐标的NumPy数组。
        # 具体来说，imgpts 是一个形状为 (N, 1, 2) 的数组，其中 N 是输入的3D点的数量。每个2D点坐标表示一个3D点在图像平面上的投影位置。这些坐标可以用于绘制或进行其他后续处理。
        # jac：这是一个可选的输出参数，是用于计算投影的雅可比矩阵（Jacobian matrix）。
        # 雅可比矩阵是一个矩阵，用于描述输入参数（在这里是 rvec 和 tvec）与输出参数（在这里是 imgpts）之间的关系。
        # 雅可比矩阵通常用于优化问题，例如在相机姿态估计或结构光三维重建中。如果不需要使用雅可比矩阵，可以忽略这个输出参数。
        imgpts, jac = cv2.projectPoints(axis, rvec, tvec, mtx, dist)

        # cv2.line接收的参数需要为整型
        imgpts = imgpts.astype(int)
        corners = corners.astype(int)
        # 可视化角点
        img = draw(image, corners, imgpts)
        cv2.imshow("img", img)


if __name__ == "__main__":
    calibration_photo(photo_path)
    cv2.waitKey()
    cv2.destroyAllWindows()

```

> 实测五张角度差异大的标定板图像，所计算得到的平均反向投影误差将增大，平均反向投影误差能增大至个位数。进而使用平均后的相机内参和畸变参数来估计相机位姿（标定相机外参）时，导致xyz轴偏差大。
> 
> 实测单张标定板图像，所计算得到的反向投影误差均小于0.1，图片越模糊，角度越大，误差越大。利用单张图片的相机内参和畸变参数能够准确标定相机外参，绘制所得的xyz轴也符合实际。

## 标定结果

如下为取五张图像平均后的相机内参和畸变参数来标定相机外参，与单张图像的标定结果

平均：

![avg_five_solvePnPRansac](https://docs-xy.oss-cn-shanghai.aliyuncs.com/vslam-chapter1/solvePnPRansac_five_tmp2.jpg)

单张:

![single_solvePnPRansac](https://docs-xy.oss-cn-shanghai.aliyuncs.com/vslam-chapter1/solvePnPRansac_tmp2.jpg)

# 思考

## 相机外参的作用是什么？
相机外参的作用是描述相机在世界坐标系中的位置和方向。它包括平移向量（通常表示为 t）和旋转矩阵（通常表示为 R），用于将相机坐标系中的点映射到世界坐标系中。具体来说，相机外参用于以下目的：

相机定位：确定相机相对于世界坐标系的位置，即相机在何处。
姿态估计：确定相机的朝向和方向，也就是相机的旋转角度。
坐标转换：将相机坐标系中的点映射到世界坐标系中，或者反之。

## 一个载体搭载着摄像头，在平面上运动，相机外参t(x,y,z)与R(roll,pitch,yaw)哪些量不可以被标定出来？

在一个载体上搭载着摄像头，进行平面运动时，相机外参中的平移向量 t(x, y, z) 和旋转矩阵 R(roll, pitch, yaw) 可能会受到限制或无法完全标定出来，具体取决于摄像头的配置和运动情况：

平移向量 t(x, y, z)：在一些情况下，平移向量可以被准确标定出来，尤其是在已知的基准点或参考物体之间进行了精确的距离测量的情况下。然而，在没有明确的参考物体或距离测量的情况下，平移向量可能无法准确标定，因为相机运动的距离和方向信息缺失。

旋转矩阵 R(roll, pitch, yaw)：旋转矩阵的标定也可能受到限制，特别是在存在运动模糊或没有足够的视觉特征来确定相机的方向时。此外，某些运动类型（如平移运动）可能会使相机的旋转无法唯一确定。

在实际应用中，通常会使用各种传感器和方法来辅助相机外参的标定，例如使用惯性测量单元（IMU）来获取平移和旋转信息，或使用SLAM（Simultaneous Localization and Mapping）算法来估计相机的运动和姿态。标定外参的可行性和精确性取决于具体的应用和环境条件。