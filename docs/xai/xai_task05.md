## Task05 CAM Captum代码实战

同济子豪兄 - XAI - [github项目链接](https://github.com/TommyZihao/Train_Custom_Dataset/tree/main/%E5%9B%BE%E5%83%8F%E5%88%86%E7%B1%BB/6-%E5%8F%AF%E8%A7%A3%E9%87%8A%E6%80%A7%E5%88%86%E6%9E%90%E3%80%81%E6%98%BE%E8%91%97%E6%80%A7%E5%88%86%E6%9E%90)【代码以Linux命令行为主，最好在云算力平台上运行】

CAM的可视化主要试用【torchcam】，notebook中的安装代码：

```shell
!git clone https://github.com/frgfm/torch-cam.git
!pip install -e torch-cam/.
```

#### 图像中只有一个类别时

![一个类别](https://docs-xy.oss-cn-shanghai.aliyuncs.com/xai02.jpg)

#### 图像中含有两个类别时
<div style="display:flex">
    <img src="https://docs-xy.oss-cn-shanghai.aliyuncs.com/xai03.jpg" style="width:50%">
    <img src="https://docs-xy.oss-cn-shanghai.aliyuncs.com/xai04.jpg" style="width:50%">
</div>

#### 可解释分析热力图 与 原图结合的结果

```python
from torchcam.utils import overlay_mask
result = overlay_mask(img_pil, Image.fromarray(activation_map), alpha=0.7)
```
<div style="display:flex">
    <img src="https://docs-xy.oss-cn-shanghai.aliyuncs.com/xai05.png" style="width:40%">
    <img src="https://docs-xy.oss-cn-shanghai.aliyuncs.com/xai06.jfif" style="width:50%">
</div>

> torchcam.utils.overlay_mask(img: Image, mask: Image, colormap: str = 'jet', alpha: float = 0.7) → Image [[source]](https://frgfm.github.io/torch-cam/_modules/torchcam/utils.html#overlay_mask)
> ```python
> from PIL import Image
> import matplotlib.pyplot as plt
> from torchcam.utils import overlay_mask
> img = ...
> cam = ...
> overlay = overlay_mask(img, cam)
> ```
> PARAMETERS:
> - img – background image
> - mask – mask to be overlayed in grayscale
> - colormap – colormap to be applied on the mask
> - alpha – transparency of the background image
>
> RETURNS:
>  - overlayed image
>
> RAISES:
>  - TypeError – when the arguments have invalid types
>  - ValueError – when the alpha argument has an incorrect value
>
