# Computer-version-toturials
计算机视觉资料

#### 什么是数字图像？
简单地来说，数字图像就是能够在计算机上显示和处理的图像，可根据其特性分为两大类——位图和矢量图。位图通常使用数字阵列来表示，常见格式有 BMP、JPG、GIF 等；矢量图由矢量数据库表示，接触最多的就是 PNG 图像。

#### 什么是位图、矢量图？
1 位图就是点阵图，比如大小是1024*768的图片，就是有1024*768个像素点，存储每个像素点的颜色值。
矢量图是用算法来描述图形，比如一个圆，存储半径和圆心，当然还有颜色，具体到图片的某个像素点的位置是什么颜色是根据算法算出来的。
处理位图时要着重考虑分辨率

2 矢量图是可以任意缩放的，比如上面说的圆，在1024 * 768 的时候是一个圆，图片放大20倍看还是圆，如果是位图，放大20倍，看到的就是一个一个的方块了。（矢量图是用算法来描述图形，比如一个圆，存储半径和圆心，当然还有颜色，具体到图片的某个像素点的位置是什么颜色是根据算法算出来的。）
数字摄像机或数字照相机得到的图像都是位图图像

#### 成像原理
光进入相机镜头，光电感应装置将光信号转换为电信号，量化电脉冲信号，记录为一个像素值。传感器响应函数设计为，要使光电感应装置产生这个电脉冲信号，光子强度必须达到一个阈值。进入镜头的光子数量取决于:相机的感受野大小，镜头能通过的光子。多光谱图像要分出多个波段，镜头会分光，红滤镜只过红光，蓝滤镜只通过蓝光，假设相同的光打到全色与多光谱镜头上，显然因为滤光的缘故，多光谱感光器接收到的光子要少于全色感光器。而这些光子已经足够全色产生电脉冲，却不够多光谱产生电脉冲，这时，为了接收到更多的光子，多光谱相机需要更大的感受野。也就是说，全色看了一眼北京市，就吃够了光子，多光谱需要看一遍河北省，才能吃的和全色一样饱。后面接收光子的底片一样大，也就是说将北京市和河北省画到同样大小的一张纸上且占满整张纸，显然北京市的一张纸细节要多的多，而河北省的红绿蓝三张纸却一片模糊。

#### 按光谱波长划分图像类型(偏遥感)
* RGB图像
可见普通的可见光相机只记录了2/3/4即红绿蓝三个波段的信息，其他波段就都丢掉了，所以我们会看到RGB图像就有3个通道。因为只记录了3个信号，所以也就没有办法根据3个通道的信息去恢复其他丢失的光谱信息。
* 全色图像
全色图像是单通道的，其中全色是指全部可见光波段0.38~0.76um，全色图像为这一波段范围的混合图像。因为是单波段，所以在图上显示为灰度图片。全色遥感图像一般空间分辨率高，但无法显示地物色彩，也就是图像的光谱信息少。 实际操作中，我们经常将全色图像与多波段图像融合处理，得到既有全色图像的高分辨率，又有多波段图像的彩色信息的图像。

* 多光谱图像
多光谱图像其实可以看做是高光谱的一种情况，即成像的波段数量比高光谱图像少，一般只有几个到十几个。由于光谱信息其实也就对应了色彩信息，所以多波段遥感图像可以得到地物的色彩信息，但是空间分辨率较低。更进一步，光谱通道越多，其分辨物体的能力就越强，即光谱分辨率越高。

* 高光谱图像
高光谱则是由很多通道组成的图像，具体有多少个通道，这需要看传感器的波长分辨率，每一个通道捕捉指定波长的光。把光谱想象成一条直线，由于波长分辨率的存在，每隔一定距离才能“看到”一个波长。“看到”这个波长就可以收集这个波长及其附近一个小范围的波段对应的信息，形成一个通道。也就是一个波段对应一个通道。注意对图中土壤的高光谱图像，如果我们沿着红线的方向，即对高光谱上某一个像素的采样，就可以针对此像素生成一个“光谱特征”

所谓高光谱图像就是在光谱的维度进行了细致的分割，不仅仅是传统的黑，白，或者R、G、B的区别，而是在光谱维度上也有N个通道，例如：我们可以把400nm-1000nm分为300个通道，一次，通过高光谱设备获取的是一个数据立方，不仅有图像的信息，并且在光谱维度上进行展开，结果不仅可以获得图像上每个点的光谱数据，还可以获得任意一个谱段的影像信息。
* 高光谱分类原理
不同物质在不同波段光谱信号下的不同表现，可以绘制成一条关于光谱波段和光谱值之间的曲线，根据曲线的差异，我们可以高光谱图像中不同物质进行分类

#### 按像素维度划分图像类型
* 二值图像（黑白图像）
二值图像(Binary Image)，是指将图像上的每一个像素只有两种可能的取值或灰度等级状态，只有两个值，0和1，0代表黑，1代表白。
保存方式也相对简单，每个像素只需要1Bit就可以完整存储信息。如果把每个像素看成随机变量，一共有N个像素，那么二值图有2的N次方种变化。

* 灰度图像
与二值图像一样，灰度图只包含一个（维度）通道的信息。但灰度图像与二值图像不同，二值图像只有黑色与白色两种颜色；灰度图像在黑色与白色之间还有许多级的颜色深度。灰度图像经常是在单个电磁波频谱如可见光内测量每个像素的亮度得到的，每个像素用8位的2进制的数字来表示，这样可以有256（2^8）级灰度。

* RGB图像
彩色图通常包含三个通道（RGB）的信息，其中的每个彩色像素点都有三个通道分别对应的红、绿、蓝三个分量。每个通道用8位的2进制的数字来表示。

* 高光谱图像
可理解为有很多个通道的图像

#### 按编码方法划分图像类型
* RGB
位图颜色的一种编码方法，用红、绿、蓝三原色的光学强度来表示一种颜色。这是最常见的位图编码方法，可以直接用于屏幕显示。
* CMYK
位图颜色的一种编码方法，用青、品红、黄、黑四种颜料含量来表示一种颜色。常用于彩色印刷。
