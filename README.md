# Computer-version-toturials
## 计算机视觉资料
### 学习方法
***
个人感觉学习具有操作性的教程，最好在3天内看完:一是越往后越没兴趣，二是学习若没有及时反馈，学习的效率不高
我通常的做法是看两遍：
 * 1）第一次先过一遍目录，了解所讲内容的脉络，框架，猜测下该内容写的什么。然后后面快速的过一遍内容，
  了解基本的概念，验证下先前的猜测。比如目录中提到“图像的频域滤波”，想下‘频域’指的啥？，是图像的像素吗？
  ‘滤波’又是啥？然后在过内容时找到答案，这时候只要搞清楚基本概念和有哪些方法可以实现这功能；
 * 2）第二次过一遍内容时看看这个功能是怎么实现的，同时可以找出里面的有实际意义的源码来敲一敲。
  只需要知道有这方面的功能就行，重要的工具函数Mark一下，以后用到了再回来查，或者谷歌百度。
  
千万不要试图去记忆所有的工具函数，这种做法是十分愚蠢的！！！
 这里同样再强调一下，在入门阶段一定，一定，要尽早写程序，敲代码！！（不要复制粘贴）
***
### 基础知识
* [计算机视觉的应用场景](https://github.com/lukkyy/Computer-version-toturials/blob/master/toturials/%E5%BA%94%E7%94%A8%E5%9C%BA%E6%99%AF.md)
* [数字图像相关基础知识](https://github.com/lukkyy/Computer-version-toturials/blob/master/toturials/%E6%95%B0%E5%AD%97%E5%9B%BE%E5%83%8F%E5%9F%BA%E7%A1%80.md)
* [图像处理相关工具和软件](https://github.com/lukkyy/Computer-version-toturials/blob/master/toturials/%E5%9B%BE%E5%83%8F%E5%A4%84%E7%90%86%E5%B7%A5%E5%85%B7.md)
* [传统的图像处理的研究内容](https://github.com/lukkyy/Computer-version-toturials/blob/master/toturials/%E4%BC%A0%E7%BB%9F%E5%9B%BE%E5%83%8F%E5%A4%84%E7%90%86%E5%86%85%E5%AE%B9.md)

***
计算机视觉研究方向及区别
 1) [图像识别中，目标分割、目标识别、目标检测和目标跟踪这几个方面区别是什么？](https://www.zhihu.com/question/36500536/answer/281943900)
 2) [计算机视觉领域不同的方向：目标识别、目标检测、语义分割等](https://blog.csdn.net/u011574296/article/details/78933427)
 3) [图像分割算法及与目标检测、目标识别、目标跟踪的关系](https://blog.csdn.net/piaoxuezhong/article/details/78985024)
 4) [computer vision一些术语-目标识别、目标检测、目标分割、语义分割等](https://blog.csdn.net/tina_ttl/article/details/51915618)
 5) [CS231n第八课：目标检测定位学习记录](https://blog.csdn.net/u014696921/article/details/53791616) 
 ###  总结
 ![计算机视觉基本任务](https://github.com/lukkyy/Computer-version-toturials/blob/master/img/%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/computer_version_task.jpg)

* 图像分类（classification）：根据图像的主要内容进行分类(这是只猫)。

* 目标检测（object detection=classification+localization）：给定一幅图像，只需找到一类目标所在的矩形框：只有两个类别，即目标所在的矩形框和非目标矩形框，例如，人脸检测（人脸为目标、背景为非目标）、猫咪检测（猫咪为目标、背景为非目标））。
* 目标识别（object recognition）：与目标检测类似，给定一幅图像检测到图像中(不同之处在于要监测所有的目标)得到检测到的目标的矩形框，并对所有检测到的矩形框进行分类。


* 语义分割（semantic segmentation）：对图像中的每个像素都划分出对应的类别，即对一幅图像实现像素级别的分类。例如：两只猫咪靠在一起，能把猫咪这一类别的像素和其他类别的像素（如小狗类别）分开
* 目标分割（Object Segmentation）：和语义分割区别还不太懂
* 实例分割（instance segmentation）：是语义分割的升级版，除了能把猫和狗分割开，还能把不同个体的猫咪分割开
* 目标追踪 (Target Tracking):这个任务涉及的数据一般是时间序列，完成这个任务首先要目标定位。常见的情况是目标被定位以后，算法需要在后面的序列数据中，快速高效地对目标进行再定位。为了避免不必要的重复计算，可以充分利用时间序列的相关性，可能涉及到一些几何变换（旋转，缩放等），运动伪影处理等问题。
[计算机视觉中，目前有哪些经典的目标跟踪算法？](https://www.zhihu.com/question/26493945)

## 发展前沿
[三年来，CNN在图像分割领域经历了怎样的技术变革？](https://www.leiphone.com/news/201704/GEJU2kNeqGDpizN2.html)
[实例分割的进阶三级跳：从 Mask R-CNN 到 Hybrid Task Cascade]（https://www.leiphone.com/news/201903/CctvkMTejB1Fvgxp.html）
***
