## 一、Pytorch环境搭建
    1.进入pytorch首页：https://pytorch.org/ 选择相应安装包的安装命令
    2.在安装过程中，若因为包太大而下载很慢的情况，可选择离线安装，具体参考https://blog.csdn.net/qimo601/article/details/106528982


## 二、Faster R-CNN
    2015年提出的目标检测算法，历经三个版本
    
    1. R-CNN
        1） 给定Input Image，由Selective Search算法生成大约2000个Region Proposals ———— 此操作耗时较大，且在CPU上执行，一张图片平均耗时2秒
        2） 对这近2000个Region Proposals，分别送到CNN中计算
        3） 对计算的结果做Bounding-box Regression和SVM（关于Bbox reg，参考https://blog.csdn.net/zijin0802034/article/details/77685438）
        
        存在的问题：
        1） 每个Region Proposal并不共享CNN，具体来说，由于每个RP的大小不一致，导致CNN的FC输入维度不同，由此导致每个RP都需使用各自的CNN，即每个RP的CNN不共享
        2） 训练耗时太长 —— 84h
        3） 检测速度慢，平均检测一张图片，耗时49s（生成RPs:2s， 目标检测:47s）
        
    1.1 R-CNN (SPP-Net)
        1）针对问题1），在最后一个Conv后、FC前引入SPP（Spatial Pyramid Pooling）（参考 https://www.cnblogs.com/zongfa/p/9076311.html），
          使得对于不同尺度的RP，在经过SPP后，能够得到相同维度的特征，由此所有的RP即可共享CNN
          
        提升：
        1） 训练时间：25.5h
        2） 平均检测一张图片，耗时4.3s（生成RPs:2s， 目标检测:2.3s）
    
    2. Fast R-CNN
        1） 给定Input Image，先对整张图做卷积，得到特征图 ———— 卷积共享，速度更快
        2） 计算初始Input Image上约2000个RPs在特征图上对应的位置，即RoIs(Regions of Interest)，但每个RoI尺度不同
        3） 引入RoI Pooling层，类似于SPP，将不同尺度的RoI池化为相同维度的特征（参考 https://blog.csdn.net/flyfor2013/article/details/108138643）
        4） 在相同维度下，可共享FCs
        5） 在FCs后做Bounding-box Regression和Softmax Classifier
        
        提升：
        1） 训练时间显著缩短 —— 8.75h
        2） 平均检测一张图片，耗时2.32s（生成RPs:2s， 目标检测:0.32s）
        
        存在的问题：
        1） RoIs是由原始图像中的RP计算而得，而RPs由Selective Search算法计算而得，该算法耗时较长
        2） 从提升2）耗时中可以看出，运行时间主要在生成RPs这一步，且该计算实在CPU中，无法在GPU中计算
    
    3. Faster R-CNN ———— Make CNN do proposals!
        1)  给定Input Image，对整张图做CNN，得到特征图
        2） 引入RPN（Region Proposal Network），计算特征图中的RoIs ———— 核心：RPN
        3） 4个损失函数：
            3.1） RPN Classification loss: 2分类，区分是否是物体，即前景还是背景
            3.2） RPN Bounding-box Regression loss
            3.3） Final Classification loss: 多分类，即最终的分类器
            3.4） Final Bounding-box Regression loss 
        4） 对生成的RoIs做后续处理（类似于Fast R-CNN）
        
        提升:
        1） 卷积共享
        2） RoIs在CNN中生成，真正意义上实现了end-end
        3） 训练时间显著提升
        4） 平均检测一张图片，耗时0.2s（不再额外耗时生成RPs）