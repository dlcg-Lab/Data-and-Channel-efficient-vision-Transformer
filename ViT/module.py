#!/usr/bin/env python
# coding: utf-8

# # 深入理解图像分类中的Transformer-Vit,DeiT
# 
# 前面课程中，我们了解到 Transformer 结构已经在 NLP 领域中得到了广泛应用。从本节课开始，将带大家学习了解 Transformer 结构在 CV 领域的具体应用。本次课将为大家详细介绍 Transformer 在 CV 领域中的两个经典算法：ViT 以及 DeiT。我们会详细地讲解 ViT 和 DeiT 的算法背景，网络结构以及实现细节。同时，也会基于 paddlepaddle2.1 实现完整的网络结构代码并附加详细注释。最后，我们会在 ImageNet 数据集上评估两个算法的具体效果。
# 
# ## 1.资源
# **⭐ ⭐ ⭐ 欢迎点个小小的[Star](https://github.com/PaddlePaddle/awesome-DeepLearning/stargazers)支持！⭐ ⭐ ⭐**
# 开源不易，希望大家多多支持~ 
# <center><img src='https://ai-studio-static-online.cdn.bcebos.com/c0fc093bffd84dc8920b33e8bf445bb0e842bc9fc29047878df03eb84691f0bf' width='700'></center>
# 
# * 更多CV和NLP中的transformer模型(BERT、ERNIE、Swin Transformer、DETR等)，请参考：[awesome-DeepLearning](https://github.com/paddlepaddle/awesome-DeepLearning)
# 
# 
# * 更多图像分类模型(ResNet_vd系列、MobileNet_v3等)，请参考：[PaddleClas](https://github.com/PaddlePaddle/PaddleClas)
# 
# * 更多学习资料请参阅[飞桨深度学习平台](https://www.paddlepaddle.org.cn/?fr=paddleEdu_aistudio)

# ## 2. ViT
# 
# ### 2.1 ViT算法综述
# 论文地址：[An Image is Worth 16x16 Words:Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
# 
# 之前的算法大都是保持CNN整体结构不变，在CNN中增加attention模块或者使用attention模块替换CNN中的某些部分。ViT算法中，作者提出没有必要总是依赖于CNN，仅仅使用Transformer结构也能够在图像分类任务中表现很好。
# 
# 受到NLP领域中Transformer成功应用的启发，ViT算法中尝试将标准的Transformer结构直接应用于图像，并对整个图像分类流程进行最少的修改。具体来讲，ViT算法中，会将整幅图像拆分成小图像块，然后把这些小图像块的线性嵌入序列作为Transformer的输入送入网络，然后使用监督学习的方式进行图像分类的训练。ViT算法的整体结构如 **图1** 所示。
# 
# <center><img src="https://ai-studio-static-online.cdn.bcebos.com/5d33d430cbfe43cb98c6c9926618cb2da8e52318a00341da92b83bc32bedeabb" width = "800"></center>
# <center><br>图1：ViT算法结构示意图</br></center>
# <br></br>
# 
# 该算法在中等规模（例如ImageNet）以及大规模（例如ImageNet-21K、JFT-300M）数据集上进行了实验验证，发现：
# * Tranformer相较于CNN结构，缺少一定的平移不变性和局部感知性，因此在数据量不充分时，很难达到同等的效果。具体表现为使用中等规模的ImageNet训练的Tranformer会比ResNet在精度上低几个百分点。
# * 当有大量的训练样本时，结果则会发生改变。使用大规模数据集进行预训练后，再使用迁移学习的方式应用到其他数据集上，可以达到或超越当前的SOTA水平。
# 
# **图2** 为大家展示了使用大规模数据集预训练后的 ViT 算法，迁移到其他小规模数据集进行训练，与使用 CNN 结构的SOTA算法精度对比。
# 
# <center><img src="https://ai-studio-static-online.cdn.bcebos.com/85d8e641c74648b3b0f34033d137b4ca0f8d9dc7e4a24eeba40e0da48dec96c7" width = "800"></center>
# <center><br>图2：ViT模型精度对比</br></center>
# <br></br>
# 
# 图中前3列为不同尺度的ViT模型，使用不同的大规模数据集进行预训练，并迁移到各个子任务上的结果。第4列为BiT算法基于JFT-300M数据集预训练后，迁移到各个子任务的结果。第5列为2020年提出的半监督算法 Noisy Student 在 ImageNet 和 ImageNet ReaL 数据集上的结果。
# 
# ---
# 
# **说明：**
# 
# BiT 与 Noisy Student 均为2020年提出的 SOTA 算法。
# 
# BiT算法：使用大规模数据集 JFT-300M 对 ResNet 结构进行预训练，其中，作者发现模型越大，预训练效果越好，最终指标最高的为4倍宽、152层深的 $ResNet152 \times 4$。论文地址：[Big Transfer (BiT): General Visual Representation Learning](https://arxiv.org/abs/1912.11370)
# 
# Noisy Student 算法：使用知识蒸馏的技术，基于 EfficientNet 结构，利用未标签数据，提高训练精度。论文地址：[Self-training with Noisy Student improves ImageNet classification](https://arxiv.org/abs/1911.04252)
# 
# ---
# 
# 
# 接下来，分别看一下ViT算法的各个组成部分。

# ### 2.2 图像分块嵌入
# 
# 考虑到之前课程中学习的，Transformer结构中，输入需要是一个二维的矩阵，矩阵的形状可以表示为 $(N,D)$，其中 $N$ 是sequence的长度，而 $D$ 是sequence中每个向量的维度。因此，在ViT算法中，首先需要设法将 $H \times W \times C$ 的三维图像转化为 $(N,D)$ 的二维输入。
# 
# ViT中的具体实现方式为：将 $H \times W \times C$ 的图像，变为一个 $N \times (P^2 * C)$ 的序列。这个序列可以看作是一系列展平的图像块，也就是将图像切分成小块后，再将其展平。该序列中一共包含了 $N=HW/P^2$ 个图像块，每个图像块的维度则是 $(P^2*C)$。其中  $P$ 是图像块的大小，$C$ 是通道数量。经过如上变换，就可以将 $N$ 视为sequence的长度了。
# 
# 但是，此时每个图像块的维度是 $(P^2*C)$，而我们实际需要的向量维度是 $D$，因此我们还需要对图像块进行 Embedding。这里 Embedding 的方式非常简单，只需要对每个 $(P^2*C)$ 的图像块做一个线性变换，将维度压缩为 $D$ 即可。
# 
# 上述对图像进行分块以及 Embedding 的具体方式如 **图3** 所示。
# 
# <center><img src="https://ai-studio-static-online.cdn.bcebos.com/a1dbbb5ad2384df88f24fc739836c31f42bf2056f78c491cbc5d31b78b933ee3" width = "800"></center>
# <center><br>图3：图像分块嵌入示意图</br></center>
# <br></br>
# 
# 具体代码实现如下所示。其中，使用了大小为 $P$ 的卷积来代替对每个大小为 $P$ 图像块展平后使用全连接进行运算的过程。

# In[8]:


# coding=utf-8
# 导入环境
import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import paddle
from paddle.io import Dataset
from paddle.nn import Conv2D, MaxPool2D, Linear, Dropout, BatchNorm, AdaptiveAvgPool2D, AvgPool2D
import paddle.nn.functional as F
import paddle.nn as nn

import ECA.module as ECA

# 图像分块、Embedding
class PatchEmbed(nn.Layer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        # 原始大小为int，转为tuple，即：img_size原始输入224，变换后为[224,224]
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        # 图像块的个数
        num_patches = (img_size[1] // patch_size[1]) *             (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        # kernel_size=块大小，即每个块输出一个值，类似每个块展平后使用相同的全连接层进行处理
        # 输入维度为3，输出维度为块向量长度
        # 与原文中：分块、展平、全连接降维保持一致
        # 输出为[B, C, H, W]
        self.proj = nn.Conv2D(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1],             "Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # [B, C, H, W] -> [B, C, H*W] ->[B, H*W, C]
        x = self.proj(x).flatten(2).transpose((0, 2, 1))
        return x


# ### 2.3 Multi-head Attention
# 
# 将图像转化为 $N \times (P^2 * C)$ 的序列后，就可以将其输入到 Tranformer 结构中进行特征提取了。在前面的课程中，我们了解到 Tranformer 结构中最重要的结构就是 Multi-head Attention，即多头注意力结构，如 **图4** 所示。
# 
# <center><img src="https://ai-studio-static-online.cdn.bcebos.com/8566c2480d554506be0c83eb0a0a60736d26aa23b23246bf8db88d59b21a55c9" width = "800"></center>
# <center><br>图4：Multi-head Attention 示意图</br></center>
# <br></br>
# 
# 
# 具有2个head的 Multi-head Attention 结构如 **图5** 所示。输入 $a^i$ 经过转移矩阵，并切分生成 $q^{(i,1)}$、$q^{(i,2)}$、$k^{(i,1)}$、$k^{(i,2)}$、$v^{(i,1)}$、$v^{(i,2)}$，然后 $q^{(i,1)}$ 与 $k^{(i,1)}$ 做 attention，得到权重向量 $\alpha$，将 $\alpha$ 与 $v^{(i,1)}$ 进行加权求和，得到最终的 $b^{(i,1)}(i=1,2,…,N)$，同理可以得到 $b^{(i,2)}(i=1,2,…,N)$。接着将它们拼接起来，通过一个线性层进行处理，得到最终的结果。
# 
# <center><img src="https://ai-studio-static-online.cdn.bcebos.com/4953243f18af450eae3d16181b9a77ce83f4623e414747298b0d7c056c3a6bfe" width = "800"></center>
# <center><br>图5：Multi-head Attention结构</br></center>
# <br></br>
# 
# 其中，使用 $q^{(i,j)}$、$k^{(i,j)}$ 与 $v^{(i,j)}$ 计算 $b^{(i,j)}(i=1,2,…,N)$ 的方法是 Scaled Dot-Product Attention。 结构如 **图6** 所示。首先使用每个 $q^{(i,j)}$ 去与 $k^{(i,j)}$ 做 attention，这里说的 attention 就是匹配这两个向量有多接近，具体的方式就是计算向量的加权内积，得到 $\alpha_{(i,j)}$。这里的加权内积计算方式如下所示：
# 
# $$ \alpha_{(1,i)} =  q^1 * k^i / \sqrt{d} $$
# 
# 其中，$d$ 是 $q$ 和 $k$ 的维度，因为 $q*k$ 的数值会随着维度的增大而增大，因此除以 $\sqrt{d}$ 的值也就相当于归一化的效果。
# 
# 接下来，把计算得到的 $\alpha_{(i,j)}$ 取 softmax 操作，再将其与 $v^{(i,j)}$ 相乘。
# 
# <center><img src="https://ai-studio-static-online.cdn.bcebos.com/5b3da7158a92461aa1f5cd0bd294a9aba0935bf02d74461b9aa15d48784e8f4e" width = "400"></center>
# <center><br>图6：Scaled Dot-Product Attention</br></center>
# <br></br>
# 
# **想了解注意力机制的更多信息，请参阅[awesome-DeepLearning](https://github.com/paddlepaddle/awesome-DeepLearning) 中的 [注意力机制知识点](https://github.com/PaddlePaddle/awesome-DeepLearning/tree/master/docs/tutorials/deep_learning/model_tuning/attention)。**
# 
# 具体代码实现如下所示。

# In[2]:


# Multi-head Attention
class Attention(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        # 计算 q,k,v 的转移矩阵
        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        # 最终的线性层
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        N, C = x.shape[1:]
        # 线性变换
        qkv = self.qkv(x).reshape((-1, N, 3, self.num_heads, C //
                                   self.num_heads)).transpose((2, 0, 3, 1, 4))
        # 分割 query key value
        q, k, v = qkv[0], qkv[1], qkv[2]
        # Scaled Dot-Product Attention
        # Matmul + Scale
        attn = (q.matmul(k.transpose((0, 1, 3, 2)))) * self.scale
        # SoftMax
        attn = nn.functional.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)
        # Matmul
        x = (attn.matmul(v)).transpose((0, 2, 1, 3)).reshape((-1, N, C))
        # 线性变换
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ### 2.4 多层感知机（MLP）
# 
#  Tranformer 结构中还有一个重要的结构就是 MLP，即多层感知机，如 **图7** 所示。
#  
#  <center><img src="https://ai-studio-static-online.cdn.bcebos.com/62a1efbf38bb4c119e89cf277dc2653394a19af9cea5476182406a2ebc0572e9" width = "600"></center>
# <center><br>图7：多层感知机</br></center>
# <br></br>
#  
#  多层感知机由输入层、输出层和至少一层的隐藏层构成。网络中各个隐藏层中神经元可接收相邻前序隐藏层中所有神经元传递而来的信息，经过加工处理后将信息输出给相邻后续隐藏层中所有神经元。在多层感知机中，相邻层所包含的神经元之间通常使用“全连接”方式进行连接。多层感知机可以模拟复杂非线性函数功能，所模拟函数的复杂性取决于网络隐藏层数目和各层中神经元数目。多层感知机的结构如 **图8** 所示。
# 
# <center><img src="https://ai-studio-static-online.cdn.bcebos.com/9ada33e2b5134412b2b3dd04dfc0e6e88e932555045147ce99a880f06d69db23" width = "400"></center>
# <center><br>图8：多层感知机结构</br></center>
# <br></br>
# 
# **想了解多层感知机的更多信息，请参阅[awesome-DeepLearning](https://github.com/paddlepaddle/awesome-DeepLearning) 中的 [多层感知机知识点](https://github.com/PaddlePaddle/awesome-DeepLearning/blob/master/docs/tutorials/deep_learning/basic_concepts/multilayer_perceptron.md)。**
# 
# 具体代码实现如下所示。

# In[3]:


class Mlp(nn.Layer):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # 输入层：线性变换
        x = self.fc1(x)
        # 应用激活函数
        x = self.act(x)
        # Dropout
        x = self.drop(x)
        # 输出层：线性变换
        x = self.fc2(x)
        # Dropout
        x = self.drop(x)
        return x


# ### 2.5 基础模块
# 
# 基于上面实现的 Attention、MLP 和 DropPath 模块就可以组合出 Vision Transformer 模型的一个基础模块，如 **图9** 所示。
# 
# <center><img src="https://ai-studio-static-online.cdn.bcebos.com/5f8f09402d5d442c8f357aa39912865c2253cc7eec374d52821d7f35e566ca67" width = "600"></center>
# <center><br>图9：Transformer 基础模块</br></center>
# <br></br>
# 

# 这里使用了DropPath（Stochastic Depth）来代替传统的Dropout结构，DropPath可以理解为一种特殊的 Dropout。其作用是在训练过程中随机丢弃子图层（randomly drop a subset of layers），而在预测时正常使用完整的 Graph。
# 
# 具体实现如下：

# In[4]:


def drop_path(x, drop_prob=0., training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = paddle.to_tensor(1 - drop_prob)
    shape = (paddle.shape(x)[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
    random_tensor = paddle.floor(random_tensor)
    output = x.divide(keep_prob) * random_tensor
    return output

class DropPath(nn.Layer):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# 基础模块的具体实现如下：

# In[10]:

cnt = 1


class Block(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer='nn.LayerNorm',
                 epsilon=1e-5):
        super().__init__()
        self.norm1 = eval(norm_layer)(dim, epsilon=epsilon)
        # Multi-head Self-attention
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)
        # DropPath
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.norm2 = eval(norm_layer)(dim, epsilon=epsilon)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)

        self.ECA1 = ECA.ECA_block()
        self.ECA2 = ECA.ECA_block()

    def forward(self, x):
        global cnt
        # print(f'=======第{cnt}次经过Transformer Encoder开始=======')

        # 切开看输入参数
        # print('Transformer输入参数大小为')
        # print(x.shape)

        # Multi-head Self-attention， Add， LayerNorm
        # x = x + self.drop_path(self.attn(self.norm1(x)))
        x = ECA.down_dimension(self.ECA1(ECA.up_dimension(x))) + self.drop_path(self.attn(self.norm1(x)))


        # 切开看参数2
        # print('Transformer中间层参数大小为')
        # print(x.shape)

        # Feed Forward， Add， LayerNorm
        # x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = ECA.down_dimension(self.ECA2(ECA.up_dimension(x))) + self.drop_path(self.mlp(self.norm2(x)))

        # 切开看输出参数
        # print('Transformer输出参数大小为')
        # print(x.shape)

        # print(f'=======第{cnt}次经过Transformer Encoder结束=======')
        cnt += 1
        return x


# ### 2.6 定义ViT网络
# 
# 基础模块构建好后，就可以构建完整的ViT网络了。ViT的完整结构如 **图10** 所示。
# 
# <center><img src="https://ai-studio-static-online.cdn.bcebos.com/60f51da9f9dc477182c9c107d27867b743ff2dcee5fe427fbf81a9d5c0a01806" width = "600"></center>
# <center><br>图10：ViT网络结构</br></center>
# <br></br>
# 
# 在实现完整网络结构之前，还需要给大家介绍几个模块：
# 
# 1. Class Token
# 
# 可以看到，假设我们将原始图像切分成 $3 \times 3$ 共9个小图像块，最终的输入序列长度却是10，也就是说我们这里人为的增加了一个向量进行输入，我们通常将人为增加的这个向量称为 Class Token。那么这个 Class Token 有什么作用呢？
# 
# 我们可以想象，如果没有这个向量，也就是将 $N=9$ 个向量输入 Transformer 结构中进行编码，我们最终会得到9个编码向量，可对于图像分类任务而言，我们应该选择哪个输出向量进行后续分类呢？
# 
# 由于选择9个中的哪个都不合适，所以ViT算法中，提出了一个可学习的嵌入向量 Class Token，将它与9个向量一起输入到 Transformer 结构中，输出10个编码向量，然后用这个 Class Token 进行分类预测即可。
# 
# 其实这里也可以理解为：ViT 其实只用到了 Transformer 中的 Encoder，而并没有用到 Decoder，而 Class Token 的作用就是寻找其他9个输入向量对应的类别。
# 
# 2. Positional Encoding
# 
# 按照 Transformer 结构中的位置编码习惯，这个工作也使用了位置编码。不同的是，ViT 中的位置编码没有采用原版 Transformer 中的 $sincos$ 编码，而是直接设置为可学习的 Positional Encoding。对训练好的 Positional Encoding 进行可视化，如 **图11** 所示。我们可以看到，位置越接近，往往具有更相似的位置编码。此外，出现了行列结构，同一行/列中的 patch 具有相似的位置编码。
# 
# <center><img src="https://ai-studio-static-online.cdn.bcebos.com/3c2889396cab4790bffb2c23b0954fe552411b45dce14a12a925e2f3ee164790" width = "600"></center>
# <center><br>图11：Positional Encoding </br></center>
# <br></br>
# 
# 3. MLP Head
# 
# 得到输出后，ViT中使用了 MLP Head对输出进行分类处理，这里的 MLP Head 由 LayerNorm 和两层全连接层组成，并且采用了 GELU 激活函数。
# 
# 具体代码如下所示。

# 首先构建基础模块部分，包括：参数初始化配置、独立的不进行任何操作的网络层。

# In[13]:


# 参数初始化配置
trunc_normal_ = nn.initializer.TruncatedNormal(std=.02)
zeros_ = nn.initializer.Constant(value=0.)
ones_ = nn.initializer.Constant(value=1.)

# 将输入 x 由 int 类型转为 tuple 类型
def to_2tuple(x):
    return tuple([x] * 2)

# 定义一个什么操作都不进行的网络层
class Identity(nn.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


# 完整代码如下所示。

# In[17]:


class VisionTransformer(nn.Layer):
    def __init__(self,
                 img_size=128,
                 patch_size=16,
                 in_chans=3,
                 class_dim=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer='nn.LayerNorm',
                 epsilon=1e-5,
                 **args):
        super().__init__()
        self.class_dim = class_dim

        self.num_features = self.embed_dim = embed_dim
        # 图片分块和降维，块大小为patch_size，最终块向量维度为768
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim)
        # 分块数量
        num_patches = self.patch_embed.num_patches
        # 可学习的位置编码
        self.pos_embed = self.create_parameter(
            shape=(1, num_patches + 1, embed_dim), default_initializer=zeros_)
        self.add_parameter("pos_embed", self.pos_embed)
        # 人为追加class token，并使用该向量进行分类预测
        self.cls_token = self.create_parameter(
            shape=(1, 1, embed_dim), default_initializer=zeros_)
        self.add_parameter("cls_token", self.cls_token)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = np.linspace(0, drop_path_rate, depth)
        # transformer
        self.blocks = nn.LayerList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                epsilon=epsilon) for i in range(depth)
        ])

        self.norm = eval(norm_layer)(embed_dim, epsilon=epsilon)

        # Classifier head
        self.head = nn.Linear(embed_dim,
                              class_dim) if class_dim > 0 else Identity()

        trunc_normal_(self.pos_embed)
        trunc_normal_(self.cls_token)
        self.apply(self._init_weights)
    # 参数初始化
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)
    # 获取图像特征
    def forward_features(self, x):
        B = paddle.shape(x)[0]
        # 将图片分块，并调整每个块向量的维度
        x = self.patch_embed(x)
        # 将class token与前面的分块进行拼接
        cls_tokens = self.cls_token.expand((B, -1, -1))
        x = paddle.concat((cls_tokens, x), axis=1)
        # 将编码向量中加入位置编码
        x = x + self.pos_embed
        x = self.pos_drop(x)
        # 堆叠 transformer 结构
        for blk in self.blocks:
            x = blk(x)
        # LayerNorm
        x = self.norm(x)
        # 提取分类 tokens 的输出
        return x[:, 0]

    def forward(self, x):
        # print('模型输入:')
        # print(x.shape)
        # 获取图像特征
        x = self.forward_features(x)
        # print('模型中间层:')
        # print(x.shape)
        # 图像分类
        x = self.head(x)
        # print('模型输出:')
        # print(x.shape)
        return x



# ## 3 基于 ImageNet 的模型评估
# 
# 上文中，我们给大家详细介绍了 ViT 的算法原理，以及如何使用飞桨框架实现 ViT 的模型结构。接下来，我们就使用 ImageNet 数据集中的验证集部分，验证一下 ViT 模型的实际效果。
# 
# ---
# 
# **说明：**
# 
# 这里的模型参数使用已经预先训练好的参数，参数来源于 PaddleClas 套件：[ViT_base_
# patch16_384](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_base_patch16_384_pretrained.pdparams)。
# 
# ---

# In[21]:


# 以下为修改添加部分
# print("begin")
#
# net = VisionTransformer()
#
# x = paddle.randn((1, 3, 384, 384))
# y = net(x)
# print(y.shape)
#
# print("end")


# ### 3.1 数据准备
# 
# [ImageNet数据集](https://image-net.org/)是为了促进计算机图像识别技术的发展而设立的一个大型图像数据集。其中常用ILSVRC2012，是ImageNet数据集的一个子集，共1000类。
# 
# 首先解压数据集以及模型权重文件，执行如下代码即可，解压执行一次就可以。

# In[8]:


# 解压数据集
# get_ipython().system('tar -xf /home/aistudio/data_np/data105740/ILSVRC2012_val.tar -C /home/aistudio/work/')
# 解压权重文件
# get_ipython().system('unzip -q -o /home/aistudio/data_np/data105741/pretrained.zip -d /home/aistudio/work/')


# ### 3.2 图像预处理
# 
# 图像分类网络对输入图片的格式、大小有一定的要求，数据灌入模型前，需要对数据进行预处理操作，使图片满足网络训练以及预测的需要。
# 
# 本实验的数据预处理共包括如下方法：
# * **图像解码**：将图像转为Numpy格式；
# * **调整图片大小**：将原图片中短边尺寸统一缩放到384；
# * **图像裁剪**：将图像的长宽统一裁剪为384×384，确保模型读入的图片数据大小统一；
# * **归一化**（normalization）：通过规范化手段，把神经网络每层中任意神经元的输入值分布改变成均值为0，方差为1的标准正太分布，使得最优解的寻优过程明显会变得平缓，训练过程更容易收敛；
# * **通道变换**：图像的数据格式为[H, W, C]（即高度、宽度和通道数），而神经网络使用的训练数据的格式为[C, H, W]，因此需要对图像数据重新排列，例如[384, 384, 3]变为[3, 384, 384]。
# 
# 下面分别介绍数据预处理方法的代码实现。

# **图像解码**

# In[9]:


# 定义decode_image函数，将图片转为Numpy格式
def decode_image(img, to_rgb=True):
    data = np.frombuffer(img, dtype='uint8')
    img = cv2.imdecode(data, 1)
    if to_rgb:
        assert img.shape[2] == 3, 'invalid shape of image[%s]' % (
            img.shape)
        img = img[:, :, ::-1]

    return img


# **调整图片大小**

# In[4]:


# 定义resize_image函数，对图片大小进行调整
def resize_image(img, size=None, resize_short=None, interpolation=-1):
    interpolation = interpolation if interpolation >= 0 else None
    if resize_short is not None and resize_short > 0:
        resize_short = resize_short
        w = None
        h = None
    elif size is not None:
        resize_short = None
        w = size if type(size) is int else size[0]
        h = size if type(size) is int else size[1]
    else:
        raise ValueError("invalid params for ReisizeImage for '            'both 'size' and 'resize_short' are None")

    img_h, img_w = img.shape[:2]
    if resize_short is not None:
        percent = float(resize_short) / min(img_w, img_h)
        w = int(round(img_w * percent))
        h = int(round(img_h * percent))
    else:
        w = w
        h = h
    if interpolation is None:
        return cv2.resize(img, (w, h))
    else:
        return cv2.resize(img, (w, h), interpolation=interpolation)


# **裁剪图片**

# In[5]:


# 定义crop_image函数，对图片进行裁剪
def crop_image(img, size):
    if type(size) is int:
        size = (size, size)
    else:
        size = size  # (h, w)

    w, h = size
    img_h, img_w = img.shape[:2]
    w_start = (img_w - w) // 2
    h_start = (img_h - h) // 2

    w_end = w_start + w
    h_end = h_start + h
    return img[h_start:h_end, w_start:w_end, :]


# **归一化**
# 
# 对每个特征进行归一化处理，需要除以255，并且减去均值和方差，使得每个特征的取值缩放到0~1之间。

# In[6]:


# 定义normalize_image函数，对图片进行归一化
def normalize_image(img, scale=None, mean=None, std=None, order= ''):
    if isinstance(scale, str):
        scale = eval(scale)
    scale = np.float32(scale if scale is not None else 1.0 / 255.0)
    mean = mean if mean is not None else [0.485, 0.456, 0.406]
    std = std if std is not None else [0.229, 0.224, 0.225]

    shape = (3, 1, 1) if order == 'chw' else (1, 1, 3)
    mean = np.array(mean).reshape(shape).astype('float32')
    std = np.array(std).reshape(shape).astype('float32')

    if isinstance(img, Image.Image):
        img = np.array(img)
    assert isinstance(img, np.ndarray), "invalid input 'img' in NormalizeImage"
    # 对图片进行归一化
    return (img.astype('float32') * scale - mean) / std


# **通道变换**

# In[7]:


# 定义to_CHW_image函数，对图片进行通道变换，将原通道为‘hwc’的图像转为‘chw‘
def to_CHW_image(img):
    if isinstance(img, Image.Image):
        img = np.array(img)
    # 对图片进行通道变换
    return img.transpose((2, 0, 1))


# **图像预处理方法汇总**

# In[14]:


# 图像预处理方法汇总
def transform(data, mode='train'):

    # 图像解码
    data = decode_image(data)
    # 图像缩放
    data = resize_image(data, resize_short=384)
    # 图像裁剪
    data = crop_image(data, size=384)
    # 标准化
    data = normalize_image(data, scale=1./255., mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    # 通道变换
    data = to_CHW_image(data)
    return data


# ### 3.3 批量数据读取
# 
# 上面的代码仅展示了读取一张图片和预处理的方法，但在真实场景的模型训练与评估过程中，通常会使用批量数据读取和预处理的方式。

# 定义数据读取器CommonDataset，实现数据批量读取和预处理。

# In[14]:


# 读取数据，如果是训练数据，随即打乱数据顺序
def get_file_list(file_list):
    with open(file_list) as flist:
        full_lines = [line.strip() for line in flist]

    return full_lines


# In[15]:


# 定义数据读取器
class CommonDataset(Dataset):
    def __init__(self, data_dir, file_list):
        self.full_lines = get_file_list(file_list)
        self.delimiter = ' '
        self.num_samples = len(self.full_lines)
        self.data_dir = data_dir
        return

    def __getitem__(self, idx):
        line = self.full_lines[idx]
        img_path, label = line.split(self.delimiter)
        img_path = os.path.join(self.data_dir, img_path)
        with open(img_path, 'rb') as f:
            img = f.read()
        
        transformed_img = transform(img)
        return (transformed_img, int(label))

    def __len__(self):
        return self.num_samples


# 数据预处理耗时较长，推荐使用 [paddle.io.DataLoader](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/io/DataLoader_cn.html#dataloader) API中的``num_workers``参数，设置进程数量，实现多进程读取数据。
# 
# > *class* paddle.io.DataLoader(*dataset,  batchsize=2,  numworkers=2*) 
# 
# 关键参数含义如下：
# 
# * batch_size (int|None) - 每个mini-batch中样本个数；
# * num_workers (int) - 加载数据的子进程个数 。
# 
# 多线程读取实现代码如下。

# In[19]:


# ====================================================own=================================================================

# =====================================================own================================================================


# DATADIR = '/home/aistudio/work/ILSVRC2012_val/'
# VAL_FILE_LIST = '/home/aistudio/work/ILSVRC2012_val/val_list.txt'
#
# # 创建数据读取类
# val_dataset = CommonDataset(DATADIR, VAL_FILE_LIST)
#
# # 使用paddle.io.DataLoader创建数据读取器，并设置batchsize，进程数量num_workers等参数
# val_loader = paddle.io.DataLoader(val_dataset, batch_size=2, num_workers=1, drop_last=True)


# In[20]:


# img, label = next(val_loader())


# In[21]:


# img.shape, label.shape


# 至此，我们完成了数据预处理、批量读取和加速等过程，通过paddle.io.Dataset可以返回图片和标签信息，接下来将处理好的数据输入到神经网络，应用到具体算法上。















# ### 3.4 模型评估
# 
# 使用保存的模型参数评估在验证集上的准确率，代码实现如下：

# In[24]:


# # 开启0号GPU
# use_gpu = True
# paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')
#
# print('start evaluation .......')
# # 实例化模型
# model = VisionTransformer(
#         patch_size=16,
#         class_dim=1000,
#         embed_dim=768,
#         depth=12,
#         num_heads=12,
#         mlp_ratio=4,
#         qkv_bias=True,
#         epsilon=1e-6)
# # 加载模型参数
# params_file_path="/home/aistudio/work/ViT_base_patch16_384_pretrained.pdparams"
# model_state_dict = paddle.load(params_file_path)
# model.load_dict(model_state_dict)
#
# model.eval()
#
# acc_set = []
# avg_loss_set = []
# for batch_id, data_np in enumerate(val_loader()):
#     x_data, y_data = data_np
#     y_data = y_data.reshape([-1, 1])
#     img = paddle.to_tensor(x_data)
#     label = paddle.to_tensor(y_data)
#     # 运行模型前向计算，得到预测值
#     logits = model(img)
#     # 多分类，使用softmax计算预测概率
#     pred = F.softmax(logits)
#     # 计算交叉熵损失函数
#     loss_func = paddle.nn.CrossEntropyLoss(reduction='none')
#     loss = loss_func(logits, label)
#     # 计算准确率
#     acc = paddle.metric.accuracy(pred, label)
#
#     acc_set.append(acc.numpy())
#     avg_loss_set.append(loss.numpy())
# print("[validation] accuracy/loss: {}/{}".format(np.mean(acc_set), np.mean(avg_loss_set)))


# 这里给出论文中不同尺度的ViT模型在不同数据集上的精度对比，如 **图12** 所示。
# 
# <center><img src="https://ai-studio-static-online.cdn.bcebos.com/a79fc04008fb4b2a99d7e7e76c87db55243ede24ce9145759ce8fd1bc9567bb8" width = "600"></center>
# <center><br>图12：不同尺度的ViT模型精度</br></center>
# <br></br>
# 
# 本实验中实现的是ViT-B/16版本，可以看到，最终在ImageNet上进行验证，准确率为84.14，与论文基本一致。

# ### 3.5 模型预测
# 
# 对数据集中任意一张图片进行预测，并可视化观察预测结果。代码实现如下：

# In[16]:


# # 开启0号GPU
# use_gpu = True
# paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')
#
# # 实例化模型
# model = VisionTransformer(
#         patch_size=16,
#         class_dim=1000,
#         embed_dim=768,
#         depth=12,
#         num_heads=12,
#         mlp_ratio=4,
#         qkv_bias=True,
#         epsilon=1e-6)
# # 加载模型参数
# params_file_path="/home/aistudio/work/ViT_base_patch16_384_pretrained.pdparams"
# model_state_dict = paddle.load(params_file_path)
# model.load_dict(model_state_dict)
#
# model.eval()
#
# with open('/home/aistudio/work/ILSVRC2012_val/val_list.txt') as flist:
#     line = flist.readline()
# img_path, label = line.split(' ')
# img_path = os.path.join('/home/aistudio/work/ILSVRC2012_val/', img_path)
# with open(img_path, 'rb') as f:
#     img = f.read()
# transformed_img = transform(img)
# true_label = int(label)
# x_data = paddle.to_tensor(transformed_img[np.newaxis,:, : ,:])
# logits = model(x_data)
# pred = F.softmax(logits)
# pred_label = int(np.argmax(pred.numpy()))
# print("Ground truth lable index: {}, Pred label index:{}".format(true_label, pred_label))
# img = Image.open(img_path)
# plt.imshow(img)
# plt.axis('off')
# plt.show()