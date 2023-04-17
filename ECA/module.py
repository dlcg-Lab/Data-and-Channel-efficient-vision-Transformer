import paddle
import paddle.nn as nn


class ECA_block(paddle.nn.Layer):
    """
    构建ECA模块
    参数：
        channel ：输入特征图的通道数
        k_size ：自适应卷积核的大小
    """
    def __init__(self, channel=1, k_size=3):
        super(ECA_block, self).__init__()
        self.avg_pool = paddle.nn.AdaptiveAvgPool2D(1)
        self.conv = paddle.nn.Conv1D(1, 1, kernel_size=k_size, padding=(k_size-1)//2, bias_attr=False)
        self.sigmoid = paddle.nn.Sigmoid()

    def forward(self, x):

        # 切开看输入参数
        # print('ECA_block输入参数大小为')
        # print(x.shape)

        # 全局空间信息的特征提取
        y = self.avg_pool(x)

        # ECA模块的两个不同分支
        # shape = [64,64,1,1]
        y = y.squeeze(-1).transpose([0, 2, 1])
        # print("infront: ",y.shape)
        y = self.conv(y)
        # print("conv out : ",y.shape)
        y = y.transpose([0, 2, 1]).unsqueeze(-1)
        # print("conv finish: ",y.shape)

        # 多尺度特征融合
        y = self.sigmoid(y)

        # y变成x一样的形状
        y = y.expand_as(x)

        #y相当于权重，也就是注意力
        result = x*y

        # 切开看输出参数
        # print('ECA_block输出参数大小为')
        # print(result.shape)

        return result


def down_dimension(x):
    x = x[0]
    return x

def up_dimension(x):
    x = x.unsqueeze(0)
    return x
