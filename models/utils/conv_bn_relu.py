import paddle
import paddle.nn as nn
import math


class Conv_BN_ReLU(nn.Layer):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0):
        super(Conv_BN_ReLU, self).__init__()
        self.conv = nn.Conv2D(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              bias_attr=False)
        self.bn = nn.BatchNorm2D(out_planes)
        self.relu = nn.ReLU()

        # for m in self.sublayers():
        #     if isinstance(m, nn.layer.conv.Conv2D):
        #         # print('conv2d', dir(m))
        #         # print('ks', m._kernel_size)
        #         # print('oc', m._out_channels)
        #         # print('weight', callable(m.weight.norm))
        #
        #         n = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
        #         m.weight.norm(0, math.sqrt(2. / n))
        #     # elif isinstance(m, nn.layer.norm.BatchNorm2D):
        #     # print('bn2d', dir(m))
        #     # m.weight.data.fill_(1)
        #     # m.bias.data.zero_()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


mylayer = Conv_BN_ReLU(3, 4)
# for i, m in enumerate(mylayer.sublayers()):
#     print(i, type(m))
