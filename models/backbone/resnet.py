import os
import sys
import paddle
import paddle.nn as nn
import math

__all__ = ['resnet50']

model_sd = ['conv1.weight', 'bn1.weight', 'bn1.bias', 'bn1._mean', 'bn1._variance', 'layer1.0.conv1.weight',
            'layer1.0.bn1.weight', 'layer1.0.bn1.bias',
            'layer1.0.bn1._mean', 'layer1.0.bn1._variance', 'layer1.0.conv2.weight', 'layer1.0.bn2.weight',
            'layer1.0.bn2.bias', 'layer1.0.bn2._mean', 'layer1.0.bn2._variance', 'layer1.0.conv3.weight',
            'layer1.0.bn3.weight', 'layer1.0.bn3.bias', 'layer1.0.bn3._mean', 'layer1.0.bn3._variance',
            'layer1.0.downsample.0.weight', 'layer1.0.downsample.1.weight', 'layer1.0.downsample.1.bias',
            'layer1.0.downsample.1._mean', 'layer1.0.downsample.1._variance', 'layer1.1.conv1.weight',
            'layer1.1.bn1.weight', 'layer1.1.bn1.bias', 'layer1.1.bn1._mean', 'layer1.1.bn1._variance',
            'layer1.1.conv2.weight', 'layer1.1.bn2.weight', 'layer1.1.bn2.bias', 'layer1.1.bn2._mean',
            'layer1.1.bn2._variance', 'layer1.1.conv3.weight', 'layer1.1.bn3.weight', 'layer1.1.bn3.bias',
            'layer1.1.bn3._mean', 'layer1.1.bn3._variance', 'layer1.2.conv1.weight', 'layer1.2.bn1.weight',
            'layer1.2.bn1.bias', 'layer1.2.bn1._mean', 'layer1.2.bn1._variance', 'layer1.2.conv2.weight',
            'layer1.2.bn2.weight', 'layer1.2.bn2.bias', 'layer1.2.bn2._mean', 'layer1.2.bn2._variance',
            'layer1.2.conv3.weight', 'layer1.2.bn3.weight', 'layer1.2.bn3.bias', 'layer1.2.bn3._mean',
            'layer1.2.bn3._variance', 'layer2.0.conv1.weight', 'layer2.0.bn1.weight', 'layer2.0.bn1.bias',
            'layer2.0.bn1._mean', 'layer2.0.bn1._variance', 'layer2.0.conv2.weight', 'layer2.0.bn2.weight',
            'layer2.0.bn2.bias', 'layer2.0.bn2._mean', 'layer2.0.bn2._variance', 'layer2.0.conv3.weight',
            'layer2.0.bn3.weight', 'layer2.0.bn3.bias', 'layer2.0.bn3._mean', 'layer2.0.bn3._variance',
            'layer2.0.downsample.0.weight', 'layer2.0.downsample.1.weight', 'layer2.0.downsample.1.bias',
            'layer2.0.downsample.1._mean', 'layer2.0.downsample.1._variance', 'layer2.1.conv1.weight',
            'layer2.1.bn1.weight', 'layer2.1.bn1.bias', 'layer2.1.bn1._mean', 'layer2.1.bn1._variance',
            'layer2.1.conv2.weight', 'layer2.1.bn2.weight', 'layer2.1.bn2.bias', 'layer2.1.bn2._mean',
            'layer2.1.bn2._variance', 'layer2.1.conv3.weight', 'layer2.1.bn3.weight', 'layer2.1.bn3.bias',
            'layer2.1.bn3._mean', 'layer2.1.bn3._variance', 'layer2.2.conv1.weight', 'layer2.2.bn1.weight',
            'layer2.2.bn1.bias', 'layer2.2.bn1._mean', 'layer2.2.bn1._variance', 'layer2.2.conv2.weight',
            'layer2.2.bn2.weight', 'layer2.2.bn2.bias', 'layer2.2.bn2._mean', 'layer2.2.bn2._variance',
            'layer2.2.conv3.weight', 'layer2.2.bn3.weight', 'layer2.2.bn3.bias', 'layer2.2.bn3._mean',
            'layer2.2.bn3._variance', 'layer2.3.conv1.weight', 'layer2.3.bn1.weight', 'layer2.3.bn1.bias',
            'layer2.3.bn1._mean', 'layer2.3.bn1._variance', 'layer2.3.conv2.weight', 'layer2.3.bn2.weight',
            'layer2.3.bn2.bias', 'layer2.3.bn2._mean', 'layer2.3.bn2._variance', 'layer2.3.conv3.weight',
            'layer2.3.bn3.weight', 'layer2.3.bn3.bias', 'layer2.3.bn3._mean', 'layer2.3.bn3._variance',
            'layer3.0.conv1.weight', 'layer3.0.bn1.weight', 'layer3.0.bn1.bias', 'layer3.0.bn1._mean',
            'layer3.0.bn1._variance', 'layer3.0.conv2.weight', 'layer3.0.bn2.weight', 'layer3.0.bn2.bias',
            'layer3.0.bn2._mean', 'layer3.0.bn2._variance', 'layer3.0.conv3.weight', 'layer3.0.bn3.weight',
            'layer3.0.bn3.bias', 'layer3.0.bn3._mean', 'layer3.0.bn3._variance',
            'layer3.0.downsample.0.weight', 'layer3.0.downsample.1.weight', 'layer3.0.downsample.1.bias',
            'layer3.0.downsample.1._mean', 'layer3.0.downsample.1._variance', 'layer3.1.conv1.weight',
            'layer3.1.bn1.weight', 'layer3.1.bn1.bias', 'layer3.1.bn1._mean', 'layer3.1.bn1._variance',
            'layer3.1.conv2.weight', 'layer3.1.bn2.weight', 'layer3.1.bn2.bias', 'layer3.1.bn2._mean',
            'layer3.1.bn2._variance', 'layer3.1.conv3.weight', 'layer3.1.bn3.weight', 'layer3.1.bn3.bias',
            'layer3.1.bn3._mean', 'layer3.1.bn3._variance', 'layer3.2.conv1.weight', 'layer3.2.bn1.weight',
            'layer3.2.bn1.bias', 'layer3.2.bn1._mean', 'layer3.2.bn1._variance', 'layer3.2.conv2.weight',
            'layer3.2.bn2.weight', 'layer3.2.bn2.bias', 'layer3.2.bn2._mean', 'layer3.2.bn2._variance',
            'layer3.2.conv3.weight', 'layer3.2.bn3.weight', 'layer3.2.bn3.bias', 'layer3.2.bn3._mean',
            'layer3.2.bn3._variance', 'layer3.3.conv1.weight', 'layer3.3.bn1.weight', 'layer3.3.bn1.bias',
            'layer3.3.bn1._mean', 'layer3.3.bn1._variance', 'layer3.3.conv2.weight', 'layer3.3.bn2.weight',
            'layer3.3.bn2.bias', 'layer3.3.bn2._mean', 'layer3.3.bn2._variance', 'layer3.3.conv3.weight',
            'layer3.3.bn3.weight', 'layer3.3.bn3.bias', 'layer3.3.bn3._mean', 'layer3.3.bn3._variance',
            'layer3.4.conv1.weight', 'layer3.4.bn1.weight', 'layer3.4.bn1.bias', 'layer3.4.bn1._mean',
            'layer3.4.bn1._variance', 'layer3.4.conv2.weight', 'layer3.4.bn2.weight', 'layer3.4.bn2.bias',
            'layer3.4.bn2._mean', 'layer3.4.bn2._variance', 'layer3.4.conv3.weight', 'layer3.4.bn3.weight',
            'layer3.4.bn3.bias', 'layer3.4.bn3._mean', 'layer3.4.bn3._variance', 'layer3.5.conv1.weight',
            'layer3.5.bn1.weight', 'layer3.5.bn1.bias', 'layer3.5.bn1._mean', 'layer3.5.bn1._variance',
            'layer3.5.conv2.weight', 'layer3.5.bn2.weight', 'layer3.5.bn2.bias', 'layer3.5.bn2._mean',
            'layer3.5.bn2._variance', 'layer3.5.conv3.weight', 'layer3.5.bn3.weight', 'layer3.5.bn3.bias',
            'layer3.5.bn3._mean', 'layer3.5.bn3._variance', 'layer4.0.conv1.weight', 'layer4.0.bn1.weight',
            'layer4.0.bn1.bias', 'layer4.0.bn1._mean', 'layer4.0.bn1._variance', 'layer4.0.conv2.weight',
            'layer4.0.bn2.weight', 'layer4.0.bn2.bias', 'layer4.0.bn2._mean', 'layer4.0.bn2._variance',
            'layer4.0.conv3.weight', 'layer4.0.bn3.weight', 'layer4.0.bn3.bias', 'layer4.0.bn3._mean',
            'layer4.0.bn3._variance', 'layer4.0.downsample.0.weight', 'layer4.0.downsample.1.weight',
            'layer4.0.downsample.1.bias', 'layer4.0.downsample.1._mean', 'layer4.0.downsample.1._variance',
            'layer4.1.conv1.weight', 'layer4.1.bn1.weight', 'layer4.1.bn1.bias', 'layer4.1.bn1._mean',
            'layer4.1.bn1._variance', 'layer4.1.conv2.weight', 'layer4.1.bn2.weight', 'layer4.1.bn2.bias',
            'layer4.1.bn2._mean', 'layer4.1.bn2._variance', 'layer4.1.conv3.weight', 'layer4.1.bn3.weight',
            'layer4.1.bn3.bias', 'layer4.1.bn3._mean', 'layer4.1.bn3._variance', 'layer4.2.conv1.weight',
            'layer4.2.bn1.weight', 'layer4.2.bn1.bias', 'layer4.2.bn1._mean', 'layer4.2.bn1._variance',
            'layer4.2.conv2.weight', 'layer4.2.bn2.weight', 'layer4.2.bn2.bias', 'layer4.2.bn2._mean',
            'layer4.2.bn2._variance', 'layer4.2.conv3.weight', 'layer4.2.bn3.weight', 'layer4.2.bn3.bias',
            'layer4.2.bn3._mean', 'layer4.2.bn3._variance', 'fc.weight', 'fc.bias']


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2D(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias_attr=False)


class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2D(planes)
        self.relu = nn.ReLU
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2D(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2D(inplanes, planes, kernel_size=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(planes)
        self.conv2 = nn.Conv2D(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(planes)
        self.conv3 = nn.Conv2D(planes, planes * 4, kernel_size=1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(planes * 4)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Convkxk(nn.Layer):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0):
        super(Convkxk, self).__init__()
        self.conv = nn.Conv2D(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              bias_attr=False)
        self.bn = nn.BatchNorm2D(out_planes)
        self.relu = nn.ReLU()

        # for m in self.sublayers():
        #     if isinstance(m, nn.Conv2D):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2D):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResNet(nn.Layer):

    def __init__(self, block, layers):
        super(ResNet, self).__init__()
        self.inplanes = 128
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = nn.BatchNorm2D(64)
        self.relu1 = nn.ReLU()
        self.conv2 = conv3x3(64, 64)
        self.bn2 = nn.BatchNorm2D(64)
        self.relu2 = nn.ReLU()
        self.conv3 = conv3x3(64, 128)
        self.bn3 = nn.BatchNorm2D(128)
        self.relu3 = nn.ReLU()
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # for m in self.sublayers():
        #     if isinstance(m, nn.Conv2D):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2D):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias_attr=False),
                nn.BatchNorm2D(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        f = []
        x = self.layer1(x)
        # print('layer1:', paddle.shape(x))
        f.append(x)
        x = self.layer2(x)
        # print('layer2:', paddle.shape(x))
        f.append(x)
        x = self.layer3(x)
        # print('layer3:', paddle.shape(x))
        f.append(x)
        x = self.layer4(x)
        # print('layer4:', paddle.shape(x))
        f.append(x)

        return tuple(f)


def resnet50(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        print("Loading pretrained model from /home/data6/yjw/PSENet_paddle/pretrained/ResNet50_pretrained.pdparams")
        pretrain_state_dict = paddle.load('/home/data6/yjw/PSENet_paddle/pretrained/ResNet50_pretrained.pdparams')
        new_sd = dict()
        for idex, value in enumerate(pretrain_state_dict.values()):
            new_sd[model_sd[idex]] = value
        model.set_state_dict(new_sd)
    return model


