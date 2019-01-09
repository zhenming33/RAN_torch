import torch
from torch import nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')

class scale_and_shift(nn.Module):
    def __init__(self):
        super(scale_and_shift, self).__init__()

        self.alpha = torch.ones(1).to(device)
        self.beta = torch.zeros(1).to(device)

    def forward(self, input):
        return input * self.alpha + self.beta



class crelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(crelu, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scale_and_shift = scale_and_shift()
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv(input)
        x = self.bn(x)
        x2 = -x
        x = torch.cat((x,x2),1)
        x = self.scale_and_shift(x)
        x = self.relu(x)
        return x


class bn_scale_relu(nn.Module):
    def __init__(self, in_channels):
        super(bn_scale_relu, self).__init__()

        self.bn = nn.BatchNorm2d(in_channels)
        self.scale_and_shift = scale_and_shift()
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.bn(input)
        x = self.scale_and_shift(x)
        x = self.relu(x)
        return x

class res_crelu(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, kernel_size, stride, padding, bsr, proj):
        super(res_crelu, self).__init__()

        self.bsr = bsr
        self.proj = proj
        self.bn_scale_relu_input = bn_scale_relu(in_channels)
        self.bn_scale_relu_conv1 = bn_scale_relu(middle_channels[0])
        if self.proj:
            self.shortcut_conv = nn.Conv2d(in_channels, out_channels, (1, 1), stride)
        self.conv1 = nn.Conv2d(in_channels, middle_channels[0], (1, 1), stride, (0, 0))
        self.conv2 = nn.Conv2d(middle_channels[0], middle_channels[1], kernel_size, (1, 1), padding)
        self.bn = nn.BatchNorm2d(middle_channels[1])
        self.scale = scale_and_shift()
        self.relu = nn.ReLU()
        self.conv3 = nn.Conv2d(2*middle_channels[1], out_channels, (1, 1), (1, 1), (0, 0))

    def forward(self, input):
        if self.bsr:
            x = self.bn_scale_relu_input(input)
        else:
            x = input
        if self.proj:
            shortcut = self.shortcut_conv(input)
        else:
            shortcut = input
        conv1 = self.conv1(x)
        bsr = self.bn_scale_relu_conv1(conv1)
        conv2 = self.conv2(bsr)
        bn = self.bn(conv2)
        bn2 = -bn
        bn = torch.cat((bn, bn2), 1)
        scale = self.scale(bn)
        relu = self.relu(scale)
        conv3 = self.conv3(relu)
        act = conv3 + shortcut
        return act

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels = 1, kernel_size = (1, 1), stride = (1, 1), padding = (0, 0)):
        super(Conv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scale = scale_and_shift()
        self.act = nn.ReLU()

    def forward(self, input):
        conv = self.conv(input)
        bn = self.bn(conv)
        scale = self.scale(bn)
        act = self.act(scale)
        return act

class inception(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, kernel, stride, proj, last=False):
        super(inception, self).__init__()

        self.stride = stride
        self.last = last
        self.proj = proj
        if self.proj:
            self.shortcut_conv = nn.Conv2d(in_channels, out_channels, (1, 1), stride)
        self.bsr = bn_scale_relu(in_channels)
        self.conv_a = nn.Conv2d(in_channels, middle_channels[0], (1, 1), stride, (0, 0))

        self.conv_b1 = Conv(in_channels, middle_channels[1][0], (1, 1), stride, (0, 0))
        self.conv_b2 = Conv(middle_channels[1][0], middle_channels[1][1], kernel, (1, 1), (1, 1))


        self.conv_c1 = Conv(in_channels, middle_channels[2][0], (1, 1), stride, (0, 0))
        self.conv_c2 = Conv(middle_channels[2][0], middle_channels[2][1], kernel, (1, 1), (1, 1))
        self.conv_c3 = Conv(middle_channels[2][1], middle_channels[2][2], kernel, (1, 1), (1, 1))

        if self.stride[1] > 1:
            self.pool_d = nn.MaxPool2d(kernel, stride, (1, 1))
            self.conv_d = Conv(in_channels, middle_channels[3], (1, 1), (1, 1), (0, 0))
            self.conv = nn.Conv2d(middle_channels[0] + middle_channels[1][1] + middle_channels[2][2] + middle_channels[3],
                                  out_channels, (1, 1), (1, 1), (0, 0))
        else:
            self.conv = nn.Conv2d(middle_channels[0] + middle_channels[1][1] + middle_channels[2][2], out_channels, (1, 1), (1, 1), (0, 0))

        if self.last:
            self.bn = nn.BatchNorm2d(out_channels)
            self.scale = scale_and_shift()

    def forward(self, input):
        if self.proj:
            shortcut = self.shortcut_conv(input)
        else:
            shortcut = input
        bsr = self.bsr(input)
        conv_a = self.conv_a(bsr)

        conv_b1 = self.conv_b1(bsr)
        conv_b2 = self.conv_b2(conv_b1)

        conv_c1 = self.conv_c1(bsr)
        conv_c2 = self.conv_c2(conv_c1)
        conv_c3 = self.conv_c3(conv_c2)

        if self.stride[1] > 1:
            pool_d = self.pool_d(bsr)
            conv_d = self.conv_d(pool_d)
            conv_concat = torch.cat((conv_a, conv_b2, conv_c3, conv_d), 1)
        else:
            conv_concat = torch.cat((conv_a, conv_b2, conv_c3), 1)

        conv = self.conv(conv_concat)

        if self.last:
            bn = self.bn(conv)
            scale = self.scale(bn)
            output = scale + shortcut
        else:
            output = conv + shortcut

        return output


class PVANet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PVANet, self).__init__()
        self.conv1_1 = crelu(in_channels, 16, (7, 7), (3, 3), (2, 2))
        self.pool1_1 = nn.MaxPool2d((3, 3), (2, 2), (1, 1))
        self.conv2_1 = res_crelu(32, [24, 24], 64, (3, 3), (1, 1), (1, 1), False, True)
        self.conv2_2 = res_crelu(64, [24, 24], 64, (3, 3), (1, 1), (1, 1), True, False)
        self.conv2_3 = res_crelu(64, [24, 24], 64, (3, 3), (1, 1), (1, 1), True, False)
        self.scale3_1 = bn_scale_relu(64)
        self.conv3_1 = res_crelu(64, [48, 48], 128, (3, 3), (2, 2), (1, 1), False, True)
        self.conv3_2 = res_crelu(128, [48, 48], 128, (3, 3), (1, 1), (1, 1), True, False)
        self.conv3_3 = res_crelu(128, [48, 48], 128, (3, 3), (1, 1), (1, 1), True, False)
        self.conv3_4 = res_crelu(128, [48, 48], 128, (3, 3), (1, 1), (1, 1), True, False)
        self.downscale = nn.MaxPool2d((3, 3), (2, 2), (1, 1))
        self.conv4_1 = inception(128, [64, [48, 128], [24, 48, 48], 128], 256, (3, 3), (2, 2), True)
        self.conv4_2 = inception(256, [64, [64, 128], [24, 48, 48]], 256, (3, 3), (1, 1), False)
        self.conv4_3 = inception(256, [64, [64, 128], [24, 48, 48]], 256, (3, 3), (1, 1), False)
        self.conv4_4 = inception(256, [64, [64, 128], [24, 48, 48]], 256, (3, 3), (1, 1), False)
        self.conv5_1 = inception(256, [64, [96, 192], [32, 64, 64], 128], 384, (3, 3), (2, 2), True)
        self.conv5_2 = inception(384, [64, [96, 192], [32, 64, 64]], 384, (3, 3), (1, 1), False)
        self.conv5_3 = inception(384, [64, [96, 192], [32, 64, 64]], 384, (3, 3), (1, 1), False)
        self.conv5_4 = inception(384, [64, [96, 192], [32, 64, 64]], 384, (3, 3), (1, 1), False, True)
        self.bsr = bn_scale_relu(384)
        self.upscale = nn.ConvTranspose2d(384, 384, (4, 4), (2, 2), (1, 1))
        self.convf = nn.Conv2d(128 + 256 + 384, out_channels, (1, 1), (1, 1), (0, 0))

    def forward(self, input):
        #input shape : (32, 3, 224, 224)
        conv1_1 = self.conv1_1(input)       #(32, 32, 74, 74)
        pool1_1 = self.pool1_1(conv1_1)     #(32, 32, 37, 37)
        conv2_1 = self.conv2_1(pool1_1)     #(32, 64, 37, 37)
        conv2_2 = self.conv2_2(conv2_1)     #(32, 64, 37, 37)
        conv2_3 = self.conv2_3(conv2_2)     #(32, 64, 37, 37)
        scale3_1 = self.scale3_1(conv2_3)   #(32, 64, 37, 37)
        conv3_1 = self.conv3_1(scale3_1)    #(32, 128, 19, 19)
        conv3_2 = self.conv3_2(conv3_1)     #(32, 128, 19, 19)
        conv3_3 = self.conv3_3(conv3_2)     #(32, 128, 19, 19)
        conv3_4 = self.conv3_4(conv3_3)     #(32, 128, 19, 19)
        downscale = self.downscale(conv3_4) #(32, 128, 10, 10)
        conv4_1 = self.conv4_1(conv3_4)     #(32, 256, 10, 10)
        conv4_2 = self.conv4_2(conv4_1)     #(32, 256, 10, 10)
        conv4_3 = self.conv4_3(conv4_2)     #(32, 256, 10, 10)
        conv4_4 = self.conv4_4(conv4_3)     #(32, 256, 10, 10)
        conv5_1 = self.conv5_1(conv4_4)     #(32, 384, 5, 5)
        conv5_2 = self.conv5_2(conv5_1)     #(32, 384, 5, 5)
        conv5_3 = self.conv5_3(conv5_2)     #(32, 384, 5, 5)
        conv5_4 = self.conv5_4(conv5_3)     #(32, 384, 5, 5)

        bsr = self.bsr(conv5_4)             #(32, 384, 5, 5)
        upscale = self.upscale(bsr)
        concat = torch.cat((downscale, conv4_4, upscale), 1)
        convf =self.convf(concat)
        return convf


if __name__ == '__main__':
    x = torch.Tensor(32,3,224,224).to(device)
    x = PVANet(3, 512).to(device)(x)
    print(x.size())

