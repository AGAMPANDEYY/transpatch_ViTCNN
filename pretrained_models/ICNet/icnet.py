import sys
sys.path.append('/kaggle/working/adversarial-patch-transferability/pretrained_models/ICNet/')
from backbone import SegBaseModel,SegBaseModel_resnet
import torch.nn as nn
import torch.nn.functional as F

class ICNet(SegBaseModel_resnet):
    """Image Cascade Network"""

    def __init__(self, nclass = 8):
        super(ICNet, self).__init__(nclass)
        self.conv_sub1 = nn.Sequential(
            _ConvBNReLU(3, 32, 3, 2),
            _ConvBNReLU(32, 32, 3, 2),
            _ConvBNReLU(32, 64, 3, 2)
        )

        self.ppm = PyramidPoolingModule()

        self.head = _ICHead(nclass)

        self.__setattr__('exclusive', ['conv_sub1', 'head'])

    def forward(self, x):
        # sub 1
        # Input = 10,3,1024,2048
        x_sub1 = self.conv_sub1(x) # (10,32,512 1024) -> (10,32,256 512) -> (10,64,128 256)

        # sub 2
        x_sub2 = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True) # (10,3,512,1024)
        #x_sub2, _ = self.base_forward(x_sub2) # [10, 64, 32, 64] output of feature10 in mobileNetv2
        _, x_sub2, _, _ = self.base_forward(x_sub2)
        

        # sub 4
        x_sub4 = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=True) # (10,3,256,512)
        #_, x_sub4 = self.base_forward(x_sub4) # [10, 1280, 8, 16] output of feature18 in mobilenetv2
        _, _, _, x_sub4 = self.base_forward(x_sub4)
        x_sub4 = self.ppm(x_sub4) # [10, 1280, 8, 16] pyramid pooling just does avg/max pooling at different res and adds it to the image

        outputs = self.head(x_sub1, x_sub2, x_sub4)

        return tuple(outputs)

class PyramidPoolingModule(nn.Module):
	def __init__(self, pyramids=[1,2,3,6]):
		super(PyramidPoolingModule, self).__init__()
		self.pyramids = pyramids

	def forward(self, input):
		feat = input
		height, width = input.shape[2:]
		for bin_size in self.pyramids:
			x = F.adaptive_avg_pool2d(input, output_size=bin_size)
			x = F.interpolate(x, size=(height, width), mode='bilinear', align_corners=True)
			feat  = feat + x
		return feat

class _ICHead(nn.Module):
    def __init__(self, nclass, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_ICHead, self).__init__()
        #self.cff_12 = CascadeFeatureFusion(512, 64, 128, nclass, norm_layer, **kwargs)
        self.cff_12 = CascadeFeatureFusion(128, 64, 128, nclass, norm_layer) # ResNet
        self.cff_24 = CascadeFeatureFusion(2048, 512, 128, nclass, norm_layer) # ResNet
        #self.cff_12 = CascadeFeatureFusion(128, 64, 128, nclass, norm_layer) # MobileNet
        #self.cff_24 = CascadeFeatureFusion(64, 32, 128, nclass, norm_layer) ## MobileNet

        self.conv_cls = nn.Conv2d(128, nclass, 1, bias=False)

    def forward(self, x_sub1, x_sub2, x_sub4):
        outputs = list()

        x_cff_24, x_24_cls = self.cff_24(x_sub4, x_sub2) # [10,128,32,64], [10,19,32,64]
        outputs.append(x_24_cls)
        x_cff_12, x_12_cls = self.cff_12(x_cff_24, x_sub1) # [10,128,128,256], [10,19,128,256]
        outputs.append(x_12_cls)

        up_x2 = F.interpolate(x_cff_12, scale_factor=2, mode='bilinear', align_corners=True) # [10,128,256,512]
        up_x2 = self.conv_cls(up_x2) # [10,19,256,512]
        outputs.append(up_x2)
        up_x8 = F.interpolate(up_x2, scale_factor=4, mode='bilinear', align_corners=True) # [10,19,1024,2048]
        outputs.append(up_x8)
        # 1 -> 1/4 -> 1/8 -> 1/16
        outputs.reverse()
        ## sizes:
        # [10,19,1024,2048]
        # [10,19,256,512]
        # [10,19,128,256]
        # [10,19,32,64]

        return outputs


class _ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,norm_layer=nn.BatchNorm2d, bias=False):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size= kernel_size, stride = stride, padding = padding, bias = bias)
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CascadeFeatureFusion(nn.Module):
    """CFF Unit"""

    def __init__(self, low_channels, high_channels, out_channels, nclass, norm_layer=nn.BatchNorm2d):
        super(CascadeFeatureFusion, self).__init__()
        self.conv_low = nn.Sequential(
            nn.Conv2d(low_channels, out_channels, 3, padding=2, dilation=2, bias=False),
            norm_layer(out_channels)
        )
        self.conv_high = nn.Sequential(
            nn.Conv2d(high_channels, out_channels, 1, bias=False),
            norm_layer(out_channels)
        )
        self.conv_low_cls = nn.Conv2d(out_channels, nclass, 1, bias=False)

    def forward(self, x_low, x_high):
        ## With x_sub4 and x_sub2
        ## Inputs:
        # x_low (x_sub4): [10, 1280, 8, 16]
        # x_high (x_sub2): [10, 64, 32, 64]
        # low,high,out: 1280, 64, 128

        ## With x_cff_24, x_sub1
        ## Inputs
        # x_low (x_cff_24): [10,128,32,64]
        # x_high (x_sub1): [10,64,128,256]
        # low,high,out: 128, 64, 128

        #(32,64,k = [3,3],p = [2,2],d = [2,2])
        #(128,256,k = [3,3],p = [2,2],d = [2,2])

        x_low = F.interpolate(x_low, size=x_high.size()[2:], mode='bilinear', align_corners=True) # [10,1280, 32, 64] / [10,128,128,256]
        x_low = self.conv_low(x_low) # [10,128,32,64] / [10,128,128,256]
        x_high = self.conv_high(x_high) # [10,128,32,64] / [10,128,128,256]
        x = x_low + x_high # [10,128,32,64] / [10,128,128,256]
        x = F.relu(x, inplace=True)# [10,128,32,64] / [10,128,128,256]
        x_low_cls = self.conv_low_cls(x_low) # [10,19,32,64] / [10,19,128,256]

        return x, x_low_cls


# x = torch.randn(10, 3, 1024,2048)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = ICNet(nclass = 19, backbone='resnet50').to(device)
# outputs = model(x.to(device))
# for i in outputs:
#     print(i.size())
