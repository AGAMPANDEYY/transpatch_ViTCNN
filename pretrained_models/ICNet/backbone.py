import torch.nn as nn
import torchvision

class SegBaseModel(nn.Module):
    """Base Model for Semantic Segmentation

    Parameters
    ----------
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    """

    def __init__(self, nclass, layers = [4,8], backbone='mobileNetv2'):
        super(SegBaseModel, self).__init__()
        dilated = True
        self.nclass = nclass
        self.layers = layers
        if backbone == 'mobileNetv2':
            self.pretrained = torchvision.models.mobilenet_v2(weights='MobileNet_V2_Weights.IMAGENET1K_V2')
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))

    def base_forward(self,x):
        """forwarding pre-trained network"""
        feat = []
        for i in range(19):
          x = self.pretrained.features[i](x)
          if i in self.layers:
            feat.append(x)
        return feat

    def evaluate(self, x):
        """evaluating network with inputs and targets"""
        return self.forward(x)[0]

    def demo(self, x):
        pred = self.forward(x)
        return pred

# x = torch.randn(10, 3, 1024,2048)
# model = SegBaseModel(nclass = 19, backbone='mobileNetv2')
# outputs = model(x)



class SegBaseModel_resnet(nn.Module):
    r"""Base Model for Semantic Segmentation

    Parameters
    ----------
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    """

    def __init__(self, nclass, backbone='resnet50'):
        super(SegBaseModel_resnet, self).__init__()
        dilated = True
        self.nclass = nclass
        if backbone == 'resnet50':
            #self.pretrained = resnet50_v1s(pretrained=pretrained_base, dilated=dilated, **kwargs)
            self.pretrained = torchvision.models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V2')
        elif backbone == 'resnet101':
            #self.pretrained = resnet101_v1s(pretrained=pretrained_base, dilated=dilated, **kwargs)
            self.pretrained = torchvision.models.resnet101(weights='ResNet101_Weights.IMAGENET1K_V2')
        elif backbone == 'resnet152':
            #self.pretrained = resnet152_v1s(pretrained=pretrained_base, dilated=dilated, **kwargs)
            self.pretrained = torchvision.models.resnet152(weights='ResNet152_Weights.IMAGENET1K_V2')
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))

    def base_forward(self,x):
        """forwarding pre-trained network"""
        # ResNet
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        c1 = self.pretrained.layer1(x)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        c4 = self.pretrained.layer4(c3)
        return c1, c2, c3, c4


    def evaluate(self, x):
        """evaluating network with inputs and targets"""
        return self.forward(x)[0]

    def demo(self, x):
        pred = self.forward(x)
        return pred