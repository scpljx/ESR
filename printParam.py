import torch

from models.resnet_fsr_1 import ResNet18_FSR_1
from models.resnet import ResNet18
from models.vgg_fsr import vgg16_FSR

from thop import profile


def main():
    model = vgg16_FSR()
    x = torch.randn(1,3,32,32)
    flops, params = profile(model, inputs=(x,))
    print(flops)
    print(params)


if __name__ == '__main__':
    main()