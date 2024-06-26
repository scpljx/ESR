import os, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

from models.BaseModel import BaseModelDNN


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


parser = argparse.ArgumentParser(description='Configuration')
parser.add_argument('--load_name', type=str, help='specify checkpoint load name')
parser.add_argument('--model', default='resnet18')
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--tau', default=0.1, type=float)
parser.add_argument('--bs', default=128, type=int, help='batch size')
parser.add_argument('--device', default=0, type=int)
parser.add_argument('--version', type=str, default='standard')
parser.add_argument('--save_dir', type=str, default='./eval_aa')
parser.add_argument('--log_path', type=str, default='./eval_aa/result.txt')

parser.add_argument('--norm', type=str, default='Linf', choices=['L2', 'Linf'])
parser.add_argument('--epsilon', type=float, default=8. / 255.)
parser.add_argument('--individual', default=False, action='store_true')

args = parser.parse_args()

device = 'cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu'


if args.model == 'resnet18':
    from models.resnet import ResNet18
    net = ResNet18

elif args.model == 'vgg16':
    from models.vgg_fsr import vgg16_FSR
    net = vgg16_FSR

elif args.model == 'wideresnet34':
    from models.wideresnet34_fsr import WideResNet34_FSR
    net = WideResNet34_FSR

elif args.model == 'resnet18_1':
    from models.resnet_fsr_1 import ResNet18_FSR_1
    net = ResNet18_FSR_1


if args.dataset == 'cifar10':
    image_size = (32, 32)
    num_classes = 10
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False)

elif args.dataset == 'svhn':
    image_size = (32, 32)
    num_classes = 10
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    testset = torchvision.datasets.SVHN(root='../data', split='test', download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False)
elif args.dataset == 'cifar100':
    image_size = (32, 32)
    num_classes = 100
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    testset = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False)



def get_pred(out, labels):
    pred = out.sort(dim=-1, descending=True)[1][:, 0]
    second_pred = out.sort(dim=-1, descending=True)[1][:, 1]
    adv_label = torch.where(pred == labels, second_pred, pred)

    return adv_label


class CE_loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, logits_final, target):
        loss = F.cross_entropy(logits_final, target)

        return loss


class CW_loss(nn.Module):
    def __init__(self, num_classes=10) -> None:
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits_final, target):
        loss = self._cw_loss(logits_final, target, num_classes=self.num_classes)

        return loss

    def _cw_loss(self, output, target, confidence=50, num_classes=10):
        target = target.data
        target_onehot = torch.zeros(target.size() + (num_classes,))
        target_onehot = target_onehot.to(device)
        target_onehot.scatter_(1, target.unsqueeze(1), 1.)
        target_var = Variable(target_onehot, requires_grad=False)
        real = (target_var * output).sum(1)
        other = ((1. - target_var) * output - target_var * 10000.).max(1)[0]
        loss = -torch.clamp(real - other + confidence, min=0.)  # equiv to max(..., 0.)
        loss = torch.sum(loss)
        return loss


class Classifier(BaseModelDNN):
    def __init__(self) -> None:
        super(BaseModelDNN).__init__()
        self.net = net(num_classes=num_classes).to(device)
        self.set_requires_grad([self.net], False)

    def predict(self, x):
        return self.net(x)


def main(test_loader):
    model = Classifier()
    checkpoint = torch.load('./weights/{}/{}/{}.pth'.format(args.dataset, args.model, args.load_name, map_location=device))
    model.net.load_state_dict(checkpoint)
    model.net.eval()

    # load attack
    from autoattack import AutoAttack
    adversary = AutoAttack(model.predict, norm=args.norm, eps=args.epsilon, log_path=args.log_path)

    l = [x for (x, y) in test_loader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_loader]
    y_test = torch.cat(l, 0)

    # cheap version
    # example of custom version
    if args.version == 'custom':
        adversary.attacks_to_run = ['apgd-ce', 'fab']
        adversary.apgd.n_restarts = 2
        adversary.fab.n_restarts = 2

    # run attack and save images
    if not args.individual:
        adv_complete = adversary.run_standard_evaluation(x_test[:10000], y_test[:10000],
                                                         bs=args.bs)
        torch.save({'adv_complete': adv_complete}, '{}/{}_{}_1_{}_eps_{:.5f}.pth'.format(
            args.save_dir, 'aa', args.version, adv_complete.shape[0], args.epsilon))


if __name__ == '__main__':
    main(testloader)