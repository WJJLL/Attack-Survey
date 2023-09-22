from __future__ import print_function
import argparse
import os
from math import log10
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
#from models import ResnetGenerator, weights_init
from material.models.generators import ResnetGenerator, weights_init
# from data import get_training_set, get_test_set
import torch.backends.cudnn as cudnn
import math
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm

torch.autograd.set_detect_anomaly(True)

# Training settings
parser = argparse.ArgumentParser(description='generative adversarial perturbations')
parser.add_argument('--imagenetTrain', type=str, default='/home/roar/Y_Exp/Data/imagenet/IN', help='ImageNet train root')
parser.add_argument('--imagenetVal', type=str, default='/home/roar/Y_Exp/Data/imagenet/val', help='ImageNet val root')
parser.add_argument('--batchSize', type=int, default=20, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=16, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--optimizer', type=str, default='adam', help='optimizer: "adam" or "sgd"')
parser.add_argument('--lr', type=float, default=0.0002, help='Learning Rate. Default=0.002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--MaxIter', type=int, default=100, help='Iterations in each Epoch')
parser.add_argument('--MaxIterTest', type=int, default=200, help='Iterations in each Epoch')
parser.add_argument('--mag_in', type=float, default=10.0, help='l_inf magnitude of perturbation')
parser.add_argument('--expname', type=str, default='tempname', help='experiment name, output folder')
parser.add_argument('--checkpoint', action='store_true', help='path to starting checkpoint')
parser.add_argument('--foolmodel', type=str, default='incv3', help='model to fool: "incv3", "vgg16", or "vgg19"')
parser.add_argument('--mode', type=str, default='train', help='mode: "train" or "test"')
parser.add_argument('--perturbation_type', type=str, help='"universal" or "imdep" (image dependent)')
parser.add_argument('--target', type=int, default=-1, help='target class: -1 if untargeted, 0..999 if targeted')
parser.add_argument('--gpu_ids', help='gpu ids: e.g. 0 or 0,1 or 1,2.', type=str, default='1')
parser.add_argument('--path_to_U_noise', type=str, default='', help='path to U_input_noise.txt (only needed for universal)')
parser.add_argument('--explicit_U', type=str, default='', help='Path to a universal perturbation to use')
opt = parser.parse_args()

print(opt)

cudnn.benchmark = True
torch.cuda.manual_seed(opt.seed)

MaxIter = opt.MaxIter
MaxIterTest = opt.MaxIterTest
gpulist = [int(i) for i in opt.gpu_ids.split(',')]
n_gpu = len(gpulist)
print('Running with n_gpu: ', n_gpu)

# define normalization means and stddevs
model_dimension = 299 if opt.foolmodel == 'incv3' else 256
center_crop = 299 if opt.foolmodel == 'incv3' else 224

mean_arr = [0.485, 0.456, 0.406]
stddev_arr = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=mean_arr,
                                 std=stddev_arr)

data_transform = transforms.Compose([
    transforms.Resize(model_dimension),
    transforms.CenterCrop(center_crop),
    transforms.ToTensor(),
    normalize,
])
print('===> Loading datasets')
test_set = torchvision.datasets.ImageFolder(root = opt.imagenetVal, transform = data_transform)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=True)
print()

if opt.foolmodel == 'incv3':
    pretrained_clf = torchvision.models.inception_v3(pretrained=True)
elif opt.foolmodel == 'vgg16':
    pretrained_clf = torchvision.models.vgg16(pretrained=True)
elif opt.foolmodel == 'vgg19':
    pretrained_clf = torchvision.models.vgg19(pretrained=True)
elif opt.foolmodel == 'resnet50':
    pretrained_clf = torchvision.models.resnet50(pretrained=True)
elif opt.foolmodel == 'resnet18':
    pretrained_clf = torchvision.models.resnet18(pretrained=True)
elif opt.foolmodel == 'resnet152':
    pretrained_clf = torchvision.models.resnet152(pretrained=True)
elif opt.foolmodel == 'googlenet':
    pretrained_clf = torchvision.models.googlenet(pretrained=True)
elif opt.foolmodel == 'densenet121':
    pretrained_clf = torchvision.models.densenet121(pretrained=True)


pretrained_clf = pretrained_clf.cuda(gpulist[0])

pretrained_clf.eval()
pretrained_clf.volatile = True

# magnitude
mag_in = opt.mag_in

def normalize_and_scale(delta_im, mode='train'):
    if opt.foolmodel == 'incv3':
        delta_im = nn.ConstantPad2d((0,-1,-1,0),0)(delta_im) # crop slightly to match inception

    delta_im = delta_im + 1 # now 0..2
    delta_im = delta_im * 0.5 # now 0..1

    # normalize image color channels
    for c in range(3):
        delta_im[:,c,:,:] = (delta_im[:,c,:,:].clone() - mean_arr[c]) / stddev_arr[c]

    # threshold each channel of each image in deltaIm according to inf norm
    # do on a per image basis as the inf norm of each image could be different
    bs = opt.batchSize if (mode == 'train') else opt.testBatchSize
    for i in range(bs):
        # do per channel l_inf normalization
        for ci in range(3):
            l_inf_channel = delta_im[i,ci,:,:].detach().abs().max()
            mag_in_scaled_c = mag_in/(255.0*stddev_arr[ci])
            gpu_id = gpulist[1] if n_gpu > 1 else gpulist[0]
            delta_im[i,ci,:,:] = delta_im[i,ci,:,:].clone() * np.minimum(1.0, mag_in_scaled_c / l_inf_channel.cpu().numpy())

    return delta_im


import logging
logger = logging.getLogger(__name__)
folder = os.path.exists('./subsrc')
if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
    os.makedirs('./subsrc')
logfile = 'subsrc/{}_to_{}_iter_{}.log'.format(opt.expname,opt.foolmodel,opt.MaxIter)
logging.basicConfig(
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO,
    filename=logfile)

print('===> Building model')
netG = ResnetGenerator(3, 3, opt.ngf, norm_type='batch', act_type='relu', gpu_ids=gpulist)


targets = [24,99,245,344,471,555,661,701]
total_tfr = 0
total_ntfr = 0
total_tfr_f = 0
for idx, target in enumerate(targets):
    logger.info('Epsilon \t Target \t ntFR. \t tFR.  \t tFR_filter')
    # Evaluation
    correct_orig = 0
    fooled = 0
    total = 0
    target_rate = 0
    target_filter_rate = 0
    correct_recon = 0

    path = opt.expname + "/netG_model_epoch_{}_".format(opt.nEpochs) + '_MaxIter_{}'.format(opt.MaxIter) + str(
        target) + ".pth"
    netG.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
    logging.info("=> loaded checkpoint '{}'".format(path))
    for i, (image, label) in enumerate(testing_data_loader):
        non_target_class_idxs = [i_label != target for i_label in label]
        non_target_class_mask = torch.Tensor(non_target_class_idxs) == True

        netG=netG.cuda(gpulist[0])
        netG.eval()

        image = image.cuda(gpulist[0])
        label = label.cuda(gpulist[0])

        if opt.perturbation_type == 'imdep':
            delta_im = netG(image)
            delta_im = normalize_and_scale(delta_im, 'test')

        recons = torch.add(image.cuda(gpulist[0]), delta_im[0:image.size(0)].cuda(gpulist[0]))

        # do clamping per channel
        for cii in range(3):
            recons[:, cii, :, :] = recons[:, cii, :, :].clone().clamp(image[:, cii, :, :].min(),
                                                                      image[:, cii, :, :].max())
        outputs_recon = pretrained_clf(recons.cuda(gpulist[0])).detach()
        outputs_orig = pretrained_clf(image.cuda(gpulist[0])).detach()

        _, predicted_recon = torch.max(outputs_recon, 1)
        _, predicted_orig = torch.max(outputs_orig, 1)
        total += image.size(0)
        correct_recon += (predicted_recon == label).sum()
        correct_orig += (predicted_orig == label).sum()

        fooled += (predicted_recon != predicted_orig).sum()

        target_label = torch.LongTensor(image.size(0))
        target_label.fill_(target)
        target_label = target_label.cuda(gpulist[0])
        target_rate += torch.sum(outputs_recon.argmax(dim=-1) == target_label).item()

        if torch.sum(non_target_class_mask) > 0:
            gt_non_target_class = label[non_target_class_mask]
            pert_output_non_target_class = outputs_recon[non_target_class_mask]
            target_cl = torch.ones_like(gt_non_target_class) * target
            target_filter_rate += torch.sum(pert_output_non_target_class.argmax(dim=-1) == target_cl).item()

    print('Targeted Fooling ratio: %.2f%%' % (100.0 * float(target_rate) / float(total)))
    print('Targeted Fooling ratio (filter): %.2f%%' % (100.0 * float(target_filter_rate) / 49950))
    logger.info(' %d\t            %d\t  %.4f\t   %.4f\t  %.4f ',
                int(mag_in), target, 100.0 * float(fooled) / float(total),
                (100.0 * float(target_rate) / float(total)), (100.0 * float(target_filter_rate) / 49950))

    total_tfr += (100.0 * float(target_rate) / float(total))
    total_ntfr += (100.0 * float(fooled) / float(total))
    total_tfr_f += (100.0 * float(target_filter_rate) / 49950)

logger.info('*' * 100)
logger.info('Average Target Transferability')
logger.info('*' * 100)
logger.info(' %d\t            %d\t  %.4f\t   %.4f\t  %.4f ',
            int(mag_in), target, total_ntfr / 8, total_tfr / 8, total_tfr_f / 8)


