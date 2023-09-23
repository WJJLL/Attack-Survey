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
from material.models.generators import ResnetGenerator, weights_init
# from data import get_training_set, get_test_set
import torch.backends.cudnn as cudnn
import math
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
from dataset_utils.coco import CocoDetection
torch.autograd.set_detect_anomaly(True)
from config import COCO_2017_TRAIN_IMGS, COCO_2017_VAL_IMGS, COCO_2017_TRAIN_ANN, COCO_2017_VAL_ANN
from custom_loss import LogitLoss,TargetRelativeCrossEntropy, BoundedLogitLossFixedRef, \
    BoundedLogitLoss_neg, CosSimLoss, RelativeCrossEntropy ,FeatureLayer,CrossEntropyLeastLikely,NegativeCrossEntropy
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
# parser.add_argument('--checkpoint', type=str, default='', help='path to starting checkpoint')
parser.add_argument('--checkpoint', action='store_true', help='path to starting checkpoint')
parser.add_argument('--foolmodel', type=str, default='incv3', help='model to fool: "incv3", "vgg16", or "vgg19"')
parser.add_argument('--mode', type=str, default='train', help='mode: "train" or "test"')
parser.add_argument('--perturbation_type', type=str, help='"universal" or "imdep" (image dependent)')
parser.add_argument('--target', type=int, default=-1, help='target class: -1 if untargeted, 0..999 if targeted')
parser.add_argument('--gpu_ids', help='gpu ids: e.g. 0 or 0,1 or 1,2.', type=str, default='1')
parser.add_argument('--path_to_U_noise', type=str, default='', help='path to U_input_noise.txt (only needed for universal)')
parser.add_argument('--explicit_U', type=str, default='', help='Path to a universal perturbation to use')
parser.add_argument('--train_dir', default='paintings', help='paintings, comics, imagenet')
parser.add_argument('--loss_function', default='ce',
                    help='Used loss function for source classes: (default: cw_logit)')
parser.add_argument('--nrp', action='store_true',
                    help='Target a specific class (default: False)')
opt = parser.parse_args()

print(opt)

# train loss history
train_loss_history = []
test_loss_history = []
test_acc_history = []
test_fooling_history = []
best_fooling = 0
itr_accum = 0

# make directories
if not os.path.exists(opt.expname):
    os.mkdir(opt.expname)

if opt.perturbation_type == 'universal':
    if not os.path.exists(opt.expname + '/U_out'):
        os.mkdir(opt.expname + '/U_out')

cudnn.benchmark = True
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)

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
if opt.mode == 'train':
    print('===> Loading training datasets')
    if opt.train_dir == 'imagenet':
        train_set = torchvision.datasets.ImageFolder(root=opt.imagenetTrain, transform=data_transform)

    elif opt.train_dir == 'coco':
        train_set = CocoDetection(root=COCO_2017_TRAIN_IMGS,
                                  annFile=COCO_2017_TRAIN_ANN,
                                  transform=data_transform)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize,
                                      shuffle=True)
print('===> Loading test datasets')
test_set = torchvision.datasets.ImageFolder(root = opt.imagenetVal, transform = data_transform)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=True)

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

pretrained_clf = pretrained_clf.cuda()
pretrained_clf.eval()
pretrained_clf.volatile = True

# magnitude
mag_in = opt.mag_in

print('===> Building model')


if not opt.explicit_U:
    # will use model paralellism if more than one gpu specified
    netG = ResnetGenerator(3, 3, opt.ngf, norm_type='batch', act_type='relu', gpu_ids=gpulist)

    # resume from checkpoint if specified
    if opt.checkpoint:
        path = opt.expname + "/netG_model_epoch_{}_".format(opt.nEpochs) + '_MaxIter_{}'.format(opt.MaxIter) + str(opt.target) + ".pth"
        print("=> loading checkpoint '{}'".format(path))
        netG.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
        print("=> loaded checkpoint '{}'".format(path))
    else:
        netG.apply(weights_init)

    # setup optimizer
    if opt.optimizer == 'adam':
        optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    elif opt.optimizer == 'sgd':
        optimizerG = optim.SGD(netG.parameters(), lr=opt.lr, momentum=0.9)


    ##untargeted_loss##
    if opt.loss_function == "neg_ce":
        criterion_pre = NegativeCrossEntropy()
    elif opt.loss_function == "bounded_logit_neg":
        criterion_pre = BoundedLogitLoss_neg(num_classes=1000, confidence=10, use_cuda=True)
    elif opt.loss_function == "cos_sim":
        criterion_pre = CosSimLoss()
    elif opt.loss_function == "r_ce":
        criterion_pre = RelativeCrossEntropy()
    elif opt.loss_function == "cell":
        criterion_pre = CrossEntropyLeastLikely()
    elif opt.loss_function == "feature_layers":
        criterion_pre = FeatureLayer()

    ##targeted_loss##
    if opt.loss_function == "ce":
        criterion_pre = torch.nn.CrossEntropyLoss()
    elif opt.loss_function == "bounded_logit_fixed_ref":
        criterion_pre = BoundedLogitLossFixedRef(num_classes=1000, confidence=10,
                                             use_cuda=True)
    elif opt.loss_function == "t_r_ce":
        criterion_pre = TargetRelativeCrossEntropy()
    elif opt.loss_function == "logit":
        criterion_pre = LogitLoss(num_classes=1000, use_cuda=True)

    # criterion_pre = nn.CrossEntropyLoss()
    criterion_pre = criterion_pre.cuda()

    # fixed noise for universal perturbation
    if opt.perturbation_type == 'universal':
        noise_data = np.random.uniform(0, 255, center_crop * center_crop * 3)
        if opt.checkpoint:
            if opt.path_to_U_noise:
                noise_data = np.loadtxt(opt.path_to_U_noise)
                np.savetxt(opt.expname + '/U_input_noise.txt', noise_data)
            else:
                noise_data = np.loadtxt(opt.expname + '/U_input_noise.txt')
        else:
            np.savetxt(opt.expname + '/U_input_noise.txt', noise_data)
        im_noise = np.reshape(noise_data, (3, center_crop, center_crop))
        im_noise = im_noise[np.newaxis, :, :, :]
        im_noise_tr = np.tile(im_noise, (opt.batchSize, 1, 1, 1))
        noise_tr = torch.from_numpy(im_noise_tr).type(torch.FloatTensor).cuda()

        im_noise_te = np.tile(im_noise, (opt.testBatchSize, 1, 1, 1))
        noise_te = torch.from_numpy(im_noise_te).type(torch.FloatTensor).cuda()

def train(epoch):
    netG.train()
    global itr_accum
    global optimizerG

    for itr, (image, _) in enumerate(training_data_loader, 1):
        if itr > MaxIter:
            break

        if opt.target != -1:
            target_label = torch.LongTensor(image.size(0))
            target_label.fill_(opt.target)
            target_label = target_label.cuda()

        itr_accum += 1
        if opt.optimizer == 'sgd':
            lr_mult = (itr_accum // 1000) + 1
            optimizerG = optim.SGD(netG.parameters(), lr=opt.lr/lr_mult, momentum=0.9)

        image = image.cuda()

        ## generate per image perturbation from fixed noise
        if opt.perturbation_type == 'universal':
            delta_im = netG(noise_tr)
        else:
            delta_im = netG(image)

        delta_im = normalize_and_scale(delta_im, 'train')

        netG.zero_grad()

        recons = torch.add(image.cuda(), delta_im.cuda())

        # do clamping per channel
        for cii in range(3):
            recons[:,cii,:,:] = recons[:,cii,:,:].clone().clamp(image[:,cii,:,:].min(), image[:,cii,:,:].max())

        output_pretrained = pretrained_clf(recons.cuda())

        if criterion_pre._get_name() == "FeatureLayer":
            loss, output = criterion_pre(pretrained_clf, recons.cuda())
        elif criterion_pre._get_name() in ['BoundedLogitLoss_neg', 'NegativeCrossEntropy', 'CosSimLoss',
                                       'RelativeCrossEntropy', 'CrossEntropyLeastLikely']:
            output_ori = pretrained_clf(image.cuda()).detach()
            loss = criterion_pre(output_ori, output_pretrained)
        elif criterion_pre._get_name() == 'TargetRelativeCrossEntropy':
            output_ori = pretrained_clf(image.cuda())
            loss = criterion_pre(output_ori, output_pretrained, target_label)
        else:
            ###'BoundedLogitLossFixedRef' or 'CE'####
            loss = criterion_pre(output_pretrained, target_label)

        loss.backward()
        optimizerG.step()

        train_loss_history.append(loss.item())
        print("===> Epoch[{}]({}/{}) loss: {:.4f}".format(epoch, itr, len(training_data_loader), loss.item()))

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
mean_tensor = torch.ones(1, 3, *(224,224))
for idx in range(3):
    mean_tensor[:,idx] *= mean[idx]
std_tensor = torch.ones(1, 3, *(224,224))
for idx in range(3):
    std_tensor[:,idx] *= std[idx]
mean_tensor, std_tensor = mean_tensor.cuda(), std_tensor.cuda()
import sys
sys.path.append(os.path.abspath('../defense'))
from NRP import *


def test():
    if not opt.explicit_U:
        netG.eval()
    correct_recon = 0
    correct_orig = 0
    fooled = 0
    total = 0
    if opt.nrp == True:
        purifier = NRP(3, 3, 64, 23)
        purifier.load_state_dict(torch.load('/home/imt-3090-1/zmluo/attack/pretrained_models/NRP.pth'))
        purifier = purifier.cuda()


    if opt.perturbation_type == 'universal':
        if opt.explicit_U:
            U_loaded = torch.load(opt.explicit_U)
            U_loaded = U_loaded.expand(opt.testBatchSize, U_loaded.size(1), U_loaded.size(2), U_loaded.size(3))
            delta_im = normalize_and_scale(U_loaded, 'test')
        else:
            delta_im = netG(noise_te)
            delta_im = normalize_and_scale(delta_im, 'test')

    for itr, (image, class_label) in enumerate(testing_data_loader):
        image = image.cuda()

        if opt.perturbation_type == 'imdep':
            delta_im = netG(image)
            delta_im = normalize_and_scale(delta_im, 'test')

        recons = torch.add(image.cuda(), delta_im[0:image.size(0)].cuda())

        # do clamping per channel
        for cii in range(3):
            recons[:,cii,:,:] = recons[:,cii,:,:].clone().clamp(image[:,cii,:,:].min(), image[:,cii,:,:].max())

        if opt.nrp == True:
            orig_adv = recons * std_tensor + mean_tensor
            # using purifier to input
            with torch.no_grad():
                adv_orig_img = purifier(orig_adv).detach()
            # Put image into normalized form
            recons = (adv_orig_img - mean_tensor) / std_tensor

        outputs_recon = pretrained_clf(recons.cuda()).detach()
        outputs_orig = pretrained_clf(image.cuda()).detach()
        _, predicted_recon = torch.max(outputs_recon, 1)
        _, predicted_orig = torch.max(outputs_orig, 1)
        total += image.size(0)
        correct_recon += (predicted_recon == class_label.cuda()).sum()
        correct_orig += (predicted_orig == class_label.cuda()).sum()

        if opt.target == -1:
            fooled += (predicted_recon != predicted_orig).sum()
        else:
            fooled += (predicted_recon == opt.target).sum()

        if itr % 50 == 1:
            print('Images evaluated:', (itr*opt.testBatchSize))
            # undo normalize image color channels
            delta_im_temp = torch.zeros(delta_im.size())
            for c2 in range(3):
                recons[:,c2,:,:] = (recons[:,c2,:,:] * stddev_arr[c2]) + mean_arr[c2]
                image[:,c2,:,:] = (image[:,c2,:,:] * stddev_arr[c2]) + mean_arr[c2]
                delta_im_temp[:,c2,:,:] = (delta_im[:,c2,:,:] * stddev_arr[c2]) + mean_arr[c2]
            if not os.path.exists(opt.expname):
                os.mkdir(opt.expname)

            post_l_inf = (recons - image[0:recons.size(0)]).abs().max() * 255.0
            print("Specified l_inf:", mag_in, "| maximum l_inf of generated perturbations: %.2f" % (post_l_inf.item()))

            # torchvision.utils.save_image(recons, opt.expname+'/reconstructed_{}.png'.format(itr))
            # torchvision.utils.save_image(image, opt.expname+'/original_{}.png'.format(itr))
            # torchvision.utils.save_image(delta_im_temp, opt.expname+'/delta_im_{}.png'.format(itr))
            # print('Saved images.')

    test_acc_history.append((100.0 * correct_recon / total))
    test_fooling_history.append((100.0 * fooled / total))
    print('Accuracy of the pretrained network on reconstructed images: %.2f%%' % (100.0 * float(correct_recon) / float(total)))
    print('Accuracy of the pretrained network on original images: %.2f%%' % (100.0 * float(correct_orig) / float(total)))
    if opt.target == -1:
        print('Fooling ratio: %.2f%%' % (100.0 * float(fooled) / float(total)))
    else:
        print('Top-1 Target Accuracy: %.2f%%' % (100.0 * float(fooled) / float(total)))

    fooling_rate = (100.0 * float(fooled) / float(total))

    return fooling_rate

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


def checkpoint_dict(epoch):
    netG.eval()
    global best_fooling
    if not os.path.exists(opt.expname):
        os.mkdir(opt.expname)

    task_label = "foolrat" if opt.target == -1 else "top1target"

    net_g_model_out_path = opt.expname + "/netG_model_epoch_{}_".format(epoch) + task_label + "_{}.pth".format(test_fooling_history[epoch-1])
    if opt.perturbation_type == 'universal':
        u_out_path = opt.expname + "/U_out/U_epoch_{}_".format(epoch) + task_label + "_{}.pth".format(test_fooling_history[epoch-1])
    if test_fooling_history[epoch-1] > best_fooling:
        best_fooling = test_fooling_history[epoch-1]
        torch.save(netG.state_dict(), net_g_model_out_path)
        if opt.perturbation_type == 'universal':
            torch.save(netG(noise_te[0:1]), u_out_path)
        print("Checkpoint saved to {}".format(net_g_model_out_path))
    else:
        print("No improvement:", test_fooling_history[epoch-1], "Best:", best_fooling)


def print_history():
    # plot history for training loss
    if opt.mode == 'train':
        plt.plot(train_loss_history)
        plt.title('Model Training Loss')
        plt.ylabel('Loss')
        plt.xlabel('Iteration')
        plt.legend(['Training Loss'], loc='upper right')
        plt.savefig(opt.expname+'/reconstructed_loss_'+opt.mode+'.png')
        plt.clf()

    # plot history for classification testing accuracy and fooling ratio
    # import ipdb
    # ipdb.set_trace()
    plt.plot(test_acc_history)
    plt.title('Model Testing Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Testing Classification Accuracy'], loc='upper right')
    plt.savefig(opt.expname+'/reconstructed_acc_'+opt.mode+'.png')
    plt.clf()

    plt.plot(test_fooling_history)
    plt.title('Model Testing Fooling Ratio')
    plt.ylabel('Fooling Ratio')
    plt.xlabel('Epoch')
    plt.legend(['Testing Fooling Ratio'], loc='upper right')
    plt.savefig(opt.expname+'/reconstructed_foolrat_'+opt.mode+'.png')
    print("Saved plots.")

def save_dict(epoch):
    netG.eval()
    if not os.path.exists(opt.expname):
        os.mkdir(opt.expname)
    task_label = opt.target
    # task_label = "foolrat" if opt.target == -1 else "top1target"

    net_g_model_out_path = opt.expname + "/netG_model_epoch_{}_".format(epoch)+'_MaxIter_{}'.format(MaxIter)+ str(task_label) + ".pth"
    if opt.perturbation_type == 'universal':
        u_out_path = opt.expname + "/U_out/U_epoch_{}_".format(epoch) +'_MaxIter_{}'.format(MaxIter)+str(task_label) + ".pth"

    torch.save(netG.state_dict(), net_g_model_out_path)
    if opt.perturbation_type == 'universal':
        torch.save(netG(noise_te[0:1]), u_out_path)
    print("Checkpoint saved to {}".format(net_g_model_out_path))


if opt.mode == 'train':
    for epoch in range(1, opt.nEpochs + 1):
        train(epoch)
        save_dict(epoch)
        # print('Testing....')
        # test()
        # checkpoint_dict(epoch)
    # print_history()
    print('Testing...')
    test()

elif opt.mode == 'test':
    print('Testing...')
    test()
    # print_history()
