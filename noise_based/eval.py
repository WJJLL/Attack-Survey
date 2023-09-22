from __future__ import division
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision
import torchvision.models as models
import os,sys, time, random, copy
import numpy as np
import torch
import argparse
import torch.backends.cudnn as cudnn
import torch.nn as nn
from collections import OrderedDict
import math
from robustness import model_utils,datasets
from networks.uap import UAP
from utils.data import get_data_specs, get_data
from utils.utils import get_model_path, get_result_path
from utils.utils import print_log
from utils.network import get_network, set_parameter_requires_grad
from utils.network import get_num_parameters, get_num_non_trainable_parameters, get_num_trainable_parameters
from utils.training import train, save_checkpoint, metrics_evaluate
from utils.custom_loss import LogitLoss,TargetRelativeCrossEntropy, BoundedLogitLossFixedRef, \
    BoundedLogitLoss_neg, CosSimLoss, RelativeCrossEntropy ,FeatureLayer,CrossEntropyLeastLikely,NegativeCrossEntropy


def parse_arguments():
    parser = argparse.ArgumentParser(description='Trains a UAP')
    # pretrained
    parser.add_argument('--dataset', default='imagenet',
                        choices=['cifar10', 'cifar100', 'imagenet', 'coco', 'voc', 'places365'],
                        help='Used dataset to generate UAP (default: imagenet)')
    parser.add_argument('--pretrained_dataset', default='imagenet', choices=['cifar10', 'cifar100', 'imagenet'],
                        help='Used dataset to train the initial model (default: imagenet)')
    parser.add_argument('--pretrained_arch', default='vgg16',
                        help='Used model architecture: (default: vgg16)')
    parser.add_argument('--source_arch', default='vgg16',
                        help='Used model architecture: (default: vgg16)')
    parser.add_argument('--pretrained_seed', type=int, default=123,
                        help='Seed used in the generation process (default: 123)')
    # Parameters regarding UAP
    parser.add_argument('--epsilon', type=float, default=0.03922,
                        help='Norm restriction of UAP (default: 10/255)')
    parser.add_argument('--num_iterations', type=int, default=1000,
                        help='Number of iterations (default: 1000)')
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='Number of iterations (default: 5)')
    parser.add_argument('--result_subfolder', default='default', type=str,
                        help='result subfolder name')
    parser.add_argument('--postfix', default='',
                        help='Postfix to attach to result folder')
    # Optimization options
    parser.add_argument('--loss_function', default='ce',
                        help='Used loss function for source classes: (default: cw_logit)')
    parser.add_argument('--confidence', default=0., type=float,
                        help='Confidence value for C&W losses (default: 0.0)')
    parser.add_argument('--targeted', action='store_true',
                        help='Target a specific class (default: False)')
    parser.add_argument('--nrp', action='store_true',
                        help='Target a specific class (default: False)')
    parser.add_argument('--target_class', type=int, default=0,
                        help='Target class (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning Rate (default: 0.001)')
    parser.add_argument('--print_freq', default=200, type=int, metavar='N',
                        help='print frequency (default: 200)')
    parser.add_argument('--ngpu', type=int, default=1,
                        help='Number of used GPUs (0 = CPU) (default: 1)')
    parser.add_argument('--workers', type=int, default=6,
                        help='Number of data loading workers (default: 6)')
    args = parser.parse_args()

    args.use_cuda = args.ngpu > 0 and torch.cuda.is_available()

    if args.pretrained_seed is None:
        args.pretrained_seed = random.randint(1, 10000)
    return args


def main():
    args = parse_arguments()

    random.seed(args.pretrained_seed)
    torch.manual_seed(args.pretrained_seed)
    if args.use_cuda:
        torch.cuda.manual_seed_all(args.pretrained_seed)
    cudnn.benchmark = True

    # get the result path to store the results
    result_path = get_result_path(dataset_name=args.dataset,
                                  network_arch=args.source_arch,
                                  random_seed=args.pretrained_seed,
                                  result_subfolder=args.result_subfolder,
                                  postfix=args.postfix + '_' + str(args.target_class))

    # Init logger
    log_file_name = os.path.join(result_path, args.source_arch+'_'+args.pretrained_arch+'_'+str(args.target_class)+'_log.txt')
    print("Log file: {}".format(log_file_name))
    log = open(log_file_name, 'w')
    print_log('save path : {}'.format(result_path), log)
    state = {k: v for k, v in args._get_kwargs()}
    for key, value in state.items():
        print_log("{} : {}".format(key, value), log)
    print_log("Random Seed: {}".format(args.pretrained_seed), log)
    print_log("Python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("Torch  version : {}".format(torch.__version__), log)
    print_log("Cudnn  version : {}".format(torch.backends.cudnn.version()), log)

    _, pretrained_data_test = get_data(args.pretrained_dataset, args.pretrained_dataset)

    pretrained_data_test_loader = torch.utils.data.DataLoader(pretrained_data_test,
                                                              batch_size=args.batch_size,
                                                              shuffle=False,
                                                              num_workers=args.workers,
                                                              pin_memory=True)

    ##### Dataloader for training ####
    num_classes, (mean, std), input_size, num_channels = get_data_specs(args.pretrained_dataset)

    # data_train, _ = get_data(args.dataset, args.pretrained_dataset)

    ####################################
    # Init model, criterion, and optimizer
    print_log("=> Creating model '{}'".format(args.pretrained_arch), log)
    # get a path for loading the model to be attacked
    model_path = get_model_path(dataset_name=args.pretrained_dataset,
                                network_arch=args.pretrained_arch,
                                random_seed=args.pretrained_seed)
    model_weights_path = os.path.join(model_path, "checkpoint.pth.tar")


    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))

    if args.pretrained_arch in model_names:
        target_network = models.__dict__[args.pretrained_arch](pretrained=True)
    elif args.pretrained_arch == 'AT_LINF.5':
        ds = datasets.ImageNet('')
        model, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=ds,
                                                      resume_path='/home/imt-3090-1/zmluo/attack/pretrained_models/resnet50_linf_eps0.5.ckpt')
        target_network = model.model
    elif args.pretrained_arch == 'AT_LINF1':
        ds = datasets.ImageNet('')
        model, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=ds,
                                                      resume_path='/home/imt-3090-1/zmluo/attack/pretrained_models/resnet50_linf_eps1.0.ckpt')
        target_network = model.model
    elif args.pretrained_arch == 'AT_L2.1':
        ds = datasets.ImageNet('')
        model, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=ds,
                                                      resume_path='/home/imt-3090-1/zmluo/attack/pretrained_models/resnet50_l2_eps0.1.ckpt')
        target_network = model.model
    elif args.pretrained_arch == 'AT_L2.5':
        ds = datasets.ImageNet('')
        model, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=ds,
                                                      resume_path='/home/imt-3090-1/zmluo/attack/pretrained_models/resnet50_l2_eps0.5.ckpt')
        target_network = model.model
    elif args.pretrained_arch == 'SIN':
        target_network = torchvision.models.resnet50(pretrained=False)
        target_network = torch.nn.DataParallel(target_network)
        checkpoint = torch.load('/home/imt-3090-1/zmluo/attack/pretrained_models/resnet50_train_60_epochs-c8e5653e.pth.tar')
        target_network.load_state_dict(checkpoint["state_dict"])
    elif args.pretrained_arch == 'SIN-IN':
        target_network = torchvision.models.resnet50(pretrained=False)
        target_network = torch.nn.DataParallel(target_network)
        checkpoint = torch.load('/home/imt-3090-1/zmluo/attack/pretrained_models/resnet50_train_45_epochs_combined_IN_SF-2a0d100e.pth.tar')
        target_network.load_state_dict(checkpoint["state_dict"])
    elif args.pretrained_arch == 'Augmix':
        target_network = torchvision.models.resnet50(pretrained=False)
        target_network = torch.nn.DataParallel(target_network)
        checkpoint = torch.load('/home/imt-3090-1/zmluo/attack/pretrained_models/checkpoint.pth.tar')
        target_network.load_state_dict(checkpoint["state_dict"])
    else:
        assert (args.target_model in model_names), 'Please provide correct target model names: {}'.format(model_names)



    # print_log("=> Network :\n {}".format(target_network), log)
    target_network = torch.nn.DataParallel(target_network, device_ids=list(range(args.ngpu)))
    # Set the target model into evaluation mode
    target_network.eval()
    # Imagenet models use the pretrained pytorch weights
    if args.pretrained_dataset != "imagenet":
        network_data = torch.load(model_weights_path)
        target_network.load_state_dict(network_data['state_dict'])

    # Set all weights to not trainable
    set_parameter_requires_grad(target_network, requires_grad=False)

    non_trainale_params = get_num_non_trainable_parameters(target_network)
    trainale_params = get_num_trainable_parameters(target_network)
    total_params = get_num_parameters(target_network)
    print_log("Target Network Trainable parameters: {}".format(trainale_params), log)
    print_log("Target Network Non Trainable parameters: {}".format(non_trainale_params), log)
    print_log("Target Network Total # parameters: {}".format(total_params), log)

    print_log("=> Inserting Generator", log)

    generator = UAP(shape=(input_size, input_size),
                    num_channels=num_channels,
                    mean=mean,
                    std=std,
                    use_cuda=args.use_cuda)
    if args.targeted:
        result_path = os.path.join('./results/imagenet_targeted/',
                                   "{}_{}_{}_{}_{}".format(args.dataset, args.source_arch, args.pretrained_seed,
                                                          args.loss_function,
                                                          args.target_class))
    else:
        result_path = os.path.join('./results/imagenet_untargeted/',
                                   "{}_{}_{}_{}_0".format(args.dataset, args.source_arch, args.pretrained_seed,
                                                          args.loss_function))

    generator.load_state_dict(torch.load(os.path.join(result_path, 'checkpoint.pth.tar'))['state_dict'])

    print_log('Load Generator Path {}_{}_{}_{}_{}'.format(args.dataset, args.source_arch, args.pretrained_seed,
                                                          args.loss_function,
                                                          args.target_class),log)
    print_log("=> Load Generator  :\n {}".format(generator), log)
    non_trainale_params = get_num_non_trainable_parameters(generator)
    trainale_params = get_num_trainable_parameters(generator)
    total_params = get_num_parameters(generator)
    print_log("Generator Trainable parameters: {}".format(trainale_params), log)
    print_log("Generator Non Trainable parameters: {}".format(non_trainale_params), log)
    print_log("Generator Total # parameters: {}".format(total_params), log)

    perturbed_net = nn.Sequential(OrderedDict([('generator', generator), ('target_model', target_network)]))
    perturbed_net = torch.nn.DataParallel(perturbed_net, device_ids=list(range(args.ngpu)))

    non_trainale_params = get_num_non_trainable_parameters(perturbed_net)
    trainale_params = get_num_trainable_parameters(perturbed_net)
    total_params = get_num_parameters(perturbed_net)
    print_log("Perturbed Net Trainable parameters: {}".format(trainale_params), log)
    print_log("Perturbed Net Non Trainable parameters: {}".format(non_trainale_params), log)
    print_log("Perturbed Net Total # parameters: {}".format(total_params), log)

    # Set the target model into evaluation mode
    perturbed_net.module.target_model.eval()
    perturbed_net.module.generator.train()


    if args.use_cuda:
        target_network.cuda()
        generator.cuda()
        perturbed_net.cuda()

    # evaluate
    print_log("Final evaluation:", log)
    metrics_evaluate(data_loader=pretrained_data_test_loader,
                     target_model=target_network,
                     perturbed_model=perturbed_net,
                     targeted=args.targeted,
                     target_class=args.target_class,
                     log=log,
                     use_cuda=args.use_cuda,
                     nrp=args.nrp
                     )
    log.close()


if __name__ == '__main__':
    main()
