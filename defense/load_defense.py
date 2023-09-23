from robustness import model_utils,datasets
import torchvision
import torch
def load_defense(target_model):
    if target_model == 'AT_LINF.5':
        ds = datasets.ImageNet('')
        model, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=ds,
                                                      resume_path='../pretrained_models/resnet50_linf_eps0.5.ckpt')
        pretrained_clf = model.model
    elif target_model == 'AT_LINF1':
        ds = datasets.ImageNet('')
        model, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=ds,
                                                      resume_path='../pretrained_models/resnet50_linf_eps1.0.ckpt')
        pretrained_clf = model.model
    elif target_model == 'AT_L2.1':
        ds = datasets.ImageNet('')
        model, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=ds,
                                                      resume_path='../pretrained_models/resnet50_l2_eps0.1.ckpt')
        pretrained_clf = model.model
    elif target_model == 'AT_L2.5':
        ds = datasets.ImageNet('')
        model, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=ds,
                                                      resume_path='../pretrained_models/resnet50_l2_eps0.5.ckpt')
        pretrained_clf = model.model
    elif target_model == 'SIN':
        pretrained_clf = torchvision.models.resnet50(pretrained=False)
        pretrained_clf = torch.nn.DataParallel(pretrained_clf)
        checkpoint = torch.load('../pretrained_models/resnet50_train_60_epochs-c8e5653e.pth.tar')
        pretrained_clf.load_state_dict(checkpoint["state_dict"])
    elif target_model == 'SIN-IN':
        pretrained_clf = torchvision.models.resnet50(pretrained=False)
        pretrained_clf = torch.nn.DataParallel(pretrained_clf)
        checkpoint = torch.load('../pretrained_models/resnet50_train_45_epochs_combined_IN_SF-2a0d100e.pth.tar')
        pretrained_clf.load_state_dict(checkpoint["state_dict"])
    elif target_model == 'Augmix':
        pretrained_clf = torchvision.models.resnet50(pretrained=False)
        pretrained_clf = torch.nn.DataParallel(pretrained_clf)
        checkpoint = torch.load('../pretrained_models/checkpoint.pth.tar')
        pretrained_clf.load_state_dict(checkpoint["state_dict"])
    return pretrained_clf
