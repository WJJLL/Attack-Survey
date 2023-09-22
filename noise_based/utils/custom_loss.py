import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss, _Loss
def one_hot(class_labels, num_classes=None):
    if num_classes==None:
        return torch.zeros(len(class_labels), class_labels.max()+1).scatter_(1, class_labels.unsqueeze(1), 1.)
    else:
        return torch.zeros(len(class_labels), num_classes).scatter_(1, class_labels.unsqueeze(1), 1.)


class LogitLoss(_WeightedLoss):
    def __init__(self, num_classes, use_cuda=False):
        super(LogitLoss, self).__init__()
        self.num_classes = num_classes
        self.use_cuda = use_cuda

    def forward(self, input, target):
        one_hot_labels = one_hot(target.cpu(), num_classes=self.num_classes)
        if self.use_cuda:
            one_hot_labels = one_hot_labels.cuda()

        # Get the logit output value
        logits = (one_hot_labels * input).max(1)[0]
        # Increase the logit value
        return torch.mean(-logits)


class BoundedLogitLoss(_WeightedLoss):
    def __init__(self, num_classes, confidence, use_cuda=False):
        super(BoundedLogitLoss, self).__init__()
        self.num_classes = num_classes
        self.confidence = confidence
        self.use_cuda = use_cuda

    def forward(self, input, target):
        one_hot_labels = one_hot(target.cpu(), num_classes=self.num_classes)
        if self.use_cuda:
            one_hot_labels = one_hot_labels.cuda()

        target_logits = (one_hot_labels * input).sum(1)
        not_target_logits = ((1. - one_hot_labels) * input - one_hot_labels * 10000.).max(1)[0]
        logit_loss = torch.clamp(not_target_logits - target_logits, min=-self.confidence)
        return torch.mean(logit_loss)


class BoundedLogitLossFixedRef(_WeightedLoss):
    def __init__(self, num_classes, confidence, use_cuda=False):
        super(BoundedLogitLossFixedRef, self).__init__()
        self.num_classes = num_classes
        self.confidence = confidence
        self.use_cuda = use_cuda

    def forward(self, input, target):
        one_hot_labels = one_hot(target.cpu(), num_classes=self.num_classes)
        if self.use_cuda:
            one_hot_labels = one_hot_labels.cuda()

        target_logits = (one_hot_labels * input).sum(1)
        not_target_logits = ((1. - one_hot_labels) * input - one_hot_labels * 10000.).max(1)[0]
        logit_loss = torch.clamp(not_target_logits.data.detach() - target_logits, min=-self.confidence)
        return torch.mean(logit_loss)

class TargetRelativeCrossEntropy(_WeightedLoss):
    def __init__(self,):
        super(TargetRelativeCrossEntropy, self).__init__()

    def forward(self,input_oig ,input_adv,target):
        loss = torch.nn.CrossEntropyLoss()(input_adv,target) + torch.nn.CrossEntropyLoss()(input_oig, input_oig.argmax(dim=-1).detach())
        return loss

####################################Untargeted Loss##################################################

class BoundedLogitLoss_neg(_WeightedLoss):
    def __init__(self, num_classes, confidence, use_cuda=False):
        super(BoundedLogitLoss_neg, self).__init__()
        self.num_classes = num_classes
        self.confidence = confidence
        self.use_cuda = use_cuda

    def forward(self, input_ori,input):
        target = input_ori.argmax(dim=-1).detach()
        one_hot_labels = one_hot(target.cpu(), num_classes=self.num_classes)
        if self.use_cuda:
            one_hot_labels = one_hot_labels.cuda()

        target_logits = (one_hot_labels * input).sum(1)

        not_target_logits = ((1. - one_hot_labels) * input - one_hot_labels * 10000.).max(1)[0]
        logit_loss = torch.clamp(target_logits - not_target_logits, min=-self.confidence)
        return torch.mean(logit_loss)


class NegativeCrossEntropy(_WeightedLoss):
    def __init__(self,):
        super(NegativeCrossEntropy, self).__init__()

    def forward(self, input_ori,input):
        target = input_ori.argmax(dim=-1).detach()
        loss = -F.cross_entropy(input, target, weight=None, ignore_index=-100, reduction='elementwise_mean')
        return loss


class CosSimLoss(_WeightedLoss):
    def __init__(self,):
        super(CosSimLoss, self).__init__()

    def forward(self, input_ori, input_adv):
        loss = F.cosine_similarity(input_ori, input_adv)
        return torch.mean(loss)


class RelativeCrossEntropy(_WeightedLoss):
    def __init__(self,):
        super(RelativeCrossEntropy, self).__init__()

    def forward(self, input_ori, input_adv):
        label = input_ori.argmax(dim=-1).detach()
        loss = -F.cross_entropy(input_adv - input_ori, label)
        return torch.mean(loss)

class CrossEntropyLeastLikely(_WeightedLoss):
    def __init__(self,):
        super(CrossEntropyLeastLikely, self).__init__()

    def forward(self, input_ori, input_adv):
        # least likely class in nontargeted case
        _, target_label = torch.min(input_ori, 1)
        loss = torch.log(torch.nn.CrossEntropyLoss()(input_adv, target_label))
        return loss

def get_conv_layers(model):
    return [module for module in model.modules() if type(module) == nn.Conv2d]

class FeatureLayer(_WeightedLoss):
    def __init__(self):
        super(FeatureLayer, self).__init__()

    def forward(self,model,image):
        image = image.cuda()
        loss = torch.tensor(0.)
        activations = []
        remove_handles = []

        def activation_recorder_hook(self, input, output):
            activations.append(output)
            return None

        for conv_layer in get_conv_layers(model):
            handle = conv_layer.register_forward_hook(activation_recorder_hook)
            remove_handles.append(handle)

        model.eval()
        model.zero_grad()
        out = model(image)

        # unregister hook so activation tensors have no references
        for handle in remove_handles:
            handle.remove()

        loss = -sum(list(map(lambda activation: torch.log(torch.sum(activation * activation) / 2), activations)))
        return loss , out
