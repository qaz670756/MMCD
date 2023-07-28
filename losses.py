# https://github.com/tuantle/regression-losses-pytorch
import torch
import torch.nn.functional as F
import random
from sklearn.metrics import cohen_kappa_score


# from ssim import ssim


def ssim_loss(inputs, targets):
    return 1.0 - ssim(inputs.cuda(), targets.cuda())


def ssim_and_mse_loss(inputs, targets):
    return 1.0 - ssim(inputs.cuda(), targets.cuda()) + torch.nn.functional.mse_loss(inputs, targets)


def ssim_and_mae_loss(inputs, targets):
    return 1.0 - ssim(inputs.cuda(), targets.cuda()) + torch.nn.functional.l1_loss(inputs, targets)


class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t, exclude_zeros=False):
        ey_t = y_t - y_prime_t
        loss = torch.log(torch.cosh(ey_t + 1e-12))

        if exclude_zeros:
            n_pixels = torch.count_nonzero(y_prime_t)
            return torch.sum(loss) / n_pixels
        else:
            return torch.mean(loss)


class XTanhLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t, exclude_zeros=False):
        ey_t = y_t - y_prime_t
        loss = ey_t * torch.tanh(ey_t)

        if exclude_zeros:
            n_pixels = torch.count_nonzero(y_prime_t)
            return torch.sum(loss) / n_pixels
        else:
            return torch.mean(loss)


class XSigmoidLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t, exclude_zeros=False):
        ey_t = y_t - y_prime_t
        loss = 2 * ey_t * torch.sigmoid(ey_t) - ey_t

        if exclude_zeros:
            n_pixels = torch.count_nonzero(y_prime_t)
            return torch.sum(loss) / n_pixels
        else:
            return torch.mean(loss)


class AlgebraicLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t, exclude_zeros=False):
        ey_t = y_t - y_prime_t
        loss = ey_t * ey_t / torch.sqrt(1 + ey_t * ey_t)

        if exclude_zeros:
            n_pixels = torch.count_nonzero(y_prime_t)
            return torch.sum(loss) / n_pixels
        else:
            return torch.mean(loss)


######################NEW
# https://discuss.pytorch.org/t/pixelwise-weights-for-mseloss/1254
# https://discuss.pytorch.org/t/mse-l2-loss-on-two-masked-images/28417

class WMSE(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, weights=None, exclude_zeros=False):
        loss = (inputs - targets) ** 2
        if weights is not None:
            loss *= weights.expand_as(loss)
        if exclude_zeros:
            n_pixels = torch.count_nonzero(targets)
            return torch.sum(loss) / n_pixels
        else:
            return torch.mean(loss)


class WMAE(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, weights=None, exclude_zeros=False):
        loss = F.l1_loss(inputs, targets, reduction='none')
        if weights is not None:
            loss *= weights.expand_as(loss)
        if exclude_zeros:
            n_pixels = torch.count_nonzero(targets)
            return torch.sum(loss) / n_pixels
        else:
            return torch.mean(loss)


class WFocalMSE(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, activate='sigmoid', beta=.2, gamma=1, weights=None, exclude_zeros=False):
        loss = (inputs - targets) ** 2
        loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
            (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
        if weights is not None:
            loss *= weights.expand_as(loss)
        if exclude_zeros:
            n_pixels = torch.count_nonzero(targets)
            return torch.sum(loss) / n_pixels
        else:
            return torch.mean(loss)


class WFocalMAE(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, activate='sigmoid', beta=.2, gamma=1, weights=None, exclude_zeros=False):
        loss = F.l1_loss(inputs, targets, reduction='none')
        loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
            (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
        if weights is not None:
            loss *= weights.expand_as(loss)
        if exclude_zeros:
            n_pixels = torch.count_nonzero(targets)
            return torch.sum(loss) / n_pixels
        else:
            return torch.mean(loss)


class WHuber(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, beta=1., weights=None, exclude_zeros=False):
        l1_loss = torch.abs(inputs - targets)
        cond = l1_loss < beta
        loss = torch.where(cond, 0.5 * l1_loss ** 2 / beta, l1_loss - 0.5 * beta)
        if weights is not None:
            loss *= weights.expand_as(loss)
        if exclude_zeros:
            n_pixels = torch.count_nonzero(targets)
            return torch.sum(loss) / n_pixels
        else:
            return torch.mean(loss)


##############2D Losses
class DiceLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # flatten label and prediction tensors
        inputs = inputs.argmax(dim=1).view(-1).float()
        targets = targets.view(-1).float()

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class DiceCELoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceCELoss, self).__init__()

        self.weight = weight

    def forward(self, inputs, targets, smooth=1):
        # flatten label and prediction tensors
        inputs_fl = inputs.argmax(dim=1).view(-1).float()
        targets_fl = targets.view(-1).float()

        intersection = (inputs_fl * targets_fl).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs_fl.sum() + targets_fl.sum() + smooth)
        CE = F.cross_entropy(inputs, targets, self.weight, reduction='mean')

        return CE + dice_loss


class FocalLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.weight = weight

    def forward(self, inputs, targets, alpha=0.5, gamma=2, smooth=1):
        inputs = inputs.float()

        # first compute cross-entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='mean', weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** gamma * ce_loss).mean()

        return focal_loss


class IoULoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # flatten label and prediction tensors
        inputs = inputs.argmax(dim=1).view(-1).float()
        targets = targets.view(-1).float()

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return 1 - IoU


class SampleFocalLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SampleFocalLoss, self).__init__()
        self.weight = weight

    def forward(self, inputs, targets, alpha=0.5, gamma=2, smooth=1):
        bs = inputs.shape[0]
        # print('I:', inputs.shape, 'T:', targets.shape)
        targets = F.one_hot(targets, num_classes=66).ravel()
        inputs = inputs.float().ravel()
        # print('I:', inputs.shape, 'T:', targets.shape)

        all_0_ind = list((targets == 0).nonzero(as_tuple=True)[0])
        other_ind = list((targets != 0).nonzero(as_tuple=True)[0])
        some_0_ind = random.sample(all_0_ind, int(200 * 200 * 0.05 * int(bs)))
        tot_ind = torch.Tensor((other_ind) + (some_0_ind)).long()

        targets = targets[tot_ind].float()
        inputs = inputs[tot_ind].float()

        # first compute cross-entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='mean', weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** gamma * ce_loss).mean()

        return focal_loss


class SampleIoULoss(torch.nn.Module):

    def __init__(self, size_average=True, sampling=0.5, dmin=30):
        super(SampleIoULoss, self).__init__()

        self.sampling = sampling
        self.dmin = dmin

    def forward(self, inputs, targets, smooth=1):
        bs = inputs.shape[0]
        # flatten label and prediction tensors
        inputs = inputs.argmax(dim=1).view(-1).float()
        targets = targets.view(-1).float()

        all_0_ind = list((targets == self.dmin).nonzero(as_tuple=True)[0])
        other_ind = list((targets != self.dmin).nonzero(as_tuple=True)[0])
        samples = min(len(all_0_ind), int(200 * 200 * self.sampling * int(bs)))
        some_0_ind = random.sample(all_0_ind, samples)
        tot_ind = torch.Tensor((other_ind) + (some_0_ind)).long()

        targets = targets[tot_ind].float()
        inputs = inputs[tot_ind].float()

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return 1 - IoU


class RecallCrossEntropy(torch.nn.Module):

    def __init__(self, n_classes=66, ignore_index=255, weight=None):
        super(RecallCrossEntropy, self).__init__()
        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.weight = weight

    def forward(self, input, target):
        # input (batch,n_classes,H,W)
        # target (batch,H,W)
        pred = input.argmax(1)
        idex = (pred != target).view(-1)

        # calculate ground truth counts
        gt_counter = torch.ones((self.n_classes,)).cuda()
        gt_idx, gt_count = torch.unique(target, return_counts=True)

        # # map ignored label to an exisiting one
        # gt_count[gt_idx==self.ignore_index] = gt_count[1]
        # gt_idx[gt_idx==self.ignore_index] = 1
        # gt_counter[gt_idx] = gt_count.float()

        # calculate false negative counts
        fn_counter = torch.ones((self.n_classes)).cuda()
        fn = target.view(-1)[idex]
        fn_idx, fn_count = torch.unique(fn, return_counts=True)

        # # map ignored label to an exisiting one
        # fn_count[fn_idx==self.ignore_index] = fn_count[1]
        # fn_idx[fn_idx==self.ignore_index] = 1
        # fn_counter[fn_idx] = fn_count.float()

        weights = fn_counter / gt_counter

        CE = F.cross_entropy(input, target, reduction='none', weight=self.weight)  # ,ignore_index=self.ignore_index)
        loss = weights[target] * CE
        return loss.mean()


class CohenKappa(torch.nn.Module):

    def __init__(self, weight=None):
        super(CohenKappa, self).__init__()
        self.weight = weight

    def forward(self, y_preds: torch.Tensor, y_targets: torch.Tensor) -> float:
        y_true = y_targets.flatten().detach().cpu().numpy()
        y_pred = y_preds.argmax(1).detach().flatten().cpu().numpy()
        return torch.tensor(cohen_kappa_score(y_true, y_pred, weights='linear'))


class OrdinalRegression(torch.nn.Module):

    def __init__(self, n_classes=12):  # , n_classes=None):
        super(OrdinalRegression, self).__init__()
        self.n_classes = n_classes

    def forward(self, predictions, targets):
        """Ordinal regression with encoding as in https://arxiv.org/pdf/0704.1028.pdf"""

        predictions = predictions.permute(0, 2, 3, 1).reshape((-1, self.n_classes))
        targets = targets.flatten()

        # Create out modified target with [batch_size, num_labels] shape
        modified_target = torch.zeros_like(predictions)

        # Fill in ordinal target function, i.e. 0 -> [1,0,0,...]
        for i, target in enumerate(targets):
            modified_target[i, 0:target + 1] = 1

        return torch.nn.MSELoss(reduction='mean')(predictions, modified_target)


class BCE(torch.nn.Module):
    def __init__(self, class_weights, class_ignored):
        super(BCE,self).__init__()
        # For avoiding non-deterministic of CrossEntropyLoss according to
        # https://discuss.pytorch.org/t/pytorchs-non-deterministic-cross-entropy-loss-and-the-problem-of-reproducibility/172180/8
        # please set reduction='none'
        self.func = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=class_ignored)
    def forward(self,x,y):

        return self.func(x,y)


def choose_criterion2d(name, class_weights, class_ignored=-99999):
    if name == 'bce':
        return BCE(class_weights, class_ignored)
    elif name == 'dice':
        return DiceLoss(weight=class_weights)
    elif name == 'dicece':
        return DiceCELoss(weight=class_weights)
    elif name == 'jaccard':
        return IoULoss(weight=class_weights)
    elif name == 'focal':
        return FocalLoss(weight=class_weights)


def choose_criterion3d(name, class_weights=None, class_ignored=-99999, **kwargs):
    # regression losses
    if name == 'logcosh':
        return LogCoshLoss()
    elif name == 'xtanh':
        return XTanhLoss()
    elif name == 'xsigmoid':
        return XSigmoidLoss()
    elif name == 'algebraic':
        return AlgebraicLoss()
    elif name == 'mse':
        return torch.nn.MSELoss()
    elif name == 'mae':
        return torch.nn.L1Loss()
    elif name == 'wmse':
        return WMSE()
    elif name == 'wmae':
        return WMAE()
    elif name == 'focalmse':
        return WFocalMSE()
    elif name == 'focalmae':
        return WFocalMAE()
    elif name == 'huber':
        return WHuber()
    elif name == 'ssim':
        return ssim_loss
    elif name == 'mse+ssim':
        return ssim_and_mse_loss
    elif name == 'mae+ssim':
        return ssim_and_mae_loss
    # classification losses
    elif name == 'wce':
        return torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=class_ignored, **kwargs)
    elif name == 'dice':
        return DiceLoss(weight=class_weights, **kwargs)
    elif name == 'dicece':
        return DiceCELoss(weight=class_weights, **kwargs)
    elif name == 'jaccard':
        return IoULoss(weight=class_weights, **kwargs)
    elif name == 'focal':
        return FocalLoss(weight=class_weights, **kwargs)
    elif name == 'sfocal':
        return SampleFocalLoss(weight=class_weights, **kwargs)
    elif name == 'siou':
        return SampleIoULoss(weight=class_weights, **kwargs)
    elif name == 'recall':
        return RecallCrossEntropy(weight=class_weights, **kwargs)
    elif name == 'kappa':
        return CohenKappa(weight=class_weights, **kwargs)
    elif name == 'ord_regr':
        return OrdinalRegression(**kwargs)
