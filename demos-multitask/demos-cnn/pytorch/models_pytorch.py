import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from copy import deepcopy

def move_data_to_gpu(x, cuda):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)

    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)

    else:
        raise Exception("Error!")

    if cuda:
        x = x.cuda()

    x = Variable(x)

    return x

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. 
    Ref: He, Kaiming, et al. "Delving deep into rectifiers: Surpassing 
    human-level performance on imagenet classification." Proceedings of the 
    IEEE international conference on computer vision. 2015.
    """

    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width

    elif layer.weight.ndimension() == 5:
        (n_out, n_in, length, height, width) = layer.weight.size()
        n = n_in * length * height * width

    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()

    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)


def init_hidden(layer):
    # Before we've done anything, we dont have any hidden state.
    # Refer to the Pytorch documentation to see exactly why they have this dimensionality.
    # The axes semantics are (num_layers, minibatch_size, hidden_dim)
    return (torch.zeros(1, 1, layer.hidden_size), torch.zeros(layer.batch_size(), 1, layer.hidden_size))


def init_bn(bn):
    """Initialize a Batchnorm layer. """

    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

######################
class FGSMAttack(object):
    def __init__(self, model=None, epsilon=None, alpha=None):
        """
        One step fast gradient sign method
        """
        self.model = model
        self.epsilon = epsilon
	self.alpha = alpha

    def perturb(self, X_nat, y, epsilons=None, cuda=False):
        """
        Given examples (X_nat, y), returns their adversarial
        counterparts with an attack length of epsilon.
        """

        # Providing epsilons in batch
        if epsilons is not None:
            self.epsilon = epsilons

        X = np.copy(X_nat)

        X_var = move_data_to_gpu(torch.from_numpy(X), cuda)
        y_var = move_data_to_gpu(torch.from_numpy(y), cuda)
        X_var.requires_grad = True

        output = self.model(X_var)
        loss = F.nll_loss(output, y_var)
        loss.backward()
        grad_sign = X_var.grad.data.cpu().sign().numpy()

        X += self.epsilon * grad_sign
        X = np.clip(X, X_nat - self.alpha, X_nat + self.alpha) # not (0, 1)

        return X


#####################################################################################################
class EWC(object):
    def __init__(self, model_attack=None, model_target=None, generator=None, data_type='train', devices=None, classes_num=7, cuda=True):

        self.model_attack = model_attack
        self.model_target = model_target

	self.generate_func = generator.generate_validate(data_type=data_type, 
                                                         devices=devices, 
                                                         shuffle=True, 
                                                         max_iteration=None)

	self.classes_num = classes_num
	self.cuda = cuda

        self.params = {n: p for n, p in self.model_attack.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in deepcopy(self.params).items():
            self._means[n] = p.data

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = p.data

	#####################
    	# Evaluate on mini-batch
        audio_names = []
    	for data in self.generate_func:
            (batch_x, batch_y, batch_audio_names) = data
            audio_names.append(batch_audio_names)
            
            batch_x = move_data_to_gpu(batch_x, self.cuda)

            # Predict
            self.model_attack.eval()
            self.model_target.eval()
            self.model_attack.zero_grad()

	    # advesarial predict
	    batch_x_adv = batch_x + self.model_attack(batch_x)
            loss_a_rec = F.mse_loss(batch_x_adv, batch_x) 	

	    # C&W loss
	    batch_output_adv = self.model_target(batch_x_adv)
            onehot_labels = torch.eye(self.classes_num)
	    onehot_labels = move_data_to_gpu(onehot_labels, self.cuda)
	    onehot_labels = onehot_labels[batch_y]

            prob_real = torch.sum(onehot_labels * batch_output_adv, dim=1)
            prob_other, _ = torch.max((1 - onehot_labels) * batch_output_adv - onehot_labels * 10000, dim=1)
            zeros = torch.zeros_like(prob_other)
            loss_a_cla = torch.max(prob_real - prob_other, zeros)
            loss_a_cla = torch.sum(loss_a_cla)

	    loss_a = 0.02 *loss_a_cla + 0.98*loss_a_rec

            loss_a.backward()	    

            for n, p in self.model_attack.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2

	audio_names = np.concatenate(audio_names, axis=0)
        for n, p in precision_matrices.items():
                precision_matrices[n].data = precision_matrices[n].data / len(audio_names)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss


#####################################################################################################
class EmbeddingLayers(nn.Module):
    def __init__(self):
        super(EmbeddingLayers, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64,
                               kernel_size=(5, 5), stride=(1, 1),
                               padding=(2, 2), bias=False)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=(5, 5), stride=(1, 1),
                               padding=(2, 2), bias=False)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256,
                               kernel_size=(5, 5), stride=(1, 1),
                               padding=(2, 2), bias=False)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512,
                               kernel_size=(5, 5), stride=(1, 1),
                               padding=(2, 2), bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.conv3)
        init_layer(self.conv4)

        init_bn(self.bn1)
        init_bn(self.bn2)
        init_bn(self.bn3)
        init_bn(self.bn4)

    def forward(self, input, return_layers=False):
        (_, seq_len, mel_bins) = input.shape

        x = input.view(-1, 1, seq_len, mel_bins)
        """(samples_num, feature_maps, time_steps, freq_num)"""

        a1 = F.relu(self.bn1(self.conv1(x)))
        a1 = F.max_pool2d(a1, kernel_size=(2, 2))
        a2 = F.relu(self.bn2(self.conv2(a1)))
        a2 = F.max_pool2d(a2, kernel_size=(2, 2))
        a3 = F.relu(self.bn3(self.conv3(a2)))
        a3 = F.max_pool2d(a3, kernel_size=(2, 2))
        emb = F.relu(self.bn4(self.conv4(a3)))
        emb = F.max_pool2d(emb, kernel_size=(2, 2))

        if return_layers is False:
            return emb
        else:
            return [a1, a2, a3, emb]

class DecisionLevelMaxPooling(nn.Module):
    def __init__(self, classes_num):
        super(DecisionLevelMaxPooling, self).__init__()

        self.emb = EmbeddingLayers()
        self.fc_final = nn.Linear(512, classes_num)

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_final)

    def forward(self, input):
        """input: (samples_num, channel, time_steps, freq_bins)
        """

        # (samples_num, channel, time_steps, freq_bins)
        x = self.emb(input)

        # (samples_num, 512, hidden_units)
        x = F.max_pool2d(x, kernel_size=x.shape[2:])
        x = x.view(x.shape[0:2])

        output = F.log_softmax(self.fc_final(x), dim=-1)

        return output


class DecisionLevelAvgPooling(nn.Module):
    def __init__(self, classes_num):
        super(DecisionLevelAvgPooling, self).__init__()

        self.emb = EmbeddingLayers()
        self.fc_final = nn.Linear(256, classes_num)

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_final)

    def forward(self, input):
        """input: (samples_num, channel, time_steps, freq_bins)
        """

        # (samples_num, channel, time_steps, freq_bins)
        x = self.emb(input)

        # (samples_num, 512, hidden_units)
        x = F.avg_pool2d(x, kernel_size=x.shape[2:])
        x = x.view(x.shape[0:2])

        output = F.log_softmax(self.fc_final(x), dim=-1)

        return output

class DecisionLevelFlatten(nn.Module):
    def __init__(self, classes_num):
        super(DecisionLevelFlatten, self).__init__()

        self.emb = EmbeddingLayers()
        self.fc_final = nn.Linear(239616, classes_num)

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_final)

    def forward(self, input):
        """input: (samples_num, channel, time_steps, freq_bins)
        """

        # (samples_num, channel, time_steps, freq_bins)
        x = self.emb(input)

        # (samples_num, 512, hidden_units)
	x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))

        output = F.log_softmax(self.fc_final(x), dim=-1)

        return output

###########################################################################################
class Attention2d(nn.Module):
    def __init__(self, n_in, n_out, att_activation, cla_activation):
        super(Attention2d, self).__init__()

        self.att_activation = att_activation
        self.cla_activation = cla_activation

        self.att = nn.Conv2d(
            in_channels=n_in, out_channels=n_out, kernel_size=(
                1, 1), stride=(
                1, 1), padding=(
                0, 0), bias=True)

        self.cla = nn.Conv2d(
            in_channels=n_in, out_channels=n_out, kernel_size=(
                1, 1), stride=(
                1, 1), padding=(
                0, 0), bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)
        self.att.weight.data.fill_(0.)

    def activate(self, x, activation):

        if activation == 'linear':
            return x

        elif activation == 'relu':
            return F.relu(x)

        elif activation == 'sigmoid':
            return F.sigmoid(x)+0.1

        elif activation == 'log_softmax':
            return F.log_softmax(x, dim=1)

    def forward(self, x):
        """input: (samples_num, channel, time_steps, freq_bins)
        """

        att = self.att(x)
        att = self.activate(att, self.att_activation)

        cla = self.cla(x)
        cla = self.activate(cla, self.cla_activation)

        # (samples_num, channel, time_steps * freq_bins)
        att = att.view(att.size(0), att.size(1), att.size(2) * att.size(3))
        cla = cla.view(cla.size(0), cla.size(1), cla.size(2) * cla.size(3))

        epsilon = 0.1 # 1e-7
        att = torch.clamp(att, epsilon, 1. - epsilon)

        norm_att = att / torch.sum(att, dim=2)[:, :, None]
        x = torch.sum(norm_att * cla, dim=2)

        Return_heatmap = False
        if Return_heatmap:
            return x, norm_att
        else:
            return x

class DecisionLevelSingleAttention(nn.Module):

    def __init__(self, classes_num):

        super(DecisionLevelSingleAttention, self).__init__()

        self.emb = EmbeddingLayers()
        self.attention = Attention2d(
            512,
            classes_num,
            att_activation='sigmoid',
            cla_activation='log_softmax')

    def init_weights(self):
        pass

    def forward(self, input):
        """input: (samples_num, freq_bins, time_steps, 1)
        """

        # (samples_num, hidden_units, time_steps, 1)
        b1 = self.emb(input)

        # (samples_num, classes_num, time_steps, 1)
        output = self.attention(b1)

        return output

###################################################################################################################
class EmbeddingLayers_pooling(nn.Module):
    def __init__(self):
        super(EmbeddingLayers_pooling, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64,
                               kernel_size=(3, 3), stride=(1, 1),  dilation=1,
                               padding=(1, 1), bias=False)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=(3, 3), stride=(1, 1),  dilation=2,
                               padding=(2, 2), bias=False)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256,
                               kernel_size=(3, 3), stride=(1, 1),  dilation=4,
                               padding=(4, 4), bias=False)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512,
                               kernel_size=(3, 3), stride=(1, 1),  dilation=8,
                               padding=(8, 8), bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.conv3)
        init_layer(self.conv4)

        init_bn(self.bn1)
        init_bn(self.bn2)
        init_bn(self.bn3)
        init_bn(self.bn4)

    def forward(self, input, return_layers=False):
        (_, seq_len, mel_bins) = input.shape

        x = input.view(-1, 1, seq_len, mel_bins)
        """(samples_num, feature_maps, time_steps, freq_num)"""

        a1 = F.relu(self.bn1(self.conv1(x)))
        a2 = F.relu(self.bn2(self.conv2(a1)))
        a3 = F.relu(self.bn3(self.conv3(a2)))
        emb = F.relu(self.bn4(self.conv4(a3)))

        if return_layers is False:
            return emb
        else:
            return [a1, a2, a3, emb]


class CnnPooling_Max(nn.Module):
    def __init__(self, classes_num):
        super(CnnPooling_Max, self).__init__()

        self.emb = EmbeddingLayers_pooling()
        self.fc_final = nn.Linear(512, classes_num)

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_final)

    def forward(self, input):
        """(samples_num, feature_maps, time_steps, freq_num)"""
        return_layers = False

        if return_layers is False:
	    a4 = self.emb(input)            
        else:
            [a1, a2, a3, a4] = self.emb(input)

        x = F.max_pool2d(a4, kernel_size=a4.shape[2:])
        x = x.view(x.shape[0:2])

        x = F.log_softmax(self.fc_final(x), dim=-1)

	#################################################### 
        if return_layers is False:
            return x
	else:
            return x, a1, a2, a3, a4


class CnnPooling_Attention(nn.Module):
    def __init__(self, classes_num):
        super(CnnPooling_Attention, self).__init__()

        self.emb = EmbeddingLayers_pooling()
        self.attention = Attention2d(
            512,
            classes_num,
            att_activation='sigmoid',
            cla_activation='log_softmax')

    def init_weights(self):
        pass

    def forward(self, input):
        """(samples_num, feature_maps, time_steps, freq_num)"""
        x = self.emb(input)

        output = self.attention(x)

        return output


#######################################################################################ResNet#################

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, num_classes, block=Bottleneck, layers=[3, 4, 6, 3], zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):  # Res18: BasicBlock [2, 2, 2, 2]
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        (_, seq_len, mel_bins) = x.shape

        x = x.view(-1, 1, seq_len, mel_bins)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
	
	x = F.log_softmax(x, dim=-1)

        return x


###########################################################################VGG16###########################################################
class VggishConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VggishConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input):
        x = input
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=(2, 2))

        return x

class VggishConvBlock3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VggishConvBlock3, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.conv3 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.conv3)
        init_bn(self.bn1)
        init_bn(self.bn2)
        init_bn(self.bn3)

    def forward(self, input):
        x = input
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=(2, 2))

        return x


class Vggish(nn.Module):
    def __init__(self, classes_num):
        super(Vggish, self).__init__()

        self.conv_block1 = VggishConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = VggishConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = VggishConvBlock3(in_channels=128, out_channels=256)
        self.conv_block4 = VggishConvBlock3(in_channels=256, out_channels=512)
        self.conv_block5 = VggishConvBlock3(in_channels=512, out_channels=512)

        self.fc_final = nn.Linear(512, classes_num, bias=True)

        #self.fc_final_binary = nn.Linear(512, 1, bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_final)
        #init_layer(self.fc_final_binary)

    def forward(self, input):
        #(_, seq_len, mel_bins) = input.shape

        x = input.view(-1, 1, input.shape[1], input.shape[2])
        '''(samples_num, feature_maps, time_steps, freq_num)'''
	
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)

        x_out = F.max_pool2d(x, kernel_size=x.shape[2:])
        x_out = x_out.view(x_out.shape[0:2])

        output = F.log_softmax(self.fc_final(x_out), dim=-1)
	
	#output_binary = torch.sigmoid(self.fc_final_binary(x))

	featureExtract = False
	if featureExtract == False:
            return output
	else:
	    return x
        #return output, output_binary


##############################################################################################################
class AlexNet(nn.Module):
    def __init__(self, classes_num):
        super(AlexNet, self).__init__()
	
	self.conv1 = nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=2)
	self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2)
	self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
	self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
	self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(96)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(384)
        self.bn4 = nn.BatchNorm2d(384)
        self.bn5 = nn.BatchNorm2d(256)

        self.fc_final = nn.Linear(256, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.conv3)
        init_layer(self.conv4)
        init_layer(self.conv5)
        init_bn(self.bn1)
        init_bn(self.bn2)
        init_bn(self.bn3)
        init_bn(self.bn4)
        init_bn(self.bn5)
        init_layer(self.fc_final)

    def forward(self, input):
        (_, seq_len, mel_bins) = input.shape

        x = input.view(-1, 1, seq_len, mel_bins)
        '''(samples_num, feature_maps, time_steps, freq_num)'''

        x = F.relu(self.bn1(self.conv1(x)))
	x = F.max_pool2d(x, kernel_size=(3, 3), stride=2)
        x = F.relu(self.bn2(self.conv2(x)))
	x = F.max_pool2d(x, kernel_size=(3, 3), stride=2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
	x = F.max_pool2d(x, kernel_size=(3, 3), stride=2)

        x = F.max_pool2d(x, kernel_size=x.shape[2:])
        x = x.view(x.shape[0:2])

        x = F.log_softmax(self.fc_final(x), dim=-1)

        return x
#################################################################################################
class CnnAtrous(nn.Module):
    def __init__(self):
        super(CnnAtrous, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64,
                               kernel_size=(5, 5), stride=(1, 1),  dilation=1,
                               padding=(2, 2), bias=False)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=(5, 5), stride=(1, 1),  dilation=2,
                               padding=(4, 4), bias=False)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=64,
                               kernel_size=(5, 5), stride=(1, 1),  dilation=4,
                               padding=(8, 8), bias=False)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=1,
                               kernel_size=(5, 5), stride=(1, 1),  dilation=8,
                               padding=(16, 16), bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(1)

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.conv3)
        init_layer(self.conv4)

        init_bn(self.bn1)
        init_bn(self.bn2)
        init_bn(self.bn3)
        init_bn(self.bn4)

    def forward(self, input):
        #(_, seq_len, mel_bins) = input.shape

        x = input.view(-1, 1, input.shape[1], input.shape[2])
        """(samples_num, feature_maps, time_steps, freq_num)"""

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        emb = F.sigmoid(self.bn4(self.conv4(x)))
	
	emb = torch.squeeze(emb, 1)

	#alpha = 0.1
	#emb = torch.max(torch.min(emb, input + alpha), input - alpha)

	#alpha = 0.4 * torch.ones(input.shape[0], input.shape[1], input.shape[2], dtype=torch.float32)
	#alpha = move_data_to_gpu(alpha, True)
	#emb = torch.max(torch.min(emb, alpha), -alpha)	

        return emb



