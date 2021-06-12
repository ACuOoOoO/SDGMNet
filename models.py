from pickle import FALSE
import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
from Utils import L1Norm, L2Norm
import numpy as np
from torch.cuda.amp import autocast,GradScaler
from torch.nn import Parameter

eps_fea_norm = 1e-5
eps_l2_norm = 1e-10

class SOSNet32x32(nn.Module):
    """
    128-dimensional SOSNet model definition trained on 32x32 patches
    
    """
    def __init__(self, dim_desc=128, drop_rate=0.1):
        super(SOSNet32x32, self).__init__()
        self.dim_desc = dim_desc
        self.drop_rate = drop_rate

        norm_layer = nn.BatchNorm2d
        activation = nn.ReLU()

        self.layers = nn.Sequential(
            nn.InstanceNorm2d(1, affine=False, eps=eps_fea_norm),
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            norm_layer(32, affine=False, eps=eps_fea_norm),
            activation,
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            norm_layer(32, affine=False, eps=eps_fea_norm),
            activation,

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(64, affine=False, eps=eps_fea_norm),
            activation,
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            norm_layer(64, affine=False, eps=eps_fea_norm),
            activation,

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(128, affine=False, eps=eps_fea_norm),
            activation,
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            norm_layer(128, affine=False, eps=eps_fea_norm),
            activation,

            nn.Dropout(self.drop_rate),
            nn.Conv2d(128, self.dim_desc, kernel_size=8, bias=False),
            norm_layer(128, affine=False, eps=eps_fea_norm),
        )

        self.desc_norm = nn.Sequential(
            nn.LocalResponseNorm(2 * self.dim_desc, alpha=2 * self.dim_desc, beta=0.5, k=0)
        )

        return

    def forward(self, patch):
        descr = self.desc_norm(self.layers(patch) + eps_l2_norm)
        descr = descr.view(descr.size(0), -1)
        return descr


class FilterResponseNormLayer(nn.Module):
    def __init__(self, num_features, eps=1e-6):
        super(FilterResponseNormLayer, self).__init__()
        self.num_features = num_features
        self.tau = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.gamma = Parameter(torch.Tensor(1, num_features, 1, 1))
        #self.eps = Parameter(torch.Tensor([eps]))
        self.eps = 1e-6
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.tau,-1)
        nn.init.constant_(self.beta,0)
        nn.init.constant_(self.gamma,0.1)

    def forward(self, input):
        with autocast():
            nu2 = torch.mean(input**2, dim=(2, 3), keepdim=True, out=None)
            t = torch.rsqrt(nu2 + self.eps)
            tt = input * t
        return torch.max(self.gamma * tt + self.beta, self.tau)

    def extra_repr(self):
        return '{}'.format(
            self.num_features
        )
        

class HardNet(nn.Module):
    """
        HardNet model definition
        input size 1*32*32
        output size 128*1*1
        
    """
    def __init__(self,dropout_rate=0.3):
        super(HardNet, self).__init__()
        self.features = nn.Sequential(
            nn.InstanceNorm2d(1, affine=False),
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias = False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias = False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias = False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(128, 128, kernel_size=8, bias = False),
            nn.BatchNorm2d(128, affine=False),
        )
        self.features.apply(weights_init)
        return
        
    def forward(self, input,BN_out = False):
        x_features = self.features(input)
        x = x_features.view(x_features.size(0), -1)
        if BN_out:
            return F.normalize(x), x
        x = F.normalize(x)
        return x

eps_l2_norm = 1e-10

class FRN(nn.Module):
    def __init__(self, num_features, eps=1e-6, is_bias=True, is_scale=True, is_eps_leanable=False):
        """
        weight = gamma, bias = beta
        beta, gamma:
            Variables of shape [1, 1, 1, C]. if TensorFlow
            Variables of shape [1, C, 1, 1]. if PyTorch
        eps: A scalar constant or learnable variable.
        """
        super(FRN, self).__init__()

        self.num_features = num_features
        self.init_eps = eps
        self.is_eps_leanable = is_eps_leanable
        self.is_bias = is_bias
        self.is_scale = is_scale


        self.weight = nn.parameter.Parameter(torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.bias = nn.parameter.Parameter(torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        if is_eps_leanable:
            self.eps = nn.parameter.Parameter(torch.Tensor(1), requires_grad=True)
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        if self.is_eps_leanable:
            nn.init.constant_(self.eps, self.init_eps)

    def extra_repr(self):
        return 'num_features={num_features}, eps={init_eps}'.format(**self.__dict__)

    def forward(self, x):
        """
        0, 1, 2, 3 -> (B, H, W, C) in TensorFlow
        0, 1, 2, 3 -> (B, C, H, W) in PyTorch
        TensorFlow code
            nu2 = tf.reduce_mean(tf.square(x), axis=[1, 2], keepdims=True)
            x = x * tf.rsqrt(nu2 + tf.abs(eps))
            # This Code include TLU function max(y, tau)
            return tf.maximum(gamma * x + beta, tau)
        """
        # Compute the mean norm of activations per channel.
        nu2 = x.pow(2).mean(dim=[2, 3], keepdim=True)

        # Perform FRN.
        x = x * torch.rsqrt(nu2 + self.eps.abs())

        # Scale and Bias
        if self.is_scale:
            x = self.weight * x
        if self.is_bias:
            x = x + self.bias
        return x

class TLU(nn.Module):
    def __init__(self, num_features):
        """max(y, tau) = max(y - tau, 0) + tau = ReLU(y - tau) + tau"""
        super(TLU, self).__init__()
        self.num_features = num_features
        self.tau = nn.parameter.Parameter(torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.zeros_(self.tau)
        nn.init.constant_(self.tau, -1)

    def extra_repr(self):
        return 'num_features={num_features}'.format(**self.__dict__)

    def forward(self, x):
        return torch.max(x, self.tau)

class SDGMNet(nn.Module):
    """
    SDGMNet model adopted from Hynet 'HyNet: Learning Local Descriptor with Hybrid Similarity Measure and Triplet Loss'(
    https://github.com/yuruntian/HyNet#hynet-learning-local-descriptor-with-hybrid-similarity-measure-and-triplet-loss).
    """
    def __init__(self, is_bias=True, is_bias_FRN=True, dim_desc=128, drop_rate=0.3):
        super(HyNet, self).__init__()
        self.dim_desc = dim_desc
        self.drop_rate = drop_rate

        self.layer1 = nn.Sequential(
            FRN(1, is_bias=is_bias_FRN),
            TLU(1),
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=is_bias),
            FRN(32, is_bias=is_bias_FRN),
            TLU(32),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=is_bias),
            FRN(32, is_bias=is_bias_FRN),
            TLU(32),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=is_bias),
            FRN(64, is_bias=is_bias_FRN),
            TLU(64),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=is_bias),
            FRN(64, is_bias=is_bias_FRN),
            TLU(64),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=is_bias),
            FRN(128, is_bias=is_bias_FRN),
            TLU(128),
        )

        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=is_bias),
            FRN(128, is_bias=is_bias_FRN),
            TLU(128),
        )

        self.layer7 = nn.Sequential(
            nn.Dropout(self.drop_rate),
            nn.Conv2d(128, self.dim_desc, kernel_size=8, bias=False),
            nn.BatchNorm2d(self.dim_desc, affine=False)
        )
        
        self.features=nn.Sequential(self.layer1, self.layer2, self.layer3, self.layer4, self.layer5, self.layer6, self.layer7)
            
    def forward(self, x):
        feat = self.features(x)
        feat_t = feat.view(-1, self.dim_desc)
        feat_norm = F.normalize(feat_t, dim=1)
        return feat_norm

class OurNet(nn.Module):
    """
    HyNet model definition.
    The FRN and TLU layer are from the papaer
    `Filter Response Normalization Layer: Eliminating Batch Dependence in the Training of Deep Neural Networks`
    https://github.com/yukkyo/PyTorch-FilterResponseNormalizationLayer
    """
    def __init__(self, is_bias=True, is_bias_FRN=True, dim_desc=128, drop_rate=0.3):
        super(OurNet, self).__init__()
        self.dim_desc = dim_desc
        self.drop_rate = drop_rate
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=is_bias),
            FRN(32, is_bias=is_bias_FRN),
            TLU(32),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=is_bias),
            FRN(32, is_bias=is_bias_FRN),
            TLU(32),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=is_bias),
            FRN(64, is_bias=is_bias_FRN),
            TLU(64),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=is_bias),
            FRN(64, is_bias=is_bias_FRN),
            TLU(64),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=is_bias),
            FRN(128, is_bias=is_bias_FRN),
            TLU(128),
        )

        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=is_bias),
            FRN(128, is_bias=is_bias_FRN),
            TLU(128),
        )

        self.layer7 = nn.Sequential(
            nn.Dropout(self.drop_rate),
            nn.Conv2d(128, self.dim_desc, kernel_size=8, bias=False),
            nn.BatchNorm2d(self.dim_desc, affine=False)
        )
        
        self.features=nn.Sequential(self.layer1, self.layer2, self.layer3, self.layer4, self.layer5, self.layer6, self.layer7)
            
    def forward(self, x):
        #x = self.ImageNorm(x)
        x_feat = self.features(x)
        x_t = x_feat.view(-1, self.dim_desc)
        x_norm = F.normalize(x_t, dim=1)
        return x_norm,x_t
    
    def ImageNorm(self,x):
        x_mean = torch.mean(x)
        x_std = torch.sqrt(torch.var(x)+eps_fea_norm)
        return (x-x_mean)/x_std

class HyNet_plus(nn.Module):
    """
    HyNet model definition.
    The FRN and TLU layer are from the papaer
    `Filter Response Normalization Layer: Eliminating Batch Dependence in the Training of Deep Neural Networks`
    https://github.com/yukkyo/PyTorch-FilterResponseNormalizationLayer
    """
    def __init__(self, is_bias=True, is_bias_FRN=True, dim_desc=128, drop_rate=0.3):
        super(HyNet_plus, self).__init__()
        self.dim_desc = dim_desc
        self.drop_rate = drop_rate
        self.gradKernel = nn.Conv2d(1,1,kernel_size=3,padding=1)
        weights = torch.tensor([[-1,-1,-1],
                          [-1,8,-1],
                          [-1,-1,-1]])
        weights = weights*1.0/16
        weights = weights.view(1,1,3,3)
        self.gradKernel.weight == nn.parameter.Parameter(weights)
        self.gradKernel.requires_grad=False


        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=is_bias),
            FRN(64, is_bias=is_bias_FRN),
            TLU(64),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=is_bias),
            FRN(128, is_bias=is_bias_FRN),
            TLU(128),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=is_bias),
            FRN(128, is_bias=is_bias_FRN),
            TLU(128),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=is_bias),
            FRN(256, is_bias=is_bias_FRN),
            TLU(256),
        )

        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=is_bias),
            FRN(256, is_bias=is_bias_FRN),
            TLU(256),
        )

        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=8, bias=False),
            nn.ReLU(),
            nn.Dropout(self.drop_rate),
        )

        self.fc = nn.Sequential(
            nn.Conv2d(512,self.dim_desc,kernel_size=1,bias=False),
            nn.BatchNorm2d(128, affine=False))

        self.layer1 = nn.Sequential(
                    FRN(1, is_bias=is_bias_FRN),
                    TLU(1),
                    nn.Conv2d(1, 128, kernel_size=3, padding=1, bias=is_bias),
                    FRN(32, is_bias=is_bias_FRN),
                    TLU(32),
                    nn.Conv2d(128, 32, kernel_size=3, padding=1, bias=is_bias),
                    FRN(32, is_bias=is_bias_FRN),
                    TLU(32),
        )
        self.branch2 = nn.Conv2d(64,64,kernel_size=1,bias=False)
        self.branch4 = nn.Conv2d(128,128,kernel_size=1,bias=False)
        self.branch6 = nn.Conv2d(256,256,kernel_size=1,bias=FALSE)
    def forward(self, x):
        x_grad = self.gradKernel(x)
        x_grad = (x_grad-torch.mean(x_grad))/torch.std(x_grad)
        x_grad = torch.clamp(x_grad,min=1.0)
        x = self.img_PreProc(x)
        x_grad = self.grad_PreProc(x_grad)
        x_new = torch.cat([x,x_grad],dim=1)
        feat = self.layer2(x_new)+self.branch2(x_new)
        feat = self.layer3(feat)
        feat = self.layer4(feat)+self.branch4(feat)
        feat = self.layer5(feat)
        feat = self.layer6(feat)+self.branch6(feat)
        feat = self.layer7(feat)
        feat = self.fc(feat)
        feat = feat.view(-1, self.dim_desc)
        feat_normed = F.normalize(feat, dim=1)
        return feat_normed

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        try:
            nn.init.orthogonal_(m.weight.data,gain=0.1)
        except:
            pass 
    return
    


class TNet(nn.Module):
    """TFeat model definition
    """
    def __init__(self):
        super(TNet, self).__init__()
        self.features = nn.Sequential(
            nn.InstanceNorm2d(1, affine=False),
            nn.Conv2d(1, 32, kernel_size=7),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=6),
            nn.Tanh()
        )
        self.descr = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.descr(x)
        return x
