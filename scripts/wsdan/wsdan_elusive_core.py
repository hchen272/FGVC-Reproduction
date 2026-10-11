"""WS-DAN numerical core extracted from fixed Elusive Images source.

BAP, ConvBNReLU, setup_model, forward, crop/drop, centerloss
are verbatim method bodies from commit 37639fd61f3d8b0b1a23a5d9922a48cd2b0f8fb0.
Only Lightning/torchmetrics glue is replaced, plus FP32 log-space probability
fusion replacing the author test softmax/log expression to prevent underflow.
Raw/crop forwards and crop behavior stay unchanged; no framework dependency.
Import this module only after the trainer enters its CUDA runtime.
"""
from functools import partial
import math
import random
import torch
EPS = 1e-6
interpolate = partial(torch.nn.functional.interpolate, mode='bilinear', align_corners=False)
normalize = torch.nn.functional.normalize

def stable_probability_fusion(logits_raw, logits_crop):
    # Mathematically the author's log(mean(softmax(raw),softmax(crop))).
    # Log-space FP32 keeps finite low probabilities when softmax would underflow.
    raw_log_probs = logits_raw.float().log_softmax(dim=1)
    crop_log_probs = logits_crop.float().log_softmax(dim=1)
    return torch.logaddexp(raw_log_probs, crop_log_probs) - math.log(2.0)


class BAP(torch.nn.Module):
    '''Bilinear Attention Pooling'''
    def __init__(self, pool='GAP'):
        super(BAP, self).__init__()
        assert pool in ['GAP', 'GMP']
        if pool == 'GAP':
            self.pool = None
        else:
            self.pool = torch.nn.AdaptiveMaxPool2d(1)

    def forward(self, features, attentions):
        B, C, H, W = features.size()
        _, M, AH, AW = attentions.size()

        # match size
        if AH != H or AW != W:
            attentions = interpolate(attentions, (H, W))

        # feature_matrix: (B, M, C) -> (B, M * C)
        if self.pool is None:
            feature_matrix = torch.einsum('imjk,injk->imn', (attentions, features))
            feature_matrix = feature_matrix.div(float(H * W)).view(B, -1)
        else:
            # This could be done without the loop
            feature_matrix = []
            for i in range(M):
                AiF = self.pool(features * attentions[:, i:i + 1, ...]).view(B, -1)
                feature_matrix.append(AiF)
            feature_matrix = torch.cat(feature_matrix, dim=1)

        # sign-sqrt
        s = feature_matrix.sign()
        feature_matrix =  s * feature_matrix.abs().add(EPS).sqrt()

        # l2 normalization along dimension M and C
        feature_matrix = normalize(feature_matrix, dim=-1)
        return feature_matrix

class ConvBNReLU(torch.nn.Module):
    '''Conv -> BatchNorm -> ReLU'''
    def __init__(self, inc, outc, ks=1, **kwargs):
        super().__init__()
        self.conv = torch.nn.Conv2d(inc, outc, ks, bias=False, **kwargs)
        self.bn = torch.nn.BatchNorm2d(outc, eps=0.001)

    def forward(self, x):
        x = self.bn(self.conv(x)) 
        return torch.nn.functional.relu(x, inplace=True)

class WSDANCore(torch.nn.Module):
    def __init__(self, backbone, num_classes=200, num_attn=32, beta=5e-2):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.num_attn = num_attn
        self.beta = beta
        self.objective = torch.nn.CrossEntropyLoss()
        # Preserve the author's training-mode 33x33 shape probe, including BN updates.
        self.setup_model()

    def setup_model(self):
        with torch.no_grad():
            out = self.backbone(torch.ones(1, 3, 33, 33))[-1]
            channels = out.shape[1]

        self.attentions = ConvBNReLU(channels, self.num_attn)
        self.bap = BAP(pool='GAP')
        self.fc = torch.nn.Linear(self.num_attn * channels, self.num_classes, bias=False)

        center = torch.zeros(self.num_classes, self.num_attn * channels)
        self.register_buffer('feature_center', center)

    def forward(self, x, p_only=False):
        features = self.backbone(x)[-1]
        attentions = self.attentions(features)
        feature_matrix = self.bap(features, attentions)

        # This mystical multiply by 100 is in the original tensorflow implementation, as well as the pytorch
        # reimplementations. No explenation is provided. I tried removing it, but training doesn't work without it...
        p = self.fc(feature_matrix * 100.0)
        # p = self.fc(feature_matrix)

        if p_only: return p

        if self.training:
            weights = attentions.detach().sum((2, 3)).add_(EPS).sqrt_()
            weights = normalize(weights, p=1, dim=1, out=weights)
            k_idx = torch.multinomial(weights, 2, replacement=True)
            b_idx = torch.arange(k_idx.shape[0], device=k_idx.device)
            attention_map = attentions[b_idx[:,None], k_idx].contiguous()
        else:
            attention_map = attentions.mean(1, keepdim=True)

        return p, feature_matrix, attention_map

    @staticmethod
    def batch_crop(images, attention_map, theta=(0.4,0.6), pad_ratio=0.1):
        bs, _, h, w = images.size()
        crop_images = []
        ph = pad_ratio * h
        pw = pad_ratio * w
        for i in range(bs):
            atten_map = attention_map[i:i+1]
            if isinstance(theta, tuple):
                theta = random.uniform(*theta)
            theta_c = theta * atten_map.max()

            crop_mask = interpolate(atten_map, (h, w)).ge(theta_c)
            nz_inds = crop_mask[0, 0].nonzero(as_tuple=False)
            hmin = max(int(nz_inds[:, 0].min().item() - ph), 0)
            hmax = min(int(nz_inds[:, 0].max().item() + ph), h)
            wmin = max(int(nz_inds[:, 1].min().item() - pw), 0)
            wmax = min(int(nz_inds[:, 1].max().item() + pw), w)

            crop_images.append(
                interpolate(images[i:i+1, :, hmin:hmax, wmin:wmax], (h, w)))
        crop_images = torch.cat(crop_images, dim=0)
        return crop_images

    @staticmethod
    def batch_drop(images, attention_map, theta=(0.2, 0.5)):
        bs, _, h, w = images.size()
        drop_masks = []
        for i in range(bs):
            atten_map = attention_map[i:i+1]
            if isinstance(theta, tuple):
                theta = random.uniform(*theta)
            theta_d = theta * atten_map.max()

            drop_masks.append(interpolate(atten_map, (h, w)).lt(theta_d))
        drop_masks = torch.cat(drop_masks, dim=0)
        drop_images = images * drop_masks.float()
        return drop_images

    @staticmethod
    def centerloss(features, centers):
        # loss = features.sub(centers).square().sum(-1).mean()
        loss = torch.nn.functional.mse_loss(features, centers, reduction='sum') / centers.shape[0]
        return loss

    def inference_step(self, x):
        # full-image predictions
        logits_raw, _, attention_map = self(x)
        # attention crop predictions
        crops = self.batch_crop(x, attention_map[:,:1])
        logits_crop = self(crops, p_only=True)
        # combined predictions
        logits = stable_probability_fusion(logits_raw, logits_crop)
        return logits

    def training_objective(self, x, y):

        # full-image predictions
        logits_raw, feature_mat, attention_map = self(x)

        # update class feature centers
        feature_center_batch = normalize(self.feature_center[y], dim=-1)
        center_update = self.beta * (feature_mat.detach() - feature_center_batch)
        idx = y[:, None].expand_as(center_update)
        self.feature_center.scatter_add_(dim=0, index=idx, src=center_update)

        # attention crop predictions: first channel of attention_map is for crop
        with torch.no_grad():
            crops = self.batch_crop(x, attention_map[:,:1])
        logits_crop = self(crops, p_only=True)

        # attention drop predictions: second channel of attention_map is for drop
        with torch.no_grad():
            drops = self.batch_drop(x, attention_map[:,1:])
        logits_drop = self(drops, p_only=True)

        # calculate loss
        obj = self.objective
        ce_raw, ce_crop, ce_drop = obj(logits_raw, y), obj(logits_crop, y), obj(logits_drop, y)
        loss = (1. / 3) * (ce_raw + ce_crop + ce_drop)
        loss_center = self.centerloss(feature_mat, feature_center_batch)
        loss = loss + loss_center
        details = {"ce_raw": ce_raw.detach(), "ce_crop": ce_crop.detach(),
                   "ce_drop": ce_drop.detach(), "center_loss": loss_center.detach(),
                   "raw_correct": logits_raw.argmax(1).eq(y).sum().detach(),
                   "crop_correct": logits_crop.argmax(1).eq(y).sum().detach(),
                   "drop_correct": logits_drop.argmax(1).eq(y).sum().detach()}
        return logits_raw, loss, details

