#!/usr/bin/env python3
"""Offline IELT adapter executing the frozen Elusive author operations.

The original model/selection/initialization methods are AST-extracted unchanged.
The shim replaces Lightning setup with the already used offline augreg loader.
Only epoch/eval handling, selection-buffer rebinding and audit interfaces are new.
No network download or randomly initialized fallback is permitted.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import numpy as np
import torch
from torch.nn.functional import normalize
import timm

AUTHOR_REPO = 'https://github.com/catalys1/elusive-images-fgvc'
AUTHOR_COMMIT = '37639fd61f3d8b0b1a23a5d9922a48cd2b0f8fb0'
VIT_NPZ_SHA256 = '69c9a2822a47838f57e522350b10ccc0d89d67fdcee07603e4ae3259b3350336'
VIT_NPZ_BYTES = 346335542
SOURCE_SHA256 = {
    'src/models/ielt.py': '6dbd5eb8bd32cb7cfd5dfea2692e24ff95395145cb9083e770368d51b529d482',
    'src/models/base.py': '0210a999c4259d7644be606a6b5f5dd7f9d9b8ed5596db5c91189ff4e2bf92a3',
}
VENDOR_ROOT = Path(__file__).resolve().parents[2] / 'third_party' / 'elusive_images'


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_source(relative):
    path = VENDOR_ROOT / relative
    if sha256_file(path) != SOURCE_SHA256[relative]:
        raise ValueError(f'Frozen author source changed: {relative}')
    return path, path.read_text(encoding='utf8')


def _compile_original_nodes(relative, names, namespace):
    path, source = _verified_source(relative)
    tree = ast.parse(source, filename=str(path))
    nodes = [node for node in tree.body
             if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise ValueError(f'Missing frozen author definitions: {relative}')
    # The selected definitions are not modified, including source locations.
    selected = ast.Module(body=nodes, type_ignores=[])
    exec(compile(selected, str(path), 'exec'), namespace)


_AUTHOR_NAMESPACE = {
    '__name__': 'frozen_elusive_ielt_operations', 'torch': torch,
    'normalize': normalize, 'deepcopy': deepcopy, 'math': math,
    'Optional': Optional, 'List': List,
}
_compile_original_nodes('src/models/base.py',
                        {'get_pretrained_submodules', 'make_parameter_groups'}, _AUTHOR_NAMESPACE)
author_get_pretrained_submodules = _AUTHOR_NAMESPACE['get_pretrained_submodules']
author_make_parameter_groups = _AUTHOR_NAMESPACE['make_parameter_groups']


class _OfflineImageClassifier(torch.nn.Module):
    """Only the parent setup of the original IELT class, without Lightning."""
    def __init__(self, base_conf=None, model_conf=None):
        super().__init__()
        self.base_conf = SimpleNamespace(**(base_conf or {}))
        self.model_conf = SimpleNamespace(**model_conf)
        self.num_classes = self.model_conf.num_classes
        self.inject_backbone_args()
        self.backbone = self.model_conf.preloaded_backbone
        # Capture before IELT replaces/moves any submodules, as ImageClassifier
        # does. The backbone root cls_token/pos_embed are intentionally absent.
        self.finetune_list = author_get_pretrained_submodules(self.backbone, prefix='backbone.')
        self.setup_model()
        self.objective = torch.nn.CrossEntropyLoss()


_AUTHOR_NAMESPACE['ImageClassifier'] = _OfflineImageClassifier
_compile_original_nodes('src/models/ielt.py',
                        {'ExtractableAttentionWrapper', 'SelectionBlock', 'MultiHeadVoting',
                         'CrossLayerRefinement', 'IELT'}, _AUTHOR_NAMESPACE)
AuthorIELT = _AUTHOR_NAMESPACE['IELT']
MultiHeadVoting = _AUTHOR_NAMESPACE['MultiHeadVoting']
SelectionBlock = _AUTHOR_NAMESPACE['SelectionBlock']


def load_offline_backbone(pretrained_npz, expected_sha256=VIT_NPZ_SHA256):
    """Load the exact existing augreg checkpoint before IELT modifies blocks."""
    path = Path(pretrained_npz)
    if not path.is_file() or path.suffix != '.npz':
        raise FileNotFoundError('IELT requires the existing local Google augreg .npz checkpoint')
    if expected_sha256 != VIT_NPZ_SHA256 or path.stat().st_size != VIT_NPZ_BYTES or sha256_file(path) != VIT_NPZ_SHA256:
        raise ValueError('IELT initialization differs from the completed Elusive ViT NPZ identity')
    if str(timm.__version__) != '1.0.15':
        raise RuntimeError('IELT parity requires the existing timm 1.0.15 runtime; do not silently change its loader')
    with np.load(path, allow_pickle=False) as original:
        required_shapes = {
            'embedding/kernel': (16, 16, 3, 768), 'cls': (1, 1, 768),
            'Transformer/posembed_input/pos_embedding': (1, 197, 768),
            'Transformer/encoder_norm/scale': (768,), 'head/bias': (1000,),
        }
        for name, shape in required_shapes.items():
            if name not in original or tuple(original[name].shape) != shape:
                raise ValueError(f'Unexpected augreg checkpoint tensor: {name}')
    backbone = timm.create_model('vit_base_patch16_224.augreg_in1k', pretrained=False,
                                 num_classes=200, img_size=448, global_pool='', proj_drop_rate=0.1)
    if len(backbone.blocks) != 12 or not isinstance(backbone.norm, torch.nn.LayerNorm):
        raise ValueError('Unexpected IELT ViT-B/16 normalization/block architecture')
    # Unlike avg-pooling plain ViT, this normal .norm can receive encoder_norm
    # directly. No norm-to-fc_norm alias is used for this model.
    backbone.load_pretrained(str(path))
    with np.load(path, allow_pickle=False) as original:
        expected_patch = torch.from_numpy(original['embedding/kernel'].transpose(3, 2, 0, 1)).float()
        expected_cls = torch.from_numpy(original['cls']).float()
        expected_norm = torch.from_numpy(original['Transformer/encoder_norm/scale']).float()
        for actual, expected, name in [
            (backbone.patch_embed.proj.weight, expected_patch, 'patch embedding'),
            (backbone.cls_token, expected_cls, 'class token'),
            (backbone.norm.weight, expected_norm, 'encoder norm'),
        ]:
            if not torch.equal(actual.detach().cpu(), expected):
                raise ValueError(f'Offline pretrained {name} was not loaded exactly')
        for i in (0, 5, 10, 11):
            prefix = f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/'
            qkv = np.concatenate([original[prefix + k + '/kernel'].reshape(768, 768).T
                                  for k in ('query', 'key', 'value')], axis=0)
            if not torch.equal(backbone.blocks[i].attn.qkv.weight.detach().cpu(), torch.from_numpy(qkv).float()):
                raise ValueError(f'Offline pretrained block {i} QKV was not loaded exactly')
    if tuple(backbone.pos_embed.shape) != (1, 785, 768):
        raise ValueError('448 position embedding must contain 28x28 patches and one class token')
    return backbone, {
        'pretraining': 'same ImageNet-1K Google augreg as completed ordinary Elusive ViT',
        'resolved_model': 'vit_base_patch16_224.augreg_in1k',
        'path': str(path), 'sha256': VIT_NPZ_SHA256, 'file_bytes': VIT_NPZ_BYTES,
        'offline_loader': 'timm 1.0.15 NPZ load_pretrained on full unmodified 12-block ViT, then author IELT setup',
        'position_embedding_resize': 'same timm NPZ bilinear interpolation, antialias=False, 14x14 to 28x28',
        'global_pool': '', 'image_size': 448, 'backbone_proj_drop_rate': 0.1,
        'backbone_drop_rate': 0.0, 'method_head_dropout': 0.7,
        'skipped_pretrained_classifier': ['head.weight', 'head.bias'],
        'loaded_tensor_checks': ['patch embedding', 'class token', 'encoder norm', 'blocks0,5,10,11 QKV'],
    }


class IELTAdapter(AuthorIELT):
    """Original IELT with explicit offline initialization and safe train/eval hooks."""
    def __init__(self, pretrained_npz, expected_sha256=VIT_NPZ_SHA256,
                 image_size=448, num_classes=200):
        if image_size != 448 or num_classes != 200:
            raise ValueError('This frozen baseline fixes full CUB200 and 448 inputs')
        backbone, provenance = load_offline_backbone(pretrained_npz, expected_sha256)
        super().__init__(reinit_final_blocks=True, votes_per_head=24,
                         total_selections=126, warmup_epochs=10, alpha=0.4, drop_rate=0.7,
                         base_conf={'base_lr': .01, 'finetune_lr_scale': .1, 'weight_decay': 1e-5},
                         model_conf={'model_name': 'vit_base_patch16_224.augreg_in1k',
                                     'num_classes': 200, 'pretrained': True,
                                     'model_kw': {'img_size': 448, 'proj_drop_rate': .1},
                                     'preloaded_backbone': backbone})
        self._epoch_zero_based = -1
        self._provenance = {**provenance, 'author_repo': AUTHOR_REPO, 'author_commit': AUTHOR_COMMIT,
                            'frozen_source_sha256': dict(SOURCE_SHA256),
                            'architecture': '11 selection blocks + CLR block + key block; shared dropout/linear classification head',
                            'reinit_final_blocks': True,
                            'random_initialization': 'author _init_block: Linear and LayerNorm weights N(0,.02), bias zero',
                            'selection_votes_per_head': 24, 'initial_total_selections': 126,
                            'selection_warmup_zero_based_epoch': 10, 'selection_ema_alpha': .001,
                            'loss': '.6 CE(part_logits) + .4 CE(complement_logits)',
                            'prediction': 'part_logits, including original assist-logit operation',
                            'preserved_author_selection_quirks': ['1-based sort_idx passed unchanged to bucketize(right=True)',
                                                                   'per-layer round quotas not forced to sum to126',
                                                                   'original torch.topk tie behavior'],
                            'adapter_only_changes': ['offline-loader and Lightning-free setup shim',
                                                     'explicit epoch hook and no eval quota mutation',
                                                     'quota-attribute rebinding after device/checkpoint changes',
                                                     'FP32 quota buffer retained during explicit dtype moves',
                                                     'standalone loss/grouping/provenance interfaces']}
        self._sync_selection_counts()

    @property
    def provenance(self):
        return deepcopy(self._provenance)

    def _sync_selection_counts(self):
        if self.num_selects.shape != (11,) or self.select_rate.shape != (11,):
            raise ValueError('Invalid IELT selection quota buffer shape')
        if self.num_selects.dtype != torch.int64 or self.select_rate.dtype != torch.float32:
            raise ValueError('IELT selection quotas must be int64 and rates float32')
        if not torch.isfinite(self.select_rate).all() or (self.num_selects < 0).any():
            raise ValueError('Invalid IELT selection quota values')
        for block, count in zip(self.backbone.blocks, self.num_selects):
            block.num_selections = count

    def set_epoch(self, epoch_zero_based):
        epoch = int(epoch_zero_based)
        if epoch < 0:
            raise ValueError('Training epoch must be zero-based and nonnegative')
        self._epoch_zero_based = epoch
        self.updating_layer_selections = self.training and epoch >= self.warmup_epochs

    def train(self, mode=True):
        super().train(mode)
        self.updating_layer_selections = bool(mode and getattr(self, '_epoch_zero_based', -1) >= self.warmup_epochs)
        return self

    def forward(self, images):
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 448, 448):
            raise ValueError('IELT expects normalized [B,3,448,448] images')
        # Mirrors Lightning on_validation_epoch_start without depending on the
        # caller to invoke a callback; the actual author forward is unchanged.
        self.updating_layer_selections = bool(self.training and self._epoch_zero_based >= self.warmup_epochs)
        return super().forward(images)

    def loss(self, part_logits, complement_logits, labels):
        # Preserve the author's native-logit CE calls. The trainer supplies
        # the same autocast context as Lightning; an explicit pre-cast changes
        # CUDA loss values and gradients even when called within autocast.
        part = self.objective(part_logits, labels)
        complement = self.objective(complement_logits, labels)
        return torch.lerp(part, complement, self.alpha)

    def _apply(self, fn, recurse=True):
        rates = self.select_rate if hasattr(self, 'select_rate') else None
        result = super()._apply(fn, recurse=recurse)
        if rates is not None:
            self.select_rate = rates.to(device=self.select_rate.device, dtype=torch.float32)
            self._sync_selection_counts()
        return result

    def load_state_dict(self, state_dict, strict=True, assign=False):
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self._sync_selection_counts()
        self.updating_layer_selections = bool(self.training and self._epoch_zero_based >= self.warmup_epochs)
        return result

    def make_parameter_groups(self, weight_decay=1e-5):
        if weight_decay != 1e-5:
            raise ValueError('Frozen Elusive IELT weight decay is 1e-5')
        groups, assignments = author_make_parameter_groups(self, base_lr=.01,
                    finetune_lr_scale=.1, weight_decay=weight_decay,
                    finetune_list=self.finetune_list, decouple=True)
        parameters = [parameter for group in groups for parameter in group['params']]
        expected = {id(p) for p in self.parameters() if p.requires_grad}
        if len(parameters) != len({id(p) for p in parameters}) or {id(p) for p in parameters} != expected:
            raise ValueError('IELT optimizer assignment is duplicated or incomplete')
        by_name = {name: {'role': role, 'decay': decay} for name, role, decay in assignments}
        for name in ('backbone.cls_token', 'backbone.pos_embed'):
            if by_name[name]['role'] != 'scratch':
                raise ValueError('Author root ViT tokens must retain scratch learning rate')
        for name, detail in by_name.items():
            if name.startswith(('cross_layer_refinement.', 'key_block.', 'key_norm.', 'head.')) and detail['role'] != 'scratch':
                raise ValueError('New IELT method components must retain scratch learning rate')
        audit = {'assignments': [{'name': name, 'role': role, 'decay': decay}
                                 for name, role, decay in assignments],
                 'finetune_submodules': list(self.finetune_list),
                 'order': ['scratch/decay', 'scratch/no_decay', 'finetune/decay', 'finetune/no_decay'],
                 'max_learning_rates': [group['lr'] for group in groups],
                 'groups': [{'lr': group['lr'], 'weight_decay': group['weight_decay'],
                             'parameter_tensors': len(group['params']),
                             'parameters': sum(p.numel() for p in group['params'])} for group in groups]}
        return groups, audit
