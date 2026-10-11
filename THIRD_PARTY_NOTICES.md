# Third-party source attribution

The training recipes and model operations use the following upstream work.

| Source | Commit | Files retained |
|---|---|---|
| [Elusive Images](https://github.com/catalys1/elusive-images-fgvc) | `37639fd61f3d8b0b1a23a5d9922a48cd2b0f8fb0` | `third_party/elusive_images/src/models/` |
| [P2P-Net](https://github.com/yangxh11/P2P-Net) | `400ad83459634597ed1f37ec6b9dad1849c5fee6` | `third_party/P2P-Net/` |

Elusive Images model source files are unchanged. The WS-DAN numerical core adapts
its model operations to plain PyTorch and computes probability fusion in FP32
log space. IELT executes selected upstream definitions with an offline pretrained
loader and a plain PyTorch training adapter.

P2P-Net training and inference operations retain the upstream algorithm. Its
unused training/evaluation entrypoints, demos, and other dataset loaders are
excluded. The retained CUB loader and numerical model/loss definitions are
unchanged. The runtime adapter replaces
the removed NumPy `np.int` alias with `np.int64` in a generated source cache.

No LICENSE file was present in either pinned upstream checkout used here. These
attributions do not grant a new license to the upstream source files.
