# Third-party acknowledgements

We thank the authors of [LLP-AHIL](https://github.com/TianhaoMa5/LLP-AHIL) for making their implementation available.

The following components include code from commit `65c4bfa201f4a4d820452e5be9de51c5f6c4b326`:

| Component | Source | Local changes |
| --- | --- | --- |
| `llp_shrink/model.py` | `WideResNet.py` | Configurable input channels and projection; numerical guard for optional normalization |
| `llp_shrink/vendor/rand_rgb.py` | `datasets/rand.py` | Integer-safe AutoContrast and Posterize |
| `llp_shrink/vendor/rand_gray.py` | `datasets/randaugment_grey.py` | Integer-safe AutoContrast and Posterize |

Original file hashes are recorded in [the source manifest](docs/upstream_manifest.json).
The source snapshot contains no LICENSE file; this acknowledgement does not assign a license to that code.

Reference: Tianhao Ma, Han Chen, Juncheng Hu, Yungang Zhu, and Ximing Li. *Forming Auxiliary High-confident Instance-level Loss to Promote Learning from Label Proportions*. [arXiv:2411.10364](https://arxiv.org/abs/2411.10364).
