# Provenance

Based on **Forming Auxiliary High-confident Instance-level Loss to Promote Learning from Label Proportions**, arXiv:2411.10364v2, Tianhao Ma et al.

- Paper: https://arxiv.org/abs/2411.10364
- Repository: https://github.com/TianhaoMa5/LLP-AHIL
- Audited commit: `65c4bfa201f4a4d820452e5be9de51c5f6c4b326`
- `WideResNet.py` -> `llp_shrink/model.py`: parameterized input channels, disabled unused projection by default, protected optional L2 normalization, removed standalone demo.
- `datasets/rand.py` -> `llp_shrink/vendor/rand_rgb.py`: attribution header, integer-safe autocontrast and posterize.
- `datasets/randaugment_grey.py` -> `llp_shrink/vendor/rand_gray.py`: same integer fixes and attribution.
- Original SHA-256 values are in `docs/upstream_manifest.json`.

The upstream snapshot does not contain a LICENSE file. This document records provenance; it does not assign a new license to upstream code. No third-party paper PDFs or datasets are bundled or uploaded.

The new loss, data orchestration, checkpointing, tests, CLI, experiment scripts and documentation implement **Learning from Label Proportions with Shrinking Bag** by Xueyou Lu, Yueyi Wang and Wenxu Gao, from the local Lu.pdf supplied for this task. The repository retains its original initial commit.
