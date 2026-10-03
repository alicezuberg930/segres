"""
SOAR Benchmarks Suite for Q1 Peer-Reviewed Research
Contains full training, epoch-by-epoch validation, and symmetrical inference pipelines for:
  1. unet       - U-Net (Ronneberger et al., MICCAI 2015)
  2. dlinknet   - D-LinkNet (Zhou et al., CVPR 2018)
  3. csnet      - CS-Net (Mou et al., MICCAI 2019)
  4. bisenetv2  - BiSeNet V2 (Yu et al., IJCV 2021)
  5. ddrnet     - DDRNet-23-slim / DDRNet-23 (Hong et al., 2021)
  6. pidnet     - PIDNet-S / PIDNet-M (Xu et al., CVPR 2023)
  7. segformer  - SegFormer-B0 / SegFormer-B1 (Xie et al., NeurIPS 2021)
"""

from .common.metrics import BenchmarkMetricAccumulator, extract_boundary, soft_skeletonize
from .common.engine import BenchmarkTrainer, CombinedSegmentationLoss
from .common.predictor import BenchmarkPredictor

__all__ = [
    "BenchmarkMetricAccumulator",
    "extract_boundary",
    "soft_skeletonize",
    "BenchmarkTrainer",
    "CombinedSegmentationLoss",
    "BenchmarkPredictor",
]
