from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from soar import SegmentationModel

def test_model_initialization():
    """Verify all resolution-aware SOAR1 models instantiate and output correct spatial shapes."""
    configs = [
        "configs/models/soar_micro1.yaml",
        "configs/models/soar_trm.yaml",
        "configs/models/soar_nano1.yaml",
        "configs/models/soar_small1.yaml",
        "configs/models/soar_medium1.yaml",
        "configs/models/soar_large1.yaml",
        "configs/models/soar_xlarge1.yaml",
    ]
    for cfg in configs:
        model = SegmentationModel(cfg, verbose=False)
        x = torch.randn(1, 3, 256, 256)
        out = model(x)
        assert out.shape == (1, 1, 256, 256), f"{cfg} output shape mismatch: {out.shape}"


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    test_model_initialization()
    print("test_models passed successfully!")
