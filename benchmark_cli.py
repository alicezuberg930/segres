from __future__ import annotations

import sys
from pathlib import Path

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmarks.cli import main

if __name__ == "__main__":
    main()
