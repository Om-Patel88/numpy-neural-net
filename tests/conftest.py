import sys
from pathlib import Path

# make the package importable when running pytest from a fresh clone
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
