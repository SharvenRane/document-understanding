import os
import sys

# Make the repo root importable so `import src` works under pytest.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
