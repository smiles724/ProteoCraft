"""The complete ESMFold2 runtime must import without Protenix installed."""
import subprocess
import sys
from pathlib import Path


def test_esmfold2_runtime_does_not_import_protenix():
    code = '''
import importlib.abc, sys
class BlockProtenix(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "protenix" or fullname.startswith("protenix."):
            raise AssertionError("ESMFold2 imported optional dependency " + fullname)
sys.meta_path.insert(0, BlockProtenix())
from bindcraft.losses import compute_rg_loss
from bindcraft.hallucination.esmfold2_runtime import ESMFold2RuntimeBackend
from bindcraft.hallucination.esmfold2_pipeline import run_esmfold2_mcts_hallucination
assert callable(compute_rg_loss)
'''
    result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).parents[2],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
