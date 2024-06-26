from packaging.version import Version
import diffusers

if Version('0.29.0') <= Version(diffusers.__version__):
    from .distri_dit_sd3_pipefusion import DistriDiTSD3PipeFusion
    from .distri_dit_hunyuandit_pipefusion import HunyuanDiTPiP
from .naive_patch_dit import NaivePatchDiT
from .distri_dit_pp import DistriDiTPP
from .distri_dit_tp import DistriDiTTP
from .distri_dit_pipefusion import DistriDiTPipeFusion


