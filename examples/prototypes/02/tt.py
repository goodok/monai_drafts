from hydra._internal.utils import _locate
from monai.transforms.adaptors import adaptor as adaptor_base
from monai.transforms.intensity.dictionary import ScaleIntensityDict  # ScaleIntensityDict - alias of ScaleIntensityd, ScaleIntensityD
from monai.transforms.utility.dictionary import ToTensorD 

from transforms_templates.transforms.compose import Compose
from transforms_templates.transforms.io.mesh import LoadPLYd
from transforms_templates.transforms.io.image2d import LoadPNGandJSONd
from transforms_templates.transforms.utility.image import Transposed

from transforms_templates.transforms.spatial.main import RandRotateXY, RandFlipX, CropRect
from transforms_templates.transforms.convert.main import ToPointCloud
from transforms_templates.transforms.other.points import SamplePoints
from transforms_templates.transforms.convert.voxelize import ToSparseVoxels
from transforms_templates.transforms.utility.debug import DebugDict



def adaptor(function, outputs, inputs=None):
    """
    See:
        https://github.com/Project-MONAI/tutorials/blob/master/modules/integrate_3rd_party_transforms.ipynb

        https://github.com/Project-MONAI/MONAI/blob/master/monai/transforms/adaptors.py

    """
    f = _locate(function)

    # TODO: better work with types
    # adaptor_base: 'outputs' must be one of (<class 'str'>, <class 'list'>, <class 'tuple'>) but is <class 'omegaconf.listconfig.ListConfig'>
    outputs = list(outputs)

    return adaptor_base(f(), outputs, inputs)

