from os.path import join as pjoin
from .registry import register_dataset
from . import decision_mappings, info_mappings
from .dataloaders import PytorchLoader
from ..evaluation import metrics as m
from .base import Dataset
from .experiments import *
import constants as c

__all__ = c.DATASETS


@dataclass
class NoiseGeneralisationParams:
    path: str = ''
    experiments: List = field(default_factory=list)
    image_size: int = 224
    metrics: list = field(default_factory=lambda: [m.Accuracy(topk=1)])
    decision_mapping: object = decision_mappings.ImageNetProbabilitiesTo16ClassesMapping()
    info_mapping: object = info_mappings.InfoMapping()


def _get_dataset(name, params, *args, **kwargs):
    assert params is not None, 'Dataset params are missing'
    params.path = pjoin(c.DATA_DIR, name)
    return Dataset(name=name, params=params, loader=PytorchLoader, *args, **kwargs)


@register_dataset(name='blurred')
def blurred(*args, **kwargs):
    return _get_dataset(name='blurred', params=NoiseGeneralisationParams(experiments=[blur_experiment]),
                        *args, **kwargs)


@register_dataset(name='deblurred_erco')
def deblurred_erco(*args, **kwargs):
    return _get_dataset(name='deblurred_erco', params=NoiseGeneralisationParams(experiments=[blur_experiment]),
                        *args, **kwargs)


@register_dataset(name='deblurred_pmp3')
def deblurred_pmp3(*args, **kwargs):
    return _get_dataset(name='deblurred_pmp3', params=NoiseGeneralisationParams(experiments=[blur_experiment]),
                        *args, **kwargs)


@register_dataset(name='deblurred_pmp3_lambda0')
def deblurred_pmp3_lambda0(*args, **kwargs):
    return _get_dataset(name='deblurred_pmp3_lambda0', params=NoiseGeneralisationParams(experiments=[blur_experiment]),
                        *args, **kwargs)


@register_dataset(name='deblurred_nb3')
def deblurred_nb3(*args, **kwargs):
    return _get_dataset(name='deblurred_nb3', params=NoiseGeneralisationParams(experiments=[blur_experiment]),
                        *args, **kwargs)


@register_dataset(name='deblurred_nb_ls_adhoc')
def deblurred_nb_ls_adhoc(*args, **kwargs):
    return _get_dataset(name='deblurred_nb_ls_adhoc', params=NoiseGeneralisationParams(experiments=[blur_experiment]),
                        *args, **kwargs)
