import torch
from torch.nn import Conv2d, Sequential, ModuleList, ReLU, BatchNorm2d
from ..nn.vgg import vgg

from .ssd_new import SSD
from .predictor import Predictor
from .config import vgg_ssd_config as config


def create_vgg_ssd_new(num_classes, trained_model, split=None, is_test=False):
    base_net2 = torch.nn.Sequential(*trained_model.base_net[23:35])

    source_layer_indexes = trained_model.source_layer_indexes

    extras = trained_model.extras

    regression_headers = trained_model.regression_headers

    classification_headers = trained_model.classification_headers

    if split == None:
        base_net1 = torch.nn.Sequential(*trained_model.base_net[:23])
        return SSD(
            num_classes,
            base_net1,
            base_net2,
            source_layer_indexes,
            extras,
            classification_headers,
            regression_headers,
            is_test=is_test,
            config=config,
        )
    else:
        maxpooling_layers = {
            0: 5,
            1: 10,
            2: 17,
        }  # indexes for maxpooling layers of the base net.
        base = trained_model.base_net[:23]
        model1 = torch.nn.Sequential(*base[: maxpooling_layers[int(split)]])
        base_net1 = torch.nn.Sequential(*base[maxpooling_layers[int(split)] :])
        return (
            model1,
            SSD(
                num_classes,
                base_net1,
                base_net2,
                source_layer_indexes,
                extras,
                classification_headers,
                regression_headers,
                is_test=is_test,
                config=config,
            ),
        )
