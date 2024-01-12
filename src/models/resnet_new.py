from typing import List, Optional, Type, Union, Callable, Tuple, Any
import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck, ResNet18_Weights
from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface
from torchvision.models._api import register_model, Weights, WeightsEnum

from ..utils.utils import load_partial_state_dict

class ResNetWithActivations(ResNet):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__(
            block, layers, num_classes, zero_init_residual, groups,
            width_per_group, replace_stride_with_dilation, norm_layer
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        activations = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        activations.append(x.clone())
        x = self.layer1(x)
        activations.append(x.clone())
        x = self.layer2(x)
        activations.append(x.clone())
        x = self.layer3(x)
        activations.append(x.clone())
        x = self.layer4(x)
        activations.append(x.clone())

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        activations.append(x.clone())

        return x, activations

def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> ResNetWithActivations:

    #if weights is not None:
    #    _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
    model = ResNetWithActivations(block, layers, **kwargs)

    if weights is not None:
        load_partial_state_dict(model, weights.get_state_dict(progress=progress))

    return model

@register_model()
@handle_legacy_interface(weights=("pretrained", ResNet18_Weights.IMAGENET1K_V1))
def get_resnet18(*, weights: Optional[ResNet18_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNetWithActivations:
    """ResNet-18 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.

    Args:
        weights (:class:`~torchvision.models.ResNet18_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet18_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet18_Weights
        :members:
    """
    weights = ResNet18_Weights.verify(weights)

    return _resnet(BasicBlock, [2, 2, 2, 2], weights, progress, **kwargs)
