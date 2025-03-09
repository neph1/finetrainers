from typing import List

import torch

from ..utils.import_utils import is_kornia_available
from .base import ProcessorMixin


if is_kornia_available():
    import kornia


class CannyProcessor(ProcessorMixin):
    r"""
    Processor for obtaining the Canny edge detection of an image.

    Args:
        output_names (`List[str]`):
            The names of the outputs that the processor should return. The first output is the Canny edge detection of
            the input image.
    """

    def __init__(self, output_names: List[str] = None) -> None:
        super().__init__()

        self.output_names = output_names

        assert len(output_names) == 1

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        r"""
        Obtain the Canny edge detection of the input image.

        Args:
            image (`torch.Tensor`):
                The input image for which the Canny edge detection should be obtained.
        """
        # TODO(aryan): think about how one can pass parameters to the underlying function from
        # a UI perspective. It's important to think about ProcessorMixin in terms of a Graph-based
        # data processing pipeline.
        return kornia.filters.canny(image)
