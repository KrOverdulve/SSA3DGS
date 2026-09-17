from typing import Any

import cv2
import numpy as np
import albumentations as A
from albumentations.core.transforms_interface import (BaseTransformInitSchema, ImageOnlyTransform)
import imageio.v3 as iio

class Dashboard(ImageOnlyTransform):
    """
    Draws a button with specified text on the bottom-center of an image.

    This transform simulates a UI element by drawing a colored, filled rectangle
    with text centered within it. The button's position is fixed to the
    horizontal center and bottom of the image, with adjustable padding.

    Args:

    Targets:
        image

    Image types:
        uint8
    """

    class InitSchema(BaseTransformInitSchema):
        image_path: str | None

    def __init__(
            self,
            image_path: str | None = None,
            p: float = 1.0,
    ):
        super().__init__(p=p)

        image_path = "dashboard2.png"
        self.dashboard_overlay = iio.imread(image_path)

    def apply(self, img: np.ndarray, **params: any) -> np.ndarray:
        """
        Applies the drawing transform to the given image.

        Args:
            img (np.ndarray): The input image.
            **params: Albumentations-style parameters (not used).

        Returns:
            np.ndarray: The image with the button drawn on it.
        """
        # Work on a copy to avoid modifying the original image array
        img_h, img_w = img.shape[:2]
        dashboard_overlay = cv2.resize(self.dashboard_overlay, (img_w, img_h))
        img_with_dashboard = dashboard_overlay[..., 3:4] * (dashboard_overlay[..., :3] / 255.0) + (1 - (dashboard_overlay[..., 3:4] / 255.0)) * img
        img_with_dashboard = img_with_dashboard.astype(np.uint8)

        return img_with_dashboard