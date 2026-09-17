from typing import Any

import cv2
import numpy as np
import albumentations as A
from albumentations.core.transforms_interface import (BaseTransformInitSchema, ImageOnlyTransform)


class DrawButton(ImageOnlyTransform):
    """
    Draws a button with specified text on the bottom-center of an image.

    This transform simulates a UI element by drawing a colored, filled rectangle
    with text centered within it. The button's position is fixed to the
    horizontal center and bottom of the image, with adjustable padding.

    Args:
        text (str): The text to display on the button. Default: "Capture".
        padding_bottom (int): The distance in pixels from the bottom edge of the image
            to the bottom edge of the button. Default: 20.
        button_color (tuple[int, int, int]): The BGR color of the button.
            Default: (40, 180, 40) which is a shade of green.
        text_color (tuple[int, int, int]): The BGR color of the text.
            Default: (255, 255, 255) which is white.
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8
    """

    class InitSchema(BaseTransformInitSchema):
        text: str | None
        padding_bottom: int | None
        button_color: tuple[int, int, int] | None
        text_color: tuple[int, int, int] | None

    def __init__(
            self,
            text: str = "Capture",
            padding_bottom: int = 20,
            button_color: tuple[int, int, int] = (40, 180, 40),  # A nice green
            text_color: tuple[int, int, int] = (255, 255, 255),  # White
            p: float = 0.5,
    ):
        super().__init__(p=p)
        self.text = text
        self.padding_bottom = padding_bottom
        self.button_color = button_color
        self.text_color = text_color

        # --- Define font properties ---
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.8
        self.font_thickness = 2
        self.button_text_padding = 15  # Internal padding within the button

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
        img_with_button = img.copy()
        img_h, img_w = img_with_button.shape[:2]

        # 1. Determine the size of the text to calculate button dimensions
        (text_w, text_h), baseline = cv2.getTextSize(
            self.text, self.font, self.font_scale, self.font_thickness
        )

        # 2. Calculate button dimensions based on text size and internal padding
        button_w = text_w + 2 * self.button_text_padding
        button_h = text_h + 2 * self.button_text_padding

        # 3. Calculate the button's top-left (x1, y1) and bottom-right (x2, y2) coordinates
        # Horizontally center the button
        x1 = (img_w - button_w) // 2
        x2 = x1 + button_w

        # Position the button at the bottom with the specified padding
        y2 = img_h - self.padding_bottom
        y1 = y2 - button_h

        # 4. Draw the filled rectangle for the button
        cv2.rectangle(
            img_with_button,
            (x1, y1),
            (x2, y2),
            self.button_color,
            -1  # Use -1 for a filled rectangle
        )

        # 5. Calculate the text position to be centered inside the button
        text_x = x1 + self.button_text_padding
        text_y = y1 + self.button_text_padding + text_h

        # 6. Draw the text on the button
        cv2.putText(
            img_with_button,
            self.text,
            (text_x, text_y),
            self.font,
            self.font_scale,
            self.text_color,
            self.font_thickness,
            cv2.LINE_AA  # For smoother text rendering
        )

        return img_with_button