from typing import Any

import cv2
import numpy as np
import albumentations as A
from albumentations.core.transforms_interface import (BaseTransformInitSchema, ImageOnlyTransform)


class Watermark(ImageOnlyTransform):
    """
    Applies a semi-transparent, repeating white watermark to an image.

    The transform loads a watermark image from a specified path, converts it to a white
    shape based on its content (using an alpha channel if present, or brightness otherwise),
    and then tiles this shape across the entire input image with a given transparency.

    Args:
        watermark_path (str): The file path to the watermark image. Any format supported
            by OpenCV (e.g., PNG, JPG) can be used. PNGs with transparency are handled correctly.
        alpha (float): The transparency of the watermark. A value of 0.0 means the
            watermark is fully transparent (invisible), while 1.0 means it is fully opaque (solid white).
            Values should be between 0 and 1. Default is 0.3.
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    Example:
        >>> import albumentations as A
        >>> # Assume 'logo.png' is a watermark image on disk
        >>> transform = Watermark(watermark_path='logo.png', alpha=0.4, p=1.0)
        >>> watermarked_image = transform(image=my_image)['image']
    """

    class InitSchema(BaseTransformInitSchema):
        watermark_path: str | None
        alpha: float | None

    def __init__(
            self,
            watermark_path: str | None = None,
            alpha: float | None = 0.3,
            p: float = 0.5,
    ):
        super().__init__(p=p)
        self.alpha = np.clip(alpha, 0.0, 1.0)

        # Load the watermark image using OpenCV
        if watermark_path is not None:
            watermark_img = cv2.imread(watermark_path, cv2.IMREAD_UNCHANGED)
            if watermark_img is None:
                raise FileNotFoundError(f"Could not load watermark image from path: {watermark_path}")

            # --- Process the watermark to create a binary mask ---
            # If the image has 4 channels (e.g., RGBA in a PNG), use the alpha channel for the mask
            if len(watermark_img.shape) == 3 and watermark_img.shape[2] == 4:
                mask = watermark_img[:, :, 3]
            # Otherwise, convert to grayscale and threshold to get the shape
            else:
                if len(watermark_img.shape) == 3:
                    gray_watermark = cv2.cvtColor(watermark_img, cv2.COLOR_BGR2GRAY)
                else:  # Already grayscale
                    gray_watermark = watermark_img
                # Consider any pixel brighter than 1 as part of the watermark
                _, mask = cv2.threshold(gray_watermark, 1, 255, cv2.THRESH_BINARY)

            # Store the final mask as a boolean array for efficient tiling
            self.watermark_mask = mask > 0

    def apply(self, img: np.ndarray, mask: np.ndarray, **params: any) -> np.ndarray:
        """
        Applies the watermark transform to the given image.

        Args:
            img (np.ndarray): The input image to apply the watermark to.
            **params: Albumentations-style parameters (not used in this transform).

        Returns:
            np.ndarray: The image with the watermark applied.
        """

        # Preserve the original data type for the final output
        original_dtype = img.dtype
        # Perform blending calculations in float32 to avoid data type overflow issues
        img_float = img.astype(np.float32)

        # Create a solid white layer of the same size as the image
        white_layer = np.full(img.shape, 255.0, dtype=np.float32)

        # Blend the original image with the white layer using the specified alpha
        # Formula: result = (1 - alpha) * image + alpha * white_layer
        blended_pixels = (1 - self.alpha) * img_float + self.alpha * white_layer

        # Use the tiled mask to selectively apply the blended white pixels onto the image
        # Where the mask is True, use the blended pixel; otherwise, use the original pixel.
        output_img = np.where(mask, blended_pixels, img_float)

        # Convert the result back to the original data type, clipping if necessary
        if np.issubdtype(original_dtype, np.integer):
            output_img = np.clip(output_img, 0, 255)

        return output_img.astype(original_dtype)

    def get_params_dependent_on_data(
            self,
            params: dict[str, Any],
            data: dict[str, Any],
    ) -> tuple[str, Any]:
        img_h, img_w = params["shape"][:2]
        wm_h, wm_w = self.watermark_mask.shape

        # Tile the boolean watermark mask to cover the entire image area
        num_repeats_h = int(np.ceil(img_h / wm_h))
        num_repeats_w = int(np.ceil(img_w / wm_w))
        tiled_mask = np.tile(self.watermark_mask, (num_repeats_h, num_repeats_w))

        # Crop the tiled mask to the exact dimensions of the input image
        tiled_mask = tiled_mask[:img_h, :img_w]

        # Add a channel dimension to the mask for broadcasting (e.g., from (H, W) to (H, W, 1))
        tiled_mask_3d = tiled_mask[..., np.newaxis]

        return {
            "mask": tiled_mask_3d
        }
