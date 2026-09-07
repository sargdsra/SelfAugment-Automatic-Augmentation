"""
Data augmentation operations for SelfAugment.

Defines the set of image transformations used in augmentation policies.
Based on the operations from AutoAugment, RandAugment, and Fast AutoAugment.

Operations include:
- Geometric: shear_x, shear_y, translate_x, translate_y, rotate
- Color: color, contrast, brightness, sharpness, solarize, invert
- Other: cutout, auto_contrast, equalize, posterize
"""

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from typing import Tuple, Optional, Callable, Dict, List
import random


class AugmentationOps:
    """
    Collection of image augmentation operations.
    
    Each operation has:
    - A function that applies the transformation
    - A magnitude parameter (lambda) controlling strength
    - A probability parameter (p) controlling chance of application
    """
    
    @staticmethod
    def shear_x(img: Image.Image, magnitude: float) -> Image.Image:
        """
        Apply shear transformation along x-axis.
        
        Args:
            img: PIL Image
            magnitude: Shear angle in degrees (0-30)
            
        Returns:
            Transformed image
        """
        return img.transform(
            img.size,
            Image.AFFINE,
            (1, magnitude * 0.01, 0, 0, 1, 0),
            resample=Image.BICUBIC
        )
    
    @staticmethod
    def shear_y(img: Image.Image, magnitude: float) -> Image.Image:
        """
        Apply shear transformation along y-axis.
        
        Args:
            img: PIL Image
            magnitude: Shear angle in degrees (0-30)
            
        Returns:
            Transformed image
        """
        return img.transform(
            img.size,
            Image.AFFINE,
            (1, 0, 0, magnitude * 0.01, 1, 0),
            resample=Image.BICUBIC
        )
    
    @staticmethod
    def translate_x(img: Image.Image, magnitude: float) -> Image.Image:
        """
        Translate image along x-axis.
        
        Args:
            img: PIL Image
            magnitude: Translation amount (0-100 pixels)
            
        Returns:
            Transformed image
        """
        pixels = magnitude * 0.01 * img.size[0]
        return img.transform(
            img.size,
            Image.AFFINE,
            (1, 0, pixels, 0, 1, 0),
            resample=Image.BICUBIC
        )
    
    @staticmethod
    def translate_y(img: Image.Image, magnitude: float) -> Image.Image:
        """
        Translate image along y-axis.
        
        Args:
            img: PIL Image
            magnitude: Translation amount (0-100 pixels)
            
        Returns:
            Transformed image
        """
        pixels = magnitude * 0.01 * img.size[1]
        return img.transform(
            img.size,
            Image.AFFINE,
            (1, 0, 0, 0, 1, pixels),
            resample=Image.BICUBIC
        )
    
    @staticmethod
    def rotate(img: Image.Image, magnitude: float) -> Image.Image:
        """
        Rotate image by a given angle.
        
        Args:
            img: PIL Image
            magnitude: Rotation angle in degrees (0-30)
            
        Returns:
            Transformed image
        """
        return img.rotate(magnitude, resample=Image.BICUBIC)
    
    @staticmethod
    def color(img: Image.Image, magnitude: float) -> Image.Image:
        """
        Adjust color balance of image.
        
        Args:
            img: PIL Image
            magnitude: Color factor (0.1-1.9 in AutoAugment)
            
        Returns:
            Transformed image
        """
        return ImageEnhance.Color(img).enhance(1 + magnitude * 0.1)
    
    @staticmethod
    def contrast(img: Image.Image, magnitude: float) -> Image.Image:
        """
        Adjust contrast of image.
        
        Args:
            img: PIL Image
            magnitude: Contrast factor (0.1-1.9)
            
        Returns:
            Transformed image
        """
        return ImageEnhance.Contrast(img).enhance(1 + magnitude * 0.1)
    
    @staticmethod
    def brightness(img: Image.Image, magnitude: float) -> Image.Image:
        """
        Adjust brightness of image.
        
        Args:
            img: PIL Image
            magnitude: Brightness factor (0.1-1.9)
            
        Returns:
            Transformed image
        """
        return ImageEnhance.Brightness(img).enhance(1 + magnitude * 0.1)
    
    @staticmethod
    def sharpness(img: Image.Image, magnitude: float) -> Image.Image:
        """
        Adjust sharpness of image.
        
        Args:
            img: PIL Image
            magnitude: Sharpness factor (0.1-1.9)
            
        Returns:
            Transformed image
        """
        return ImageEnhance.Sharpness(img).enhance(1 + magnitude * 0.1)
    
    @staticmethod
    def solarize(img: Image.Image, magnitude: float) -> Image.Image:
        """
        Invert all pixels above a threshold.
        
        Args:
            img: PIL Image
            magnitude: Solarization threshold (0-255)
            
        Returns:
            Transformed image
        """
        return ImageOps.solarize(img, int(magnitude * 2.55))
    
    @staticmethod
    def invert(img: Image.Image, magnitude: float) -> Image.Image:
        """
        Invert image colors.
        
        Args:
            img: PIL Image
            magnitude: Not used for invert
            
        Returns:
            Transformed image
        """
        return ImageOps.invert(img)
    
    @staticmethod
    def auto_contrast(img: Image.Image, magnitude: float) -> Image.Image:
        """
        Apply automatic contrast adjustment.
        
        Args:
            img: PIL Image
            magnitude: Not used for auto_contrast
            
        Returns:
            Transformed image
        """
        return ImageOps.autocontrast(img)
    
    @staticmethod
    def equalize(img: Image.Image, magnitude: float) -> Image.Image:
        """
        Equalize image histogram.
        
        Args:
            img: PIL Image
            magnitude: Not used for equalize
            
        Returns:
            Transformed image
        """
        return ImageOps.equalize(img)
    
    @staticmethod
    def posterize(img: Image.Image, magnitude: float) -> Image.Image:
        """
        Reduce number of bits for each color channel.
        
        Args:
            img: PIL Image
            magnitude: Number of bits (1-8)
            
        Returns:
            Transformed image
        """
        bits = max(1, int(8 - magnitude * 0.1))
        return ImageOps.posterize(img, bits)
    
    @staticmethod
    def cutout(img: Image.Image, magnitude: float) -> Image.Image:
        """
        Randomly mask out a square region of the image.
        
        Args:
            img: PIL Image
            magnitude: Size of cutout as fraction of image (0-0.5)
            
        Returns:
            Transformed image
        """
        img_array = np.array(img)
        h, w = img_array.shape[:2]
        
        # Calculate cutout size
        size = int(magnitude * 0.01 * min(h, w))
        
        # Random position
        y = np.random.randint(0, h - size)
        x = np.random.randint(0, w - size)
        
        # Apply cutout (set to 0)
        img_array[y:y+size, x:x+size] = 0
        
        return Image.fromarray(img_array)
    
    @staticmethod
    def identity(img: Image.Image, magnitude: float) -> Image.Image:
        """
        Identity operation (no transformation).
        
        Args:
            img: PIL Image
            magnitude: Not used
            
        Returns:
            Original image
        """
        return img


# Mapping of operation names to functions
OPERATIONS: Dict[str, Callable] = {
    'shear_x': AugmentationOps.shear_x,
    'shear_y': AugmentationOps.shear_y,
    'translate_x': AugmentationOps.translate_x,
    'translate_y': AugmentationOps.translate_y,
    'rotate': AugmentationOps.rotate,
    'color': AugmentationOps.color,
    'contrast': AugmentationOps.contrast,
    'brightness': AugmentationOps.brightness,
    'sharpness': AugmentationOps.sharpness,
    'solarize': AugmentationOps.solarize,
    'invert': AugmentationOps.invert,
    'auto_contrast': AugmentationOps.auto_contrast,
    'equalize': AugmentationOps.equalize,
    'posterize': AugmentationOps.posterize,
    'cutout': AugmentationOps.cutout,
    'identity': AugmentationOps.identity,
}

# Magnitude ranges for each operation (from Fast AutoAugment)
MAGNITUDE_RANGES: Dict[str, Tuple[float, float]] = {
    'shear_x': (0, 30),
    'shear_y': (0, 30),
    'translate_x': (0, 100),
    'translate_y': (0, 100),
    'rotate': (0, 30),
    'color': (0, 9),
    'contrast': (0, 9),
    'brightness': (0, 9),
    'sharpness': (0, 9),
    'solarize': (0, 100),
    'invert': (0, 0),      # No magnitude
    'auto_contrast': (0, 0),
    'equalize': (0, 0),
    'posterize': (0, 40),
    'cutout': (0, 50),
    'identity': (0, 0),
}