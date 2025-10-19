"""
OCR Service for multi-modal text extraction from images.

This module provides OCR capabilities using EasyOCR and Pytesseract
for extracting text from images with preprocessing and multi-language support.
Integrates with the Material Kai Vision Platform for enhanced multi-modal processing.
"""

import logging
import os
import tempfile
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import easyocr
import pytesseract
from dataclasses import dataclass

# Fix for Pillow 10.0+ compatibility with EasyOCR
if not hasattr(Image, 'ANTIALIAS'):
    Image.ANTIALIAS = Image.LANCZOS

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """Data class for OCR extraction results."""
    text: str
    confidence: float
    bbox: Optional[List[int]] = None  # [x1, y1, x2, y2]
    language: Optional[str] = None
    method: Optional[str] = None  # 'easyocr' or 'tesseract'


@dataclass
class OCRConfig:
    """Configuration for OCR processing."""
    languages: List[str] = None
    use_gpu: bool = False
    confidence_threshold: float = 0.5
    preprocessing_enabled: bool = True
    
    def __post_init__(self):
        if self.languages is None:
            self.languages = ['en']


class ImagePreprocessor:
    """Image preprocessing utilities for better OCR accuracy."""
    
    @staticmethod
    def enhance_image(image: np.ndarray) -> np.ndarray:
        """
        Apply image enhancements to improve OCR accuracy.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Enhanced image as numpy array
        """
        try:
            # Convert to PIL Image for enhancement
            if len(image.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image)
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(1.5)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(pil_image)
            pil_image = enhancer.enhance(2.0)
            
            # Apply slight blur to reduce noise
            pil_image = pil_image.filter(ImageFilter.MedianFilter(size=3))
            
            # Convert back to numpy array
            enhanced = np.array(pil_image)
            
            # Convert back to BGR if original was BGR
            if len(image.shape) == 3:
                enhanced = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
                
            return enhanced
            
        except Exception as e:
            logger.warning(f"Image enhancement failed: {str(e)}")
            return image
    
    @staticmethod
    def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for optimal OCR performance.

        Args:
            image: Input image as numpy array

        Returns:
            Preprocessed image as numpy array
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )

            # Morphological operations to clean up
            kernel = np.ones((2, 2), np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            return cleaned

        except Exception as e:
            logger.warning(f"Image preprocessing failed: {str(e)}")
            return image


class OCRService:
    """
    OCR Service providing text extraction from images using EasyOCR and Pytesseract.
    
    Features:
    - EasyOCR integration for local text extraction
    - Pytesseract fallback support
    - Multi-language OCR capabilities
    - Image preprocessing for better accuracy
    - Integration with Material Kai Vision Platform
    """
    
    def __init__(self, config: Optional[OCRConfig] = None):
        """
        Initialize OCR Service.
        
        Args:
            config: OCR configuration settings
        """
        self.config = config or OCRConfig()
        self.preprocessor = ImagePreprocessor()
        self._easyocr_reader: Optional[easyocr.Reader] = None
        self._initialized = False
        
        logger.info(f"OCR Service initialized with languages: {self.config.languages}")
    
    def initialize(self) -> None:
        """
        Initialize OCR engines and validate dependencies.
        
        Raises:
            RuntimeError: If OCR engines cannot be initialized
        """
        try:
            # Initialize EasyOCR reader
            self._easyocr_reader = easyocr.Reader(
                self.config.languages,
                gpu=self.config.use_gpu
            )

            self._initialized = True
            logger.info("OCR Service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize OCR Service: {str(e)}")
            raise RuntimeError(f"OCR Service initialization failed: {str(e)}")
    
    def extract_text_easyocr(self, image: np.ndarray) -> List[OCRResult]:
        """
        Extract text using EasyOCR.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of OCR results with text, confidence, and bounding boxes
        """
        if not self._initialized:
            self.initialize()
        
        try:
            results = self._easyocr_reader.readtext(image)
            ocr_results = []
            
            for bbox, text, confidence in results:
                if confidence >= self.config.confidence_threshold:
                    # Convert bbox to [x1, y1, x2, y2] format
                    bbox_coords = [
                        int(min(point[0] for point in bbox)),  # x1
                        int(min(point[1] for point in bbox)),  # y1
                        int(max(point[0] for point in bbox)),  # x2
                        int(max(point[1] for point in bbox))   # y2
                    ]
                    
                    ocr_results.append(OCRResult(
                        text=text.strip(),
                        confidence=confidence,
                        bbox=bbox_coords,
                        method='easyocr'
                    ))
            
            logger.debug(f"EasyOCR extracted {len(ocr_results)} text regions")
            return ocr_results
            
        except Exception as e:
            logger.error(f"EasyOCR extraction failed: {str(e)}")
            raise
    

    def extract_text_from_image(
        self, 
        image_input: Union[str, Path, np.ndarray, Image.Image],
        use_preprocessing: bool = None
    ) -> List[OCRResult]:
        """
        Extract text from image using available OCR engines.
        
        Args:
            image_input: Image file path, numpy array, or PIL Image
            use_preprocessing: Whether to apply image preprocessing
            
        Returns:
            List of OCR results from all successful engines
            
        Raises:
            ValueError: If image input is invalid
            RuntimeError: If all OCR engines fail
        """
        if not self._initialized:
            self.initialize()
        
        # Load and convert image to numpy array
        image = self._load_image(image_input)
        
        # Apply preprocessing if enabled
        if use_preprocessing or (use_preprocessing is None and self.config.preprocessing_enabled):
            image = self.preprocessor.enhance_image(image)
            image = self.preprocessor.preprocess_for_ocr(image)
        
        # Use EasyOCR for text extraction
        try:
            easyocr_results = self.extract_text_easyocr(image)
            logger.info(f"EasyOCR extracted {len(easyocr_results)} text regions")
            return easyocr_results
        except Exception as e:
            logger.error(f"EasyOCR extraction failed: {str(e)}")
            raise
        
        # Sort results by confidence (highest first)
        all_results.sort(key=lambda x: x.confidence, reverse=True)
        
        logger.info(f"Total OCR extraction: {len(all_results)} text regions")
        return all_results
    
    def extract_text_simple(
        self, 
        image_input: Union[str, Path, np.ndarray, Image.Image]
    ) -> str:
        """
        Extract text from image and return as simple concatenated string.
        
        Args:
            image_input: Image file path, numpy array, or PIL Image
            
        Returns:
            Extracted text as a single string
        """
        results = self.extract_text_from_image(image_input)
        
        # Concatenate all text results
        text_parts = [result.text for result in results if result.text.strip()]
        return ' '.join(text_parts)
    
    def get_text_with_confidence(
        self, 
        image_input: Union[str, Path, np.ndarray, Image.Image],
        min_confidence: float = 0.5
    ) -> Dict[str, Any]:
        """
        Extract text with confidence metrics and metadata.
        
        Args:
            image_input: Image file path, numpy array, or PIL Image
            min_confidence: Minimum confidence threshold for results
            
        Returns:
            Dictionary with extracted text, confidence metrics, and metadata
        """
        results = self.extract_text_from_image(image_input)
        
        # Filter by confidence
        filtered_results = [r for r in results if r.confidence >= min_confidence]
        
        if not filtered_results:
            return {
                'text': '',
                'confidence': 0.0,
                'word_count': 0,
                'regions': 0,
                'methods_used': [],
                'metadata': {}
            }
        
        # Calculate metrics
        all_text = ' '.join([r.text for r in filtered_results])
        avg_confidence = sum(r.confidence for r in filtered_results) / len(filtered_results)
        methods_used = list(set(r.method for r in filtered_results if r.method))
        
        return {
            'text': all_text,
            'confidence': avg_confidence,
            'word_count': len(all_text.split()),
            'regions': len(filtered_results),
            'methods_used': methods_used,
            'metadata': {
                'languages': self.config.languages,
                'preprocessing_used': self.config.preprocessing_enabled,
                'results': [
                    {
                        'text': r.text,
                        'confidence': r.confidence,
                        'bbox': r.bbox,
                        'method': r.method
                    } for r in filtered_results
                ]
            }
        }
    
    def _load_image(self, image_input: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Load and convert image to numpy array.
        
        Args:
            image_input: Image in various formats
            
        Returns:
            Image as numpy array
            
        Raises:
            ValueError: If image format is not supported
        """
        try:
            if isinstance(image_input, (str, Path)):
                # Load from file path
                image_path = Path(image_input)
                if not image_path.exists():
                    raise ValueError(f"Image file not found: {image_path}")

                image = cv2.imread(str(image_path))
                if image is None:
                    raise ValueError(f"Could not load image: {image_path}")

            elif isinstance(image_input, np.ndarray):
                # Already a numpy array
                image = image_input.copy()

            elif isinstance(image_input, Image.Image):
                # Convert PIL Image to numpy array
                image = np.array(image_input)
                if len(image.shape) == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    
            else:
                raise ValueError(f"Unsupported image input type: {type(image_input)}")
            
            return image
            
        except Exception as e:
            logger.error(f"Failed to load image: {str(e)}")
            raise ValueError(f"Invalid image input: {str(e)}")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on OCR service.
        
        Returns:
            Dictionary with health status and capabilities
        """
        status = {
            'initialized': self._initialized,
            'easyocr_available': False,
            'tesseract_available': False,
            'languages': self.config.languages,
            'gpu_enabled': self.config.use_gpu
        }
        
        # Check EasyOCR
        try:
            if self._easyocr_reader is not None:
                status['easyocr_available'] = True
        except Exception:
            pass
        
        # Check Tesseract
        try:
            pytesseract.get_tesseract_version()
            status['tesseract_available'] = True
        except Exception:
            pass
        
        status['healthy'] = status['easyocr_available'] or status['tesseract_available']
        
        return status


# Global instance
_ocr_service: Optional[OCRService] = None


def get_ocr_service(config: Optional[OCRConfig] = None) -> OCRService:
    """
    Get the global OCR service instance.
    
    Args:
        config: OCR configuration (used only on first call)
        
    Returns:
        OCRService instance
    """
    global _ocr_service
    
    if _ocr_service is None:
        _ocr_service = OCRService(config)
    
    return _ocr_service


def initialize_ocr_service(config: Optional[OCRConfig] = None) -> None:
    """
    Initialize the global OCR service.
    
    Args:
        config: OCR configuration settings
    """
    service = get_ocr_service(config)
    service.initialize()