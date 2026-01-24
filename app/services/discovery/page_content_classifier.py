"""
Page Content Classifier - Phase 2 Page Range Refinement

This module classifies pages by their content type to refine page ranges.
After conservative calculation (end_page = next_start - 1), this service
analyzes each page to identify non-product content that can be trimmed.

Content Types:
- PRODUCT: Main product content (images, descriptions, specifications on product)
- ARCHITECT_INTRO: Designer/architect introduction pages
- TECHNICAL_SPEC: Technical specification pages (dimensions, materials)
- CERTIFICATION: Certificate and compliance pages
- DECORATIVE: Decorative/lifestyle images without product info
- UNKNOWN: Unclassified pages

Trimming Rules:
- Trim ARCHITECT_INTRO and DECORATIVE from BOTH ends of ranges
- Keep PRODUCT, TECHNICAL_SPEC, CERTIFICATION pages (they contain valuable info)
- UNKNOWN pages are kept (could be product content in images)
"""

import logging
import re
from enum import Enum
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class PageContentType(Enum):
    """Classification of page content."""
    PRODUCT = "product"  # Main product content
    ARCHITECT_INTRO = "architect_intro"  # Designer/architect introduction
    TECHNICAL_SPEC = "technical_spec"  # Technical specifications
    CERTIFICATION = "certification"  # Certificates and compliance
    DECORATIVE = "decorative"  # Decorative/lifestyle images
    UNKNOWN = "unknown"  # Unclassified


@dataclass
class PageClassification:
    """Result of page content classification."""
    page_num: int
    content_type: PageContentType
    confidence: float
    reasoning: str


class PageContentClassifier:
    """
    Classifies pages by their content type using text pattern analysis.

    This is a lightweight classifier that uses keyword patterns to identify
    page content types. For more accurate classification, consider using
    vision models (Qwen) in future iterations.
    """

    # Patterns for architect/designer introduction pages
    ARCHITECT_PATTERNS = [
        r'\barchitect\b',
        r'\bdesigner\b',
        r'\bstudio\b',
        r'\bfounded\b',
        r'\bestablished\b',
        r'\bbiography\b',
        r'\bportfolio\b',
        r'\bpractice\b',
        r'\bphilosophy\b',
        r'\bborn\s+in\b',
        r'\bstudied\s+at\b',
        r'\baward\s+winning\b',
        r'\bdesign\s+philosophy\b',
        r'\bcreative\s+director\b',
    ]

    # Patterns for certification/compliance pages
    CERTIFICATION_PATTERNS = [
        r'\bcertificat',
        r'\bcompliance\b',
        r'\biso\s*\d+',
        r'\bstandard\b',
        r'\bquality\s+assurance\b',
        r'\btesting\b',
        r'\bflammability\b',
        r'\benvironmental\b',
        r'\bsustainab',
        r'\brecycl',
        r'\bfsc\b',
        r'\boeko-tex\b',
        r'\bgreenguard\b',
        r'\bleed\b',
        r'\bwrcc\b',
        r'\bfire\s+rating\b',
    ]

    # Patterns for technical specification pages
    TECHNICAL_SPEC_PATTERNS = [
        r'\bspecification\b',
        r'\bdimensions?\b',
        r'\bmeasurement',
        r'\bweight\b',
        r'\bmaterial\s*:',
        r'\bcomposition\b',
        r'\btechnical\s+data\b',
        r'\bproperties\b',
        r'\bperformance\b',
        r'\babrasion\b',
        r'\bcolorfastness\b',
        r'\bpilling\b',
        r'\bmartindale\b',
        r'\bwyzenbeek\b',
    ]

    # Patterns for product content (positive indicators)
    PRODUCT_PATTERNS = [
        r'\bcolour\s*(range|options|ways)\b',
        r'\bcolor\s*(range|options|ways)\b',
        r'\bcollection\b',
        r'\bpattern\b',
        r'\bdesign\b',
        r'\bfabric\b',
        r'\btextile\b',
        r'\bwallcovering\b',
        r'\bupholstery\b',
        r'\bdrapery\b',
        r'\buse\s+code\b',
    ]

    def __init__(self):
        """Initialize classifier with compiled regex patterns."""
        self._architect_regex = re.compile(
            '|'.join(self.ARCHITECT_PATTERNS),
            re.IGNORECASE
        )
        self._certification_regex = re.compile(
            '|'.join(self.CERTIFICATION_PATTERNS),
            re.IGNORECASE
        )
        self._technical_regex = re.compile(
            '|'.join(self.TECHNICAL_SPEC_PATTERNS),
            re.IGNORECASE
        )
        self._product_regex = re.compile(
            '|'.join(self.PRODUCT_PATTERNS),
            re.IGNORECASE
        )

    def classify_page(
        self,
        page_num: int,
        page_text: str,
        product_name: Optional[str] = None
    ) -> PageClassification:
        """
        Classify a single page by its content type.

        Args:
            page_num: Page number (1-based)
            page_text: Lowercased text content of the page
            product_name: Optional product name to check for

        Returns:
            PageClassification with type and confidence
        """
        if not page_text or len(page_text.strip()) < 50:
            return PageClassification(
                page_num=page_num,
                content_type=PageContentType.DECORATIVE,
                confidence=0.7,
                reasoning="Very little text content - likely decorative/image page"
            )

        # Count pattern matches
        architect_matches = len(self._architect_regex.findall(page_text))
        certification_matches = len(self._certification_regex.findall(page_text))
        technical_matches = len(self._technical_regex.findall(page_text))
        product_matches = len(self._product_regex.findall(page_text))

        # Check if product name appears on page
        product_name_present = False
        if product_name:
            product_name_present = product_name.lower() in page_text

        # Calculate scores
        scores = {
            PageContentType.PRODUCT: product_matches * 2 + (3 if product_name_present else 0),
            PageContentType.ARCHITECT_INTRO: architect_matches * 2,
            PageContentType.TECHNICAL_SPEC: technical_matches * 2,
            PageContentType.CERTIFICATION: certification_matches * 2,
        }

        # Find highest scoring type
        max_score = max(scores.values())

        if max_score == 0:
            return PageClassification(
                page_num=page_num,
                content_type=PageContentType.UNKNOWN,
                confidence=0.5,
                reasoning="No clear content patterns detected"
            )

        # Get the winning type
        winning_type = max(scores, key=scores.get)
        total_matches = sum(scores.values())
        confidence = min(0.9, 0.5 + (scores[winning_type] / max(total_matches, 1)) * 0.4)

        # Build reasoning
        match_counts = {
            "architect": architect_matches,
            "certification": certification_matches,
            "technical": technical_matches,
            "product": product_matches
        }
        non_zero = [f"{k}={v}" for k, v in match_counts.items() if v > 0]
        reasoning = f"Pattern matches: {', '.join(non_zero) if non_zero else 'none'}"
        if product_name_present:
            reasoning += f"; product name '{product_name}' found"

        return PageClassification(
            page_num=page_num,
            content_type=winning_type,
            confidence=confidence,
            reasoning=reasoning
        )

    def classify_page_range(
        self,
        pages_content: Dict[int, str],
        start_page: int,
        end_page: int,
        product_name: Optional[str] = None
    ) -> List[PageClassification]:
        """
        Classify all pages in a range.

        Args:
            pages_content: Dict mapping page_num -> lowercased page text
            start_page: First page of range (inclusive)
            end_page: Last page of range (inclusive)
            product_name: Product name to look for

        Returns:
            List of PageClassification for each page
        """
        classifications = []

        for page_num in range(start_page, end_page + 1):
            page_text = pages_content.get(page_num, "")
            classification = self.classify_page(
                page_num=page_num,
                page_text=page_text,
                product_name=product_name
            )
            classifications.append(classification)

        return classifications

    def refine_page_range(
        self,
        classifications: List[PageClassification],
        original_start: int,
        original_end: int
    ) -> Tuple[int, int, str]:
        """
        Refine page range by trimming non-product pages from BOTH ends.

        Rules:
        - Trim ARCHITECT_INTRO and DECORATIVE from BEGINNING AND END
        - Keep PRODUCT, TECHNICAL_SPEC, CERTIFICATION pages (valuable content)
        - UNKNOWN pages are kept (could be product content in images)
        - If all pages are non-product, keep original range (safety)

        Args:
            classifications: List of PageClassification for the range
            original_start: Original start page
            original_end: Original end page

        Returns:
            Tuple of (new_start, new_end, reason)
        """
        trimmable_types = {PageContentType.ARCHITECT_INTRO, PageContentType.DECORATIVE}

        # =============================================
        # STEP 1: Trim from BEGINNING
        # =============================================
        new_start = original_start
        trimmed_from_start = []

        for classification in classifications:
            if classification.content_type in trimmable_types:
                trimmed_from_start.append(classification.page_num)
                new_start = classification.page_num + 1
            else:
                # Found a page to keep, stop trimming from start
                break

        # =============================================
        # STEP 2: Trim from END
        # =============================================
        new_end = original_end
        trimmed_from_end = []

        # Iterate backwards through classifications
        for classification in reversed(classifications):
            if classification.content_type in trimmable_types:
                trimmed_from_end.append(classification.page_num)
                new_end = classification.page_num - 1
            else:
                # Found a page to keep, stop trimming from end
                break

        # =============================================
        # STEP 3: Validate result
        # =============================================
        # Safety: don't trim if it would remove all pages or create invalid range
        if new_start > new_end:
            return original_start, original_end, "no_change: trimming would remove all pages, keeping original range"

        # Build reason
        all_trimmed = trimmed_from_start + trimmed_from_end
        if all_trimmed:
            reason = f"trimmed: start={trimmed_from_start}, end={trimmed_from_end}"
        else:
            reason = "no_change: all pages contain product content"

        return new_start, new_end, reason


def refine_product_page_ranges(
    products: List,  # List of ProductInfo objects
    pages_content: Dict[int, str],
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Refine page ranges for all products using content classification.

    This is the main entry point for Phase 2 page range refinement.
    Modifies products in place.

    Args:
        products: List of ProductInfo objects with page_range set
        pages_content: Dict mapping page_num -> lowercased page text
        logger: Optional logger for output
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    classifier = PageContentClassifier()

    logger.info("   🔍 PHASE 2: Refining page ranges with content classification...")

    refined_count = 0

    for product in products:
        if not product.page_range:
            continue

        original_start = min(product.page_range)
        original_end = max(product.page_range)

        # Classify pages in range
        classifications = classifier.classify_page_range(
            pages_content=pages_content,
            start_page=original_start,
            end_page=original_end,
            product_name=product.name
        )

        # Refine range
        new_start, new_end, reason = classifier.refine_page_range(
            classifications=classifications,
            original_start=original_start,
            original_end=original_end
        )

        # Update if changed (either start OR end)
        if new_start != original_start or new_end != original_end:
            old_range = f"{original_start}-{original_end}"
            new_range = f"{new_start}-{new_end}"
            product.page_range = list(range(new_start, new_end + 1))
            logger.info(f"      ✂️ {product.name}: {old_range} → {new_range} ({reason})")
            refined_count += 1
        else:
            logger.debug(f"      ✓ {product.name}: pages {original_start}-{original_end} ({reason})")

    if refined_count > 0:
        logger.info(f"   ✅ Refined {refined_count}/{len(products)} product page ranges")
    else:
        logger.info("   ✓ No page ranges needed refinement")
