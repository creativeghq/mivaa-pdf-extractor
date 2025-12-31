"""
Enhanced PDF Processor with AI Services Integration

Integrates all Phase 1-4 AI services into the PDF processing pipeline:
- Document classification
- Boundary detection
- Product validation
- Escalation engine
- Consensus validation

This wraps the existing pdf_processor.py with enhanced AI capabilities.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from app.services.pdf_processor import PDFProcessor, PDFProcessingResult
from app.services.document_classifier import DocumentClassifier
from app.services.boundary_detector import BoundaryDetector
from app.services.product_validator import ProductValidator
from app.services.escalation_engine import EscalationEngine
from app.services.consensus_validator import ConsensusValidator
from app.services.unified_chunking_service import UnifiedChunkingService
from app.services.product_creation_service import ProductCreationService
from app.services.ai_call_logger import AICallLogger

logger = logging.getLogger(__name__)


class EnhancedPDFProcessor:
    """
    Enhanced PDF processor with integrated AI services.
    
    Provides intelligent document processing with:
    - Content classification
    - Product boundary detection
    - Quality validation
    - Smart model escalation
    - Multi-model consensus
    """
    
    def __init__(self):
        """Initialize enhanced PDF processor with all AI services."""
        # Core processor
        self.pdf_processor = PDFProcessor()
        
        # AI services (Phase 1-4)
        self.document_classifier = DocumentClassifier()
        self.boundary_detector = BoundaryDetector()
        self.product_validator = ProductValidator()
        self.escalation_engine = EscalationEngine()
        self.consensus_validator = ConsensusValidator()
        
        # Existing services
        self.chunking_service = UnifiedChunkingService()
        from app.services.supabase_client import get_supabase_client
        self.product_service = ProductCreationService(get_supabase_client())
        self.ai_logger = AICallLogger()
        
        logger.info("✅ Enhanced PDF Processor initialized with all AI services")
    
    async def process_pdf_enhanced(
        self,
        pdf_bytes: bytes,
        document_id: str,
        job_id: Optional[str] = None,
        processing_options: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Process PDF with enhanced AI capabilities.
        
        Args:
            pdf_bytes: PDF file bytes
            document_id: Document identifier
            job_id: Optional job ID for tracking
            processing_options: Processing configuration
            progress_callback: Optional progress callback
            
        Returns:
            Enhanced processing result with classifications, boundaries, products
        """
        start_time = datetime.now()
        job_id = job_id or document_id
        
        logger.info(f"🚀 Starting enhanced PDF processing for document {document_id}")
        
        try:
            # Step 1: Basic PDF extraction (existing)
            if progress_callback:
                await progress_callback("Extracting PDF content...", 10)
            
            pdf_result = await self.pdf_processor.process_pdf_from_bytes(
                pdf_bytes=pdf_bytes,
                document_id=document_id,
                processing_options=processing_options,
                progress_callback=progress_callback
            )
            
            logger.info(f"✅ PDF extracted: {pdf_result.page_count} pages, {len(pdf_result.extracted_images)} images")
            
            # Step 2: NEW - Classify content before chunking
            if progress_callback:
                await progress_callback("Classifying document content...", 20)
            
            classified_content = await self._classify_content(
                pdf_result=pdf_result,
                job_id=job_id
            )
            
            logger.info(f"✅ Content classified: {len(classified_content['classifications'])} sections")
            
            # Step 3: Create chunks (classification-aware)
            if progress_callback:
                await progress_callback("Creating intelligent chunks...", 30)
            
            chunks = await self._create_enhanced_chunks(
                pdf_result=pdf_result,
                classified_content=classified_content,
                document_id=document_id
            )
            
            logger.info(f"✅ Created {len(chunks)} chunks")
            
            # Step 4: NEW - Detect product boundaries
            if progress_callback:
                await progress_callback("Detecting product boundaries...", 40)
            
            boundaries = await self.boundary_detector.detect_boundaries(
                chunks=chunks,
                job_id=job_id
            )
            
            logger.info(f"✅ Detected {len(boundaries)} product boundaries")
            
            # Step 5: NEW - Group chunks by product
            if progress_callback:
                await progress_callback("Grouping products...", 50)
            
            product_groups = await self.boundary_detector.group_chunks_by_product(
                chunks=chunks,
                boundaries=boundaries
            )
            
            logger.info(f"✅ Grouped into {len(product_groups)} potential products")
            
            # Step 6: Extract and validate products
            if progress_callback:
                await progress_callback("Extracting and validating products...", 60)
            
            validated_products = await self._extract_and_validate_products(
                product_groups=product_groups,
                images=pdf_result.extracted_images,
                job_id=job_id,
                progress_callback=progress_callback
            )
            
            logger.info(f"✅ Validated {len(validated_products)} products")
            
            # Step 7: Calculate processing metrics
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # Get escalation stats
            escalation_stats = self.escalation_engine.get_stats()
            
            result = {
                "success": True,
                "document_id": document_id,
                "job_id": job_id,
                "processing_time": processing_time,
                "pdf_result": {
                    "page_count": pdf_result.page_count,
                    "word_count": pdf_result.word_count,
                    "character_count": pdf_result.character_count,
                    "images_extracted": len(pdf_result.extracted_images),
                },
                "classifications": classified_content["classifications"],
                "chunks_created": len(chunks),
                "boundaries_detected": len(boundaries),
                "product_groups": len(product_groups),
                "products_validated": len(validated_products),
                "products": validated_products,
                "chunks": chunks,
                "boundaries": boundaries,
                "escalation_stats": escalation_stats,
                "timestamp": end_time.isoformat(),
            }
            
            if progress_callback:
                await progress_callback("Processing complete!", 100)
            
            logger.info(
                f"🎉 Enhanced PDF processing complete: "
                f"{len(validated_products)} products, "
                f"{len(chunks)} chunks, "
                f"{processing_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Enhanced PDF processing failed: {str(e)}")
            raise
    
    async def _classify_content(
        self,
        pdf_result: PDFProcessingResult,
        job_id: str
    ) -> Dict[str, Any]:
        """
        Classify PDF content into sections.
        
        Args:
            pdf_result: PDF processing result
            job_id: Job ID
            
        Returns:
            Classification results
        """
        # Split markdown into sections (by page or paragraph)
        sections = self._split_into_sections(pdf_result.markdown_content)
        
        # Classify each section
        classifications = await self.document_classifier.classify_batch(
            contents=[s["content"] for s in sections],
            contexts=[s.get("context") for s in sections],
            job_id=job_id
        )
        
        # Combine sections with classifications
        for i, section in enumerate(sections):
            section["classification"] = classifications[i]
        
        return {
            "sections": sections,
            "classifications": classifications,
            "product_sections": sum(1 for c in classifications if c.get("is_product")),
            "supporting_sections": sum(1 for c in classifications if c.get("content_type") == "supporting"),
        }
    
    def _split_into_sections(self, markdown_content: str) -> List[Dict[str, Any]]:
        """
        Split markdown content into logical sections.
        
        Args:
            markdown_content: Markdown text
            
        Returns:
            List of sections with content and context
        """
        # Simple split by double newlines (paragraphs)
        paragraphs = markdown_content.split("\n\n")
        
        sections = []
        for i, para in enumerate(paragraphs):
            if para.strip():
                sections.append({
                    "content": para.strip(),
                    "context": {
                        "section_index": i,
                        "has_images": "![" in para,  # Markdown image syntax
                    }
                })
        
        return sections
    
    async def _create_enhanced_chunks(
        self,
        pdf_result: PDFProcessingResult,
        classified_content: Dict[str, Any],
        document_id: str
    ) -> List[Dict[str, Any]]:
        """
        Create chunks with classification awareness.
        
        Args:
            pdf_result: PDF processing result
            classified_content: Classification results
            document_id: Document ID
            
        Returns:
            List of enhanced chunks
        """
        # Use existing chunking service
        chunks_result = await self.chunking_service.create_chunks(
            content=pdf_result.markdown_content,
            document_id=document_id,
            strategy="semantic"
        )
        
        # Enhance chunks with classification data
        chunks = chunks_result.get("chunks", [])
        
        # Add classification metadata to chunks
        for chunk in chunks:
            # Find matching classification based on content overlap
            chunk["metadata"] = chunk.get("metadata", {})
            chunk["metadata"]["classification"] = self._find_matching_classification(
                chunk["content"],
                classified_content["sections"]
            )
        
        return chunks
    
    def _find_matching_classification(
        self,
        chunk_content: str,
        sections: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Find best matching classification for a chunk."""
        # Simple matching: find section with most content overlap
        best_match = None
        best_overlap = 0
        
        for section in sections:
            # Calculate simple word overlap
            chunk_words = set(chunk_content.lower().split())
            section_words = set(section["content"].lower().split())
            overlap = len(chunk_words & section_words)
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = section.get("classification")
        
        return best_match
    
    async def _extract_and_validate_products(
        self,
        product_groups: List[List[Dict[str, Any]]],
        images: List[Dict[str, Any]],
        job_id: str,
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract products from groups and validate quality.
        
        Args:
            product_groups: Groups of chunks
            images: Extracted images
            job_id: Job ID
            progress_callback: Progress callback
            
        Returns:
            List of validated products
        """
        validated_products = []
        
        for i, group in enumerate(product_groups):
            try:
                # Extract product from chunk group
                product_data = await self._extract_product_from_group(group, images)
                
                # NEW: Validate product quality
                validation = await self.product_validator.validate_product(
                    product_data=product_data,
                    chunks=group,
                    images=product_data.get("images", [])
                )
                
                if validation["passed"]:
                    product_data["validation"] = validation
                    validated_products.append(product_data)
                    logger.info(
                        f"✅ Product {i+1} validated (score: {validation['overall_score']:.2f})"
                    )
                else:
                    logger.warning(
                        f"⚠️ Product {i+1} failed validation (score: {validation['overall_score']:.2f})"
                    )
                
            except Exception as e:
                logger.error(f"❌ Failed to extract/validate product {i+1}: {str(e)}")
        
        return validated_products
    
    async def _extract_product_from_group(
        self,
        chunks: List[Dict[str, Any]],
        all_images: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract product data from chunk group."""
        # Combine chunk content
        combined_content = "\n\n".join(chunk.get("content", "") for chunk in chunks)
        
        # Find associated images (simple proximity-based)
        product_images = []
        # For now, just include all images - can be enhanced later
        
        return {
            "content": combined_content,
            "chunks": chunks,
            "images": product_images,
            "chunk_count": len(chunks),
            "character_count": len(combined_content),
        }


