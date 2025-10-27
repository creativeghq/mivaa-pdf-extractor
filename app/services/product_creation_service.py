"""
Product Creation Service

This service automatically creates products from processed PDF chunks.
It analyzes chunks to identify product-like content and creates product records
with proper metadata, embeddings, and relationships.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import time

logger = logging.getLogger(__name__)


class ProductCreationService:
    """Service for creating products from PDF chunks"""

    def __init__(self, supabase_client):
        """
        Initialize product creation service.
        
        Args:
            supabase_client: Supabase client instance for database operations
        """
        self.supabase = supabase_client
        self.logger = logging.getLogger(__name__)

    async def create_products_from_layout_candidates(
        self,
        document_id: str,
        workspace_id: str = "ffafc28b-1b8b-4b0d-b226-9f9a6154004e",
        min_confidence: float = 0.5,
        min_quality_score: float = 0.5
    ) -> Dict[str, Any]:
        """
        ✅ NEW: Create products from layout-based product candidates.
        This method uses the enhanced htmlDOMAnalyzer to detect product candidates
        before chunking, filtering out index pages, sustainability content, etc.

        Args:
            document_id: UUID of the processed document
            workspace_id: UUID of the workspace
            min_confidence: Minimum confidence score for product candidates
            min_quality_score: Minimum quality score for product candidates

        Returns:
            Dictionary with creation statistics
        """
        try:
            self.logger.info(f"🎯 Starting layout-based product creation for document {document_id}")

            # Get layout analysis results with product candidates
            layout_analysis = await self._get_layout_analysis_results(document_id)

            if not layout_analysis or not layout_analysis.get('product_candidates'):
                self.logger.warning(f"No layout analysis or product candidates found for document {document_id}")
                # Fallback to original method - NO LIMIT to find all products
                return await self.create_products_from_chunks(document_id, workspace_id, max_products=None)

            product_candidates = layout_analysis['product_candidates']
            self.logger.info(f"🔍 Found {len(product_candidates)} product candidates from layout analysis")

            # Filter candidates by confidence and quality
            high_quality_candidates = [
                candidate for candidate in product_candidates
                if (candidate.get('confidence', 0) >= min_confidence and
                    candidate.get('qualityScore', 0) >= min_quality_score and
                    candidate.get('contentType') == 'product')
            ]

            self.logger.info(f"✅ {len(high_quality_candidates)} candidates meet quality thresholds")

            if not high_quality_candidates:
                self.logger.warning("No high-quality product candidates found, falling back to chunk-based creation")
                return await self.create_products_from_chunks(document_id, workspace_id, max_products=None)

            # Create products from high-quality candidates
            products_created = 0
            products_failed = 0

            for i, candidate in enumerate(high_quality_candidates):
                try:
                    # Find associated chunks for this candidate
                    associated_chunks = await self._find_chunks_for_candidate(document_id, candidate)

                    if not associated_chunks:
                        self.logger.warning(f"No chunks found for candidate {candidate.get('name', 'Unknown')}")
                        continue

                    # Create product from candidate and associated chunks
                    product_data = self._create_product_from_candidate(
                        candidate=candidate,
                        chunks=associated_chunks,
                        document_id=document_id,
                        workspace_id=workspace_id,
                        index=i
                    )

                    # Insert product into database
                    product_response = self.supabase.client.table('products').insert(product_data).execute()

                    if product_response.data:
                        products_created += 1
                        product_id = product_response.data[0]['id']
                        self.logger.info(f"✅ Created product {i+1}/{len(high_quality_candidates)}: {product_id} - {candidate.get('name', 'Unknown')}")

                        # ✅ ENHANCED: Enrich product with real image data
                        try:
                            from .product_enrichment_service import ProductEnrichmentService
                            enrichment_service = ProductEnrichmentService(self.supabase)

                            enrichment_result = await enrichment_service.enrich_product(
                                product_id=product_id,
                                product_data=product_data,
                                document_id=document_id,
                                workspace_id=workspace_id
                            )

                            if enrichment_result.get('success'):
                                self.logger.info(f"✅ Product enrichment complete: {product_id}")
                            else:
                                self.logger.warning(f"⚠️ Product enrichment skipped: {enrichment_result.get('error')}")
                        except Exception as e:
                            self.logger.warning(f"⚠️ Product enrichment failed: {e}")
                    else:
                        products_failed += 1
                        self.logger.warning(f"⚠️ Failed to create product from candidate {candidate.get('name', 'Unknown')}")

                except Exception as e:
                    products_failed += 1
                    self.logger.error(f"❌ Error creating product from candidate {candidate.get('name', 'Unknown')}: {e}")
                    continue

            return {
                "success": True,
                "products_created": products_created,
                "products_failed": products_failed,
                "candidates_processed": len(high_quality_candidates),
                "total_candidates_found": len(product_candidates),
                "method": "layout_based_detection",
                "message": f"Created {products_created} products from {len(high_quality_candidates)} layout-detected candidates"
            }

        except Exception as e:
            self.logger.error(f"Failed to create products from layout candidates: {e}")
            # Fallback to original method
            self.logger.info("Falling back to chunk-based product creation")
            return await self.create_products_from_chunks(document_id, workspace_id, max_products=None)

    async def create_products_from_chunks(
        self,
        document_id: str,
        workspace_id: str = "ffafc28b-1b8b-4b0d-b226-9f9a6154004e",
        max_products: Optional[int] = None,
        min_chunk_length: int = 100,
        enrich_products: bool = True
    ) -> Dict[str, Any]:
        """
        ✅ ENHANCED: Create products from document chunks with two-stage classification.

        Stage 1: Fast text-only classification using Claude Haiku for initial filtering
        Stage 2: Deep enrichment using Claude Sonnet for confirmed products

        Args:
            document_id: UUID of the processed document
            workspace_id: UUID of the workspace
            max_products: Maximum number of products to create (None = unlimited)
            min_chunk_length: Minimum chunk content length to consider for products

        Returns:
            Dictionary with creation statistics including timing metrics
        """
        try:
            self.logger.info(f"🏭 Starting two-stage product creation for document {document_id}")

            # Fetch all chunks for this document
            chunks_response = self.supabase.client.table('document_chunks').select('*').eq('document_id', document_id).order('chunk_index').execute()

            if not chunks_response.data:
                self.logger.warning(f"No chunks found for document {document_id}")
                return {
                    "success": True,
                    "products_created": 0,
                    "chunks_processed": 0,
                    "message": "No chunks found to create products from"
                }

            chunks = chunks_response.data
            self.logger.info(f"📦 Found {len(chunks)} chunks to process")

            # Filter chunks by minimum length
            eligible_chunks = [
                chunk for chunk in chunks
                if len(chunk.get('content', '')) >= min_chunk_length
            ]

            self.logger.info(f"📋 {len(eligible_chunks)} chunks meet minimum length requirement ({min_chunk_length} chars)")

            # ✅ NEW: Stage 1 - Fast Classification with Claude Haiku
            self.logger.info(f"🚀 Stage 1: Fast classification with Claude Haiku...")
            stage1_start = time.time()

            product_candidates = await self._stage1_fast_classification(eligible_chunks)

            stage1_time = time.time() - stage1_start
            self.logger.info(f"⚡ Stage 1 completed in {stage1_time:.2f}s: {len(product_candidates)} candidates from {len(eligible_chunks)} chunks")

            # ✅ NEW: Deduplicate products by name BEFORE Stage 2
            if product_candidates:
                product_candidates = self._deduplicate_product_chunks(product_candidates)
                self.logger.info(f"🔄 After deduplication: {len(product_candidates)} unique products")

            # ✅ NEW: Stage 2 - Deep Enrichment with Claude Sonnet
            self.logger.info(f"🎯 Stage 2: Deep enrichment with Claude Sonnet...")
            stage2_start = time.time()

            products_created = 0
            products_failed = 0
            chunks_processed = 0

            for i, candidate in enumerate(product_candidates):
                try:
                    # Stage 2: Deep enrichment and validation
                    enriched_product = await self._stage2_deep_enrichment(
                        candidate=candidate,
                        document_id=document_id,
                        workspace_id=workspace_id,
                        index=i
                    )

                    if enriched_product:
                        # Insert product into database
                        insert_response = self.supabase.client.table('products').insert(enriched_product).execute()

                        if insert_response.data:
                            products_created += 1
                            chunks_processed += 1
                            product_id = insert_response.data[0]['id']
                            self.logger.info(f"✅ Created enriched product {products_created}: {enriched_product['name']}")

                            # ✅ ENHANCED: Enrich product with real image data
                            if enrich_products:
                                try:
                                    from .product_enrichment_service import ProductEnrichmentService
                                    enrichment_service = ProductEnrichmentService(self.supabase)

                                    enrichment_result = await enrichment_service.enrich_product(
                                        product_id=product_id,
                                        product_data=enriched_product,
                                        document_id=document_id,
                                        workspace_id=workspace_id
                                    )

                                    if enrichment_result.get('success'):
                                        self.logger.info(f"✅ Product enrichment complete: {product_id}")
                                    else:
                                        self.logger.warning(f"⚠️ Product enrichment skipped: {enrichment_result.get('error')}")
                                except Exception as e:
                                    self.logger.warning(f"⚠️ Product enrichment failed: {e}")
                                    # Continue without enrichment - product is still created
                        else:
                            products_failed += 1
                            self.logger.error(f"❌ Failed to insert enriched product for candidate {i+1}")
                    else:
                        products_failed += 1
                        self.logger.warning(f"⚠️ Stage 2 rejected candidate {i+1}")

                except Exception as e:
                    products_failed += 1
                    self.logger.error(f"❌ Error in stage 2 for candidate {i+1}: {str(e)}")
                    continue

            stage2_time = time.time() - stage2_start
            total_time = stage1_time + stage2_time

            self.logger.info(f"🎉 Two-stage creation completed in {total_time:.2f}s:")
            self.logger.info(f"   Stage 1 (Haiku): {stage1_time:.2f}s, {len(product_candidates)} candidates")
            self.logger.info(f"   Stage 2 (Sonnet): {stage2_time:.2f}s, {products_created} products created")
            self.logger.info(f"   Performance: {products_created} created, {products_failed} failed")

            return {
                "success": True,
                "products_created": products_created,
                "products_failed": products_failed,
                "chunks_processed": chunks_processed,
                "total_chunks": len(chunks),
                "eligible_chunks": len(eligible_chunks),
                "stage1_candidates": len(product_candidates),
                "stage1_time": stage1_time,
                "stage2_time": stage2_time,
                "total_time": total_time,
                "message": f"Two-stage creation: {products_created} products from {len(product_candidates)} candidates in {total_time:.2f}s"
            }

        except Exception as e:
            self.logger.error(f"❌ Error in create_products_from_chunks: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "products_created": 0,
                "chunks_processed": 0
            }

    def _is_valid_product_chunk(self, chunk: Dict[str, Any]) -> bool:
        """
        ✅ NEW: Validate if a chunk contains actual product content.
        Filters out index pages, sustainability info, certifications, technical tables, etc.
        """
        content = chunk.get('content', '').lower()

        # Skip very short content
        if len(content) < 100:
            return False

        # Skip index/table of contents
        if any(keyword in content for keyword in [
            'table of contents', 'index', 'contents', 'page numbers',
            'signature book', 'signature index', 'collections index'
        ]):
            self.logger.debug("Skipping: Index/TOC content")
            return False

        # Skip sustainability content
        if any(keyword in content for keyword in [
            'sustainability', 'environmental', 'eco-friendly', 'carbon footprint',
            'recycled', 'leed', 'greenguard', 'environmental performance',
            'iso 14001', 'environmental management'
        ]) and not any(product_keyword in content for product_keyword in [
            'dimensions', 'designer', 'collection', '×', 'cm', 'mm'
        ]):
            self.logger.debug("Skipping: Sustainability content")
            return False

        # Skip certification content
        if any(keyword in content for keyword in [
            'quality certifications', 'sustainability certifications',
            'iso 9001', 'une-en iso', 'certification', 'certifies',
            'quality management system'
        ]) and not any(product_keyword in content for product_keyword in [
            'dimensions', 'designer', 'collection', '×', 'cm', 'mm'
        ]):
            self.logger.debug("Skipping: Certification content")
            return False

        # Skip technical specifications tables
        if any(keyword in content for keyword in [
            'technical characteristics', 'technical data', 'specifications',
            'fire rating', 'weight per', 'thickness', 'water absorption',
            'breaking strength', 'thermal expansion'
        ]) and ('|' in content or content.count('\n') > 20):
            self.logger.debug("Skipping: Technical specifications table")
            return False

        # ✅ ENHANCED: Skip moodboard content (unless it has strong product indicators)
        if any(keyword in content.lower() for keyword in [
            'moodboard', 'mood board', 'inspiration', 'fresh inspiration', 'signature moodboard'
        ]) and not any(product_keyword in content for product_keyword in [
            'dimensions', 'designer', '×', 'cm', 'mm'
        ]):
            self.logger.debug("Skipping: Moodboard content")
            return False

        # ✅ NEW: Skip cleaning/maintenance content
        cleaning_keywords = [
            'cleaning', 'cleaner', 'maintenance', 'fila', 'faber', 'remover',
            'degreaser', 'floor cleaner', 'tile cleaner', 'epoxy pro',
            'post-construction', 'application guide', 'cleaning system'
        ]
        if any(keyword in content.lower() for keyword in cleaning_keywords):
            # Only skip if it doesn't have strong product indicators
            if not any(pattern in content for pattern in ['×', 'cm', 'mm']) or \
               'not applicable' in content.lower() or \
               'guidance documentation' in content.lower():
                self.logger.debug("Skipping: Cleaning/maintenance content")
                return False

        # ✅ NEW: Skip generic descriptive content
        generic_keywords = [
            'artisan clay', 'mediterranean sand', 'deep contrast',
            'not specified', 'not applicable'
        ]
        if any(keyword in content.lower() for keyword in generic_keywords) and \
           len(content) < 200:  # Short generic descriptions
            self.logger.debug("Skipping: Generic descriptive content")
            return False

        # ✅ NEW: Skip designer biographies - CRITICAL FIX
        designer_bio_keywords = [
            'biography', 'born in', 'graduated from', 'studied at',
            'career began', 'founded in', 'established in',
            'renowned designer', 'award-winning', 'based in',
            'studio was founded', 'design philosophy', 'creative director',
            'years of experience', 'portfolio includes', 'education',
            'professional background', 'design journey', 'trained at'
        ]
        if any(keyword in content.lower() for keyword in designer_bio_keywords):
            self.logger.debug("Skipping: Designer biography content")
            return False

        # ✅ NEW: Skip factory/manufacturing details - CRITICAL FIX
        factory_keywords = [
            'factory location', 'manufacturing facility', 'production capacity',
            'plant location', 'headquarters', 'production site',
            'manufacturing process', 'quality control', 'production line',
            'factory address', 'production facility', 'manufacturing plant',
            'industrial complex', 'production area', 'manufacturing site'
        ]
        if any(keyword in content.lower() for keyword in factory_keywords):
            self.logger.debug("Skipping: Factory/manufacturing details")
            return False

        # Require product indicators for valid products
        has_uppercase_name = any(word.isupper() and len(word) > 2 for word in content.split())
        has_dimensions = any(pattern in content for pattern in ['×', 'x ', 'cm', 'mm'])
        has_product_context = any(keyword in content.lower() for keyword in [
            'designer', 'collection', 'material', 'ceramic', 'porcelain', 'tile',
            'estudi{h}ac', 'dsignio', 'alt design', 'mut', 'yonoh', 'stacy garcia'
        ])

        # ✅ NEW: Skip technical specs without product name - CRITICAL FIX
        has_technical_specs = any(keyword in content.lower() for keyword in [
            'water absorption', 'breaking strength', 'slip resistance',
            'frost resistance', 'chemical resistance', 'thermal shock',
            'modulus of rupture', 'abrasion resistance', 'stain resistance'
        ])
        if has_technical_specs and not has_uppercase_name:
            self.logger.debug("Skipping: Technical specifications without product name")
            return False

        # ✅ ENHANCED: Require ALL 3 indicators for high confidence (was 2 of 3)
        product_score = sum([has_uppercase_name, has_dimensions, has_product_context])

        if product_score >= 3:  # ✅ CHANGED: Was 2, now 3 (stricter)
            self.logger.debug(f"Valid product chunk: uppercase={has_uppercase_name}, dimensions={has_dimensions}, context={has_product_context}")
            return True

        self.logger.debug(f"Skipping: Insufficient product indicators (score: {product_score}/3, need 3)")
        return False

    def _extract_product_name(self, content: str) -> Optional[str]:
        """
        ✅ NEW: Extract actual product name from content.
        Looks for UPPERCASE product names like VALENOVA, PIQUÉ, ONA, etc.
        """
        import re

        lines = content.split('\n')

        # Look for product names in headers (## PRODUCT_NAME)
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()

            # Header pattern: ## PRODUCT_NAME or # PRODUCT_NAME
            header_match = re.search(r'^#+\s+([A-Z]{2,}(?:\s+[A-Z]{2,})*)', line)
            if header_match:
                return header_match.group(1).strip()

            # Standalone UPPERCASE line
            if re.match(r'^[A-Z]{2,}(?:\s+[A-Z]{2,})*$', line) and len(line) <= 20:
                return line.strip()

            # UPPERCASE word followed by dimensions or designer
            uppercase_match = re.search(r'\b([A-Z]{3,}(?:\s+[A-Z]{3,})*)\b', line)
            if uppercase_match:
                candidate = uppercase_match.group(1).strip()
                # Verify it's followed by product context in next few lines
                next_lines = '\n'.join(lines[lines.index(line):lines.index(line)+3])
                if any(pattern in next_lines.lower() for pattern in [
                    '×', 'cm', 'mm', 'designer', 'estudi', 'dsignio', 'alt design', 'mut', 'yonoh'
                ]):
                    return candidate

        # Look for UPPERCASE words in the content
        uppercase_words = re.findall(r'\b[A-Z]{3,}\b', content)

        # Filter out common non-product words
        excluded_words = {
            'THE', 'AND', 'FOR', 'WITH', 'FROM', 'THAT', 'THIS', 'HAVE', 'BEEN',
            'WILL', 'THEY', 'WERE', 'SAID', 'EACH', 'WHICH', 'THEIR', 'TIME',
            'HARMONY', 'COLLECTION', 'DESIGN', 'CERAMIC', 'PORCELAIN', 'TILE',
            'TECHNICAL', 'SPECIFICATIONS', 'CHARACTERISTICS', 'QUALITY'
        }

        for word in uppercase_words:
            if word not in excluded_words and len(word) >= 3:
                # Check if this word appears near product context
                word_context = content[max(0, content.find(word)-100):content.find(word)+100]
                if any(pattern in word_context.lower() for pattern in [
                    '×', 'cm', 'mm', 'designer', 'collection'
                ]):
                    return word

        return None

    def _extract_product_metadata(self, content: str) -> Dict[str, Any]:
        """
        ✅ NEW: Extract product metadata like dimensions, designer, colors, etc.
        """
        import re
        metadata = {}

        # Extract dimensions (e.g., "15×38", "20×40", "11.8×11.8")
        dimension_patterns = [
            r'(\d+(?:\.\d+)?)\s*[×x]\s*(\d+(?:\.\d+)?)\s*(?:cm|mm)?',
            r'(\d+(?:\.\d+)?)\s*[×x]\s*(\d+(?:\.\d+)?)\s*[×x]\s*(\d+(?:\.\d+)?)\s*(?:cm|mm)?'
        ]

        for pattern in dimension_patterns:
            matches = re.findall(pattern, content)
            if matches:
                if len(matches[0]) == 2:  # 2D dimensions
                    metadata['dimensions'] = f"{matches[0][0]}×{matches[0][1]}"
                elif len(matches[0]) == 3:  # 3D dimensions
                    metadata['dimensions'] = f"{matches[0][0]}×{matches[0][1]}×{matches[0][2]}"
                break

        # Extract designer/studio
        designer_patterns = [
            r'(?:by|BY|designer|DESIGNER|studio|STUDIO)\s+([A-Z][A-Za-z\s{}\-]+)',
            r'(ESTUDI\{H\}AC|DSIGNIO|ALT DESIGN|MUT|YONOH|STACY GARCIA|SG NY)',
            r'([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:NY|STUDIO|DESIGN)'
        ]

        for pattern in designer_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                designer = matches[0].strip()
                if len(designer) > 2 and designer not in ['THE', 'AND', 'FOR']:
                    metadata['designer'] = designer
                    break

        # Extract colors (UPPERCASE color names)
        color_patterns = [
            r'\b(TAUPE|SAND|CLAY|WHITE|BLACK|GREY|GRAY|ANTHRACITE|BEIGE|BROWN|BLUE|GREEN|RED)\b'
        ]

        colors = []
        for pattern in color_patterns:
            matches = re.findall(pattern, content)
            colors.extend(matches)

        if colors:
            metadata['colors'] = list(set(colors))  # Remove duplicates

        # Extract collection name
        collection_matches = re.findall(r'(?:collection|COLLECTION)\s+([A-Z][a-z]+)', content)
        if collection_matches:
            metadata['collection'] = collection_matches[0]

        # Extract material type
        material_patterns = [
            r'\b(ceramic|porcelain|tile|stone|marble|granite)\b'
        ]

        for pattern in material_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                metadata['material_type'] = matches[0].lower()
                break

        return metadata

    def _create_product_from_chunk(
        self,
        chunk: Dict[str, Any],
        document_id: str,
        workspace_id: str,
        index: int
    ) -> Dict[str, Any]:
        """
        Create a product record from a chunk.
        
        Args:
            chunk: Chunk data from database
            document_id: UUID of source document
            workspace_id: UUID of workspace
            index: Index of chunk in processing order
            
        Returns:
            Product data dictionary ready for insertion
        """
        content = chunk.get('content', '')
        chunk_id = chunk.get('id')
        chunk_index = chunk.get('chunk_index', index)
        page_number = chunk.get('page_number')
        
        # ✅ NEW: Extract actual product name from content
        product_name = self._extract_product_name(content)
        if not product_name:
            # Fallback to generic name
            product_name = f"Product from Chunk {chunk_index}"

        # ✅ NEW: Extract product metadata (dimensions, designer, etc.)
        product_metadata = self._extract_product_metadata(content)

        # Use first 200 characters as description
        description = content[:200].strip()
        if len(content) > 200:
            description += "..."

        # Use full content as long description (up to 1000 chars)
        long_description = content[:1000].strip()
        if len(content) > 1000:
            long_description += "..."
        
        # Build product data
        product_data = {
            "name": product_name,
            "description": description,
            "long_description": long_description,
            "source_document_id": document_id,
            "source_chunks": [chunk_id],  # Array of chunk IDs
            "properties": {
                "source_chunk_id": chunk_id,
                "document_id": document_id,
                "chunk_index": chunk_index,
                "page_number": page_number,
                "content_length": len(content),
                "auto_generated": True,
                "generation_timestamp": datetime.utcnow().isoformat()
            },
            "metadata": {
                "extracted_from": "pdf_chunk",
                "chunk_metadata": chunk.get('metadata', {}),
                "extraction_date": datetime.utcnow().isoformat(),
                "auto_created": True,
                "workspace_id": workspace_id,
                **product_metadata  # ✅ NEW: Include extracted product metadata
            },
            "status": "draft",
            "created_from_type": "pdf_processing",
            # Note: embedding will be generated separately if needed
        }
        
        return product_data

    async def get_product_creation_stats(self, document_id: str) -> Dict[str, Any]:
        """
        Get statistics about products created from a document.
        
        Args:
            document_id: UUID of the document
            
        Returns:
            Statistics dictionary
        """
        try:
            # Count products created from this document
            products_response = self.supabase.client.table('products').select('id, name, status, created_at').eq('source_document_id', document_id).execute()
            
            products = products_response.data or []
            
            return {
                "success": True,
                "document_id": document_id,
                "total_products": len(products),
                "products_by_status": self._count_by_status(products),
                "products": products
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get product stats: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _count_by_status(self, products: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count products by status"""
        counts = {}
        for product in products:
            status = product.get('status', 'unknown')
            counts[status] = counts.get(status, 0) + 1
        return counts

    async def _get_layout_analysis_results(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        ✅ NEW: Get markdown-based product candidates analysis.
        Instead of HTML conversion, analyze markdown content directly for product patterns.
        """
        try:
            # First try to get existing layout analysis
            response = self.supabase.client.table('document_layout_analysis').select('analysis_metadata').eq('document_id', document_id).execute()

            if response.data and len(response.data) > 0:
                analysis_metadata = response.data[0].get('analysis_metadata', {})
                if analysis_metadata.get('product_candidates'):
                    return analysis_metadata

            # If no layout analysis exists, create it from markdown content
            self.logger.info(f"No layout analysis found, creating from markdown content for document {document_id}")
            return await self._analyze_markdown_for_products(document_id)

        except Exception as e:
            self.logger.error(f"Failed to get layout analysis results: {e}")
            return None

    async def _analyze_markdown_for_products(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        ✅ NEW: Analyze markdown content directly for product candidates.
        This is more effective than HTML conversion since PyMuPDF4LLM markdown
        preserves the structure we need for product detection.
        """
        try:
            # Get the original markdown content from the document
            doc_response = self.supabase.client.table('documents').select('content').eq('id', document_id).execute()

            if not doc_response.data:
                self.logger.warning(f"No document content found for {document_id}")
                return None

            markdown_content = doc_response.data[0].get('content', '')
            if not markdown_content:
                self.logger.warning(f"Empty markdown content for document {document_id}")
                return None

            # Analyze markdown for product patterns
            product_candidates = self._detect_products_in_markdown(markdown_content)

            # Store the analysis results
            analysis_metadata = {
                'product_candidates': product_candidates,
                'total_candidates': len(product_candidates),
                'high_quality_candidates': len([c for c in product_candidates if c.get('qualityScore', 0) > 0.7]),
                'analysis_timestamp': datetime.utcnow().isoformat(),
                'analysis_method': 'markdown_pattern_detection',
                'layout_analysis_version': '1.1.0'
            }

            # Store in database
            await self._store_markdown_analysis(document_id, analysis_metadata)

            return analysis_metadata

        except Exception as e:
            self.logger.error(f"Failed to analyze markdown for products: {e}")
            return None

    async def _find_chunks_for_candidate(self, document_id: str, candidate: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        ✅ NEW: Find document chunks associated with a product candidate.
        Uses page number and bounding box information to find relevant chunks.
        """
        try:
            page_number = candidate.get('pageNumber', 1)

            # Get chunks from the same page
            chunks_response = self.supabase.client.table('document_chunks').select('*').eq('document_id', document_id).eq('page_number', page_number).execute()

            if not chunks_response.data:
                # Fallback: get chunks from nearby pages
                chunks_response = self.supabase.client.table('document_chunks').select('*').eq('document_id', document_id).execute()

            chunks = chunks_response.data or []

            # If we have bounding box info, filter by proximity
            if candidate.get('boundingBox') and chunks:
                # For now, return the first chunk from the same page
                # TODO: Implement spatial proximity filtering
                return chunks[:1]

            return chunks[:1] if chunks else []

        except Exception as e:
            self.logger.error(f"Failed to find chunks for candidate: {e}")
            return []

    def _create_product_from_candidate(
        self,
        candidate: Dict[str, Any],
        chunks: List[Dict[str, Any]],
        document_id: str,
        workspace_id: str,
        index: int
    ) -> Dict[str, Any]:
        """
        ✅ NEW: Create a product record from a layout-detected candidate and associated chunks.
        """
        extracted_data = candidate.get('extractedData', {})
        patterns = candidate.get('patterns', {})

        # Use extracted product name or generate one
        product_name = extracted_data.get('productName') or candidate.get('name', f"Product {index + 1}")

        # Build description from chunks
        chunk_contents = [chunk.get('content', '') for chunk in chunks]
        combined_content = ' '.join(chunk_contents)

        # Use first 200 characters as description
        description = combined_content[:200].strip()
        if len(combined_content) > 200:
            description += "..."

        # Use full content as long description (up to 1000 chars)
        long_description = combined_content[:1000].strip()
        if len(combined_content) > 1000:
            long_description += "..."

        # Build enhanced properties with extracted data
        properties = {
            "layout_detected": True,
            "confidence": candidate.get('confidence', 0),
            "quality_score": candidate.get('qualityScore', 0),
            "content_type": candidate.get('contentType', 'product'),
            "page_number": candidate.get('pageNumber', 1),
            "patterns_detected": patterns,
            "source_chunks": [chunk.get('id') for chunk in chunks],
            "bounding_box": candidate.get('boundingBox'),
            "auto_generated": True,
            "generation_method": "layout_based_detection",
            "generation_timestamp": datetime.utcnow().isoformat()
        }

        # Add extracted product data to properties
        if extracted_data.get('dimensions'):
            properties['dimensions'] = extracted_data['dimensions']
        if extracted_data.get('designer'):
            properties['designer'] = extracted_data['designer']
        if extracted_data.get('colors'):
            properties['colors'] = extracted_data['colors']
        if extracted_data.get('materials'):
            properties['materials'] = extracted_data['materials']

        # Build metadata
        metadata = {
            "extracted_from": "layout_analysis",
            "layout_candidate_id": candidate.get('id'),
            "extraction_date": datetime.utcnow().isoformat(),
            "auto_created": True,
            "workspace_id": workspace_id,
            "detection_method": "layout_based_product_detection",
            "patterns_found": {
                "product_name": patterns.get('hasProductName', False),
                "dimensions": patterns.get('hasDimensions', False),
                "designer": patterns.get('hasDesignerAttribution', False),
                "description": patterns.get('hasProductDescription', False)
            }
        }

        return {
            "name": product_name,
            "description": description,
            "long_description": long_description,
            "source_document_id": document_id,
            "source_chunks": [chunk.get('id') for chunk in chunks],
            "properties": properties,
            "metadata": metadata,
            "status": "draft",
            "created_from_type": "layout_based_detection",
        }

    def _detect_products_in_markdown(self, markdown_content: str) -> List[Dict[str, Any]]:
        """
        ✅ NEW: Detect product candidates directly from markdown content.
        This analyzes the PyMuPDF4LLM markdown output for product patterns.
        """
        candidates = []

        # Split markdown into sections (PyMuPDF4LLM uses ----- as page breaks)
        sections = markdown_content.split('-----')

        for i, section in enumerate(sections):
            section = section.strip()
            if len(section) < 50:  # Skip very short sections
                continue

            # Analyze this section for product patterns
            candidate = self._analyze_markdown_section(section, i + 1)
            if candidate and candidate.get('confidence', 0) > 0.3:
                candidates.append(candidate)

        # Filter out non-product content
        product_candidates = [
            c for c in candidates
            if c.get('contentType') == 'product' and c.get('qualityScore', 0) > 0.5
        ]

        self.logger.info(f"🎯 Detected {len(product_candidates)} product candidates from {len(candidates)} total candidates")

        return product_candidates

    def _analyze_markdown_section(self, section: str, page_number: int) -> Optional[Dict[str, Any]]:
        """
        ✅ NEW: Analyze a markdown section for product patterns.
        """
        # Classify content type first
        content_type = self._classify_markdown_content(section)

        # Skip non-product content early
        if content_type != 'product' and content_type != 'unknown':
            return None

        # Extract product patterns
        patterns = self._detect_markdown_patterns(section)
        extracted_data = self._extract_markdown_data(section)

        # Calculate confidence based on patterns found
        confidence = 0
        if patterns.get('hasProductName'): confidence += 0.4
        if patterns.get('hasDimensions'): confidence += 0.3
        if patterns.get('hasDesignerAttribution'): confidence += 0.2
        if patterns.get('hasProductDescription'): confidence += 0.1

        # Calculate quality score
        quality_score = self._calculate_markdown_quality_score(section, patterns, extracted_data)

        return {
            'id': f"markdown_candidate_{page_number}",
            'name': extracted_data.get('productName') or f"Product Candidate {page_number}",
            'confidence': confidence,
            'pageNumber': page_number,
            'contentType': content_type,
            'patterns': patterns,
            'extractedData': extracted_data,
            'qualityScore': quality_score,
            'sourceSection': section[:200] + "..." if len(section) > 200 else section
        }

    def _classify_markdown_content(self, text: str) -> str:
        """
        ✅ NEW: Classify markdown content type based on patterns.
        """
        lower_text = text.lower()

        # Index/Table of Contents patterns (check first, highest priority)
        if (
            'table of contents' in lower_text or
            ('index' in lower_text and lower_text.count('page') > 1) or
            'contents' in lower_text or
            lower_text.count('page') > 2 or  # Multiple page references
            '...' in text  # Dotted lines typical in TOC
        ):
            return 'index'

        # Sustainability/Certification patterns
        if any(keyword in lower_text for keyword in [
            'sustainability', 'certification', 'environmental', 'eco-friendly',
            'carbon footprint', 'recycled', 'leed', 'greenguard'
        ]) and not any(product_keyword in lower_text for product_keyword in ['dimensions', 'designer', 'collection']):
            return 'sustainability'

        # Technical specifications patterns (table-like content)
        if (
            any(keyword in lower_text for keyword in [
                'technical characteristics', 'specifications', 'technical data',
                'properties', 'fire rating', 'weight per'
            ]) and
            ('|' in text or 'thickness' in lower_text) and  # Table format or thickness specs
            not any(product_keyword in lower_text for product_keyword in ['designer', 'collection'])
        ):
            return 'technical'

        # Moodboard patterns (but not if it has strong product indicators)
        if (
            any(keyword in lower_text for keyword in [
                'moodboard', 'mood board', 'inspiration', 'collection overview'
            ]) and
            not any(product_keyword in lower_text for product_keyword in ['dimensions', 'designer'])
        ):
            return 'moodboard'

        # Product patterns (UPPERCASE names + dimensions + descriptive content)
        uppercase_words = [word for word in text.split() if word.isupper() and len(word) > 1]
        has_dimensions = any(pattern in text for pattern in ['×', 'x', 'cm', 'mm'])
        has_product_context = any(keyword in lower_text for keyword in [
            'designer', 'collection', 'material', 'ceramic', 'porcelain', 'tile'
        ])

        if (
            len(uppercase_words) > 0 and  # Has uppercase product names
            has_dimensions and  # Has dimensions
            (has_product_context or len(text) > 200)  # Has product context or substantial content
        ):
            return 'product'

        return 'unknown'

    def _detect_markdown_patterns(self, text: str) -> Dict[str, bool]:
        """
        ✅ NEW: Detect product-specific patterns in markdown text.
        """
        import re

        # Look for product names in headers (## PRODUCT_NAME) or standalone UPPERCASE words
        has_product_name = bool(
            re.search(r'##?\s+[A-Z]{2,}(?:\s+[A-Z]{2,})*', text) or  # Header with UPPERCASE
            re.search(r'^[A-Z]{2,}(?:\s+[A-Z]{2,})*$', text, re.MULTILINE) or  # Standalone UPPERCASE line
            re.search(r'\b[A-Z]{2,}(?:\s+[A-Z]{2,})*\b', text)  # Any UPPERCASE words
        )

        return {
            'hasProductName': has_product_name,
            'hasDimensions': bool(re.search(r'\d+\s*[×x]\s*\d+|\d+\s*(?:mm|cm)', text)),  # Dimensions
            'hasDesignerAttribution': bool(re.search(r'(?:by|BY)\s+[A-Z][a-zA-Z\s{}]+|(?:studio|estudi)', text, re.IGNORECASE)),  # Designer
            'hasProductDescription': len(text) > 100 and bool(re.search(r'material|texture|finish|color|collection', text, re.IGNORECASE))
        }

    def _extract_markdown_data(self, text: str) -> Dict[str, Any]:
        """
        ✅ NEW: Extract structured product data from markdown text.
        """
        import re

        data = {}

        # Extract product name (prioritize headers, then standalone lines, then any UPPERCASE)
        product_name_match = (
            re.search(r'##?\s+([A-Z]{2,}(?:\s+[A-Z]{2,})*)', text) or  # Header format
            re.search(r'^([A-Z]{2,}(?:\s+[A-Z]{2,})*)$', text, re.MULTILINE) or  # Standalone line
            re.search(r'\b([A-Z]{2,}(?:\s+[A-Z]{2,})*)\b', text)  # Any UPPERCASE words
        )
        if product_name_match:
            data['productName'] = product_name_match.group(1)

        # Extract dimensions
        dimension_matches = re.findall(r'\d+\s*[×x]\s*\d+|\d+\s*(?:mm|cm)', text)
        if dimension_matches:
            data['dimensions'] = dimension_matches

        # Extract designer/studio
        designer_match = re.search(r'(?:by|BY)\s+([A-Z][a-zA-Z\s{}]+)|(?:studio|estudi)\s*([A-Z][a-zA-Z\s{}]*)', text, re.IGNORECASE)
        if designer_match:
            data['designer'] = (designer_match.group(1) or designer_match.group(2)).strip()

        # Extract colors (common color names)
        color_matches = re.findall(r'\b(?:white|black|grey|gray|beige|taupe|sand|clay|anthracite|cream|ivory|brown|blue|green|red|yellow|orange|purple|pink)\b', text, re.IGNORECASE)
        if color_matches:
            data['colors'] = list(set([c.lower() for c in color_matches]))

        # Extract materials
        material_matches = re.findall(r'\b(?:ceramic|porcelain|stone|marble|granite|wood|metal|glass|concrete|tile|vinyl|laminate)\b', text, re.IGNORECASE)
        if material_matches:
            data['materials'] = list(set([m.lower() for m in material_matches]))

        return data

    def _calculate_markdown_quality_score(self, text: str, patterns: Dict[str, bool], extracted_data: Dict[str, Any]) -> float:
        """
        ✅ NEW: Calculate quality score for markdown-based product candidate.
        """
        score = 0

        # Base score for having product patterns
        if patterns.get('hasProductName'): score += 0.3
        if patterns.get('hasDimensions'): score += 0.25
        if patterns.get('hasDesignerAttribution'): score += 0.2
        if patterns.get('hasProductDescription'): score += 0.15

        # Bonus for extracted data quality
        if extracted_data.get('productName') and len(extracted_data['productName']) > 2: score += 0.1
        if extracted_data.get('dimensions'): score += 0.1
        if extracted_data.get('designer'): score += 0.1
        if extracted_data.get('colors'): score += 0.05
        if extracted_data.get('materials'): score += 0.05

        # Penalty for very short content
        if len(text) < 100: score *= 0.5

        # Penalty for index-like content
        if 'page' in text.lower() and len(re.findall(r'\d+', text)) > 3: score *= 0.3

        return min(1.0, score)

    async def _store_markdown_analysis(self, document_id: str, analysis_metadata: Dict[str, Any]) -> None:
        """
        ✅ NEW: Store markdown analysis results in database.
        """
        try:
            # Store in document_layout_analysis table
            self.supabase.client.table('document_layout_analysis').insert({
                'document_id': document_id,
                'page_number': 1,
                'layout_elements': [],  # Empty for markdown analysis
                'reading_order': [],  # Empty for markdown analysis
                'structure_confidence': 0.8,  # Default confidence for markdown analysis
                'processing_version': '1.1.0',
                'analysis_metadata': analysis_metadata
            }).execute()

            self.logger.info(f"✅ Stored markdown analysis for document {document_id} with {analysis_metadata['total_candidates']} candidates")

        except Exception as e:
            self.logger.error(f"Failed to store markdown analysis: {e}")
            # Don't raise exception, just log the error

    # ============================================================================
    # Two-Stage Product Classification System
    # ============================================================================

    async def _stage1_fast_classification(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        ✅ NEW: Stage 1 - Fast text-only classification using Claude Haiku.

        Quickly filters chunks to identify potential product candidates.
        Uses Claude 4.5 Haiku for speed and cost efficiency.

        Args:
            chunks: List of document chunks to classify

        Returns:
            List of product candidates with classification metadata
        """
        try:
            from app.config import get_settings
            import anthropic

            settings = get_settings()
            client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

            product_candidates = []
            batch_size = 10  # Process chunks in batches for efficiency

            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]

                # Build batch classification prompt
                batch_prompt = self._build_stage1_batch_prompt(batch)

                # Call Claude Haiku for fast classification
                response = await self._call_claude_haiku(client, batch_prompt)

                # Parse batch results
                batch_results = self._parse_stage1_results(response, batch)

                # Add valid candidates to results
                for result in batch_results:
                    if result.get('is_product_candidate', False):
                        product_candidates.append(result)

                self.logger.debug(f"Stage 1 batch {i//batch_size + 1}: {len(batch_results)} candidates from {len(batch)} chunks")

            self.logger.info(f"🚀 Stage 1 classification: {len(product_candidates)} candidates from {len(chunks)} chunks")
            return product_candidates

        except Exception as e:
            self.logger.error(f"❌ Stage 1 classification failed: {str(e)}")
            # Fallback to basic filtering
            return [
                {
                    'chunk': chunk,
                    'is_product_candidate': self._is_valid_product_chunk(chunk),
                    'confidence': 0.5,
                    'classification_method': 'fallback'
                }
                for chunk in chunks
                if self._is_valid_product_chunk(chunk)
            ]

    async def _stage2_deep_enrichment(
        self,
        candidate: Dict[str, Any],
        document_id: str,
        workspace_id: str,
        index: int
    ) -> Optional[Dict[str, Any]]:
        """
        ✅ NEW: Stage 2 - Deep enrichment using Claude Sonnet.

        Performs detailed analysis and enrichment of confirmed product candidates.
        Uses Claude 4.5 Sonnet for high-quality results.

        Args:
            candidate: Product candidate from Stage 1
            document_id: Document ID
            workspace_id: Workspace ID
            index: Candidate index

        Returns:
            Enriched product data or None if validation fails
        """
        try:
            from app.config import get_settings
            import anthropic

            settings = get_settings()
            client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

            chunk = candidate['chunk']
            content = chunk.get('content', '')

            # Build enrichment prompt
            enrichment_prompt = self._build_stage2_enrichment_prompt(content, candidate)

            # Call Claude Sonnet for deep analysis
            response = await self._call_claude_sonnet(client, enrichment_prompt)

            # Parse enrichment results
            enrichment_data = self._parse_stage2_results(response)

            # Validate enrichment quality
            if not self._validate_enrichment_quality(enrichment_data):
                self.logger.warning(f"Stage 2 validation failed for candidate {index}")
                return None

            # Build final product data
            product_data = self._build_enriched_product_data(
                chunk=chunk,
                enrichment_data=enrichment_data,
                document_id=document_id,
                workspace_id=workspace_id,
                index=index
            )

            self.logger.debug(f"✅ Stage 2 enriched product: {product_data.get('name', 'Unknown')}")
            return product_data

        except Exception as e:
            self.logger.error(f"❌ Stage 2 enrichment failed for candidate {index}: {str(e)}")
            return None

    # ============================================================================
    # Two-Stage Classification Helper Methods
    # ============================================================================

    def _build_stage1_batch_prompt(self, chunks: List[Dict[str, Any]]) -> str:
        """Build prompt for Stage 1 batch classification with Claude Haiku."""
        chunk_texts = []
        for i, chunk in enumerate(chunks):
            content = chunk.get('content', '')[:500]  # Limit content for speed
            chunk_texts.append(f"CHUNK_{i}:\n{content}\n")

        return f"""You are a fast product classifier. Analyze these text chunks and identify which ones contain actual product information.

CHUNKS TO ANALYZE:
{chr(10).join(chunk_texts)}

For each chunk, determine if it contains:
- Product names (usually UPPERCASE like VALENOVA, PIQUÉ, ONA)
- Product dimensions (like 15×38, 20×40)
- Designer/brand information
- Material specifications

SKIP THE FOLLOWING (NOT PRODUCTS):
- Index pages, table of contents, signature books
- Sustainability content, environmental certifications, LEED info
- Technical tables without product names (just specs/data)
- Designer biographies (e.g., "John Doe was born in...", "Studio founded in...")
- Factory details (e.g., "Factory location: Spain...", "Production capacity...")
- Standalone technical specifications (water absorption, breaking strength without product)
- Moodboards, inspiration boards, design philosophy
- Cleaning/maintenance guides, application instructions
- Certification pages (ISO, ANSI, ASTM)

EXAMPLES OF NON-PRODUCTS TO SKIP:
- "ESTUDI{{H}}AC was founded in 2003 by designers..." → SKIP (designer biography)
- "Factory location: Castellón, Spain. Production capacity: 10,000 m²/day" → SKIP (factory details)
- "Water absorption: <0.5%, Breaking strength: >1300N" → SKIP (specs without product name)
- "Sustainability: Our commitment to the environment..." → SKIP (sustainability content)
- "Fresh Inspiration: Moodboard featuring natural textures..." → SKIP (moodboard)

ONLY CLASSIFY AS PRODUCT IF:
- Has a specific product name (UPPERCASE) AND
- Has dimensions/specifications AND
- Is NOT biography/factory/sustainability content

RESPOND WITH JSON ONLY:
{{
  "results": [
    {{
      "chunk_index": 0,
      "is_product": true/false,
      "confidence": 0.0-1.0,
      "product_name": "extracted name or null",
      "reasoning": "brief explanation"
    }}
  ]
}}

Focus on speed and accuracy. Be strict - when in doubt, mark as NOT a product."""

    def _build_stage2_enrichment_prompt(self, content: str, candidate: Dict[str, Any]) -> str:
        """Build prompt for Stage 2 deep enrichment with Claude Sonnet."""
        return f"""You are an expert product analyst. Perform deep analysis and enrichment of this product content.

PRODUCT CONTENT:
{content}

STAGE 1 ANALYSIS:
- Confidence: {candidate.get('confidence', 0)}
- Initial Assessment: {candidate.get('reasoning', 'N/A')}

FIRST, VALIDATE THIS IS ACTUALLY A PRODUCT:
- Does it have a specific product name (not just a designer/studio/factory name)?
- Does it have dimensions or technical specifications?
- Is this product content or designer biography/factory details/sustainability info?

RED FLAGS (NOT PRODUCTS):
- Designer biographies: "John Doe was born in...", "Studio founded in...", "Career began..."
- Factory details: "Factory location...", "Production capacity...", "Manufacturing facility..."
- Sustainability content: "Our commitment to environment...", "Carbon footprint...", "LEED certified..."
- Technical specs only: Just tables of data without product names
- Moodboards: "Fresh inspiration...", "Design philosophy...", "Mood board..."

IF THIS IS NOT A PRODUCT (biography/factory/sustainability/etc.), RESPOND:
{{
  "is_valid_product": false,
  "rejection_reason": "Designer biography / Factory details / Sustainability content / Technical specs only / Moodboard",
  "confidence_score": 0.0,
  "quality_assessment": "rejected"
}}

IF THIS IS A VALID PRODUCT, PERFORM COMPREHENSIVE ANALYSIS:

1. PRODUCT IDENTIFICATION:
   - Extract exact product name (must be a product, not a person/place)
   - Identify product category/type
   - Determine collection/series

2. SPECIFICATIONS:
   - Dimensions (extract all size variants)
   - Materials and composition
   - Colors and finishes available
   - Technical properties

3. DESIGN INFORMATION:
   - Designer/studio name (who designed it, not biography)
   - Design inspiration/story (brief, not full biography)
   - Style characteristics

4. METADATA:
   - Product codes/SKUs
   - Availability information
   - Related products
   - Applications/use cases

RESPOND WITH DETAILED JSON:
{{
  "is_valid_product": true,
  "product_name": "exact product name",
  "category": "product category",
  "collection": "collection name",
  "designer": "designer/studio",
  "dimensions": ["size1", "size2"],
  "materials": ["material1", "material2"],
  "colors": ["color1", "color2"],
  "description": "detailed description",
  "specifications": {{}},
  "metadata": {{}},
  "confidence_score": 0.0-1.0,
  "quality_assessment": "high/medium/low"
}}

Be thorough and accurate. REJECT non-product content. Extract all available information for valid products."""

    async def _call_claude_haiku(self, client, prompt: str) -> str:
        """Call Claude 4.5 Haiku for fast classification."""
        try:
            from app.config import get_settings
            settings = get_settings()

            response = client.messages.create(
                model=settings.anthropic_model_classification,  # claude-4-5-haiku-20250514
                max_tokens=2048,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}]
            )

            return response.content[0].text if response.content else ""

        except Exception as e:
            self.logger.error(f"Claude Haiku call failed: {str(e)}")
            return ""

    async def _call_claude_sonnet(self, client, prompt: str) -> str:
        """Call Claude 4.5 Sonnet for deep enrichment."""
        try:
            from app.config import get_settings
            settings = get_settings()

            response = client.messages.create(
                model=settings.anthropic_model_enrichment,  # claude-4-5-sonnet-20250514
                max_tokens=4096,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}]
            )

            return response.content[0].text if response.content else ""

        except Exception as e:
            self.logger.error(f"Claude Sonnet call failed: {str(e)}")
            return ""

    def _extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """
        Robust JSON extraction from Claude responses.
        Handles cases where Claude returns extra text before/after JSON.
        """
        import json
        import re

        if not response or not response.strip():
            raise ValueError("Empty response from Claude")

        response_clean = response.strip()

        # Remove markdown code blocks
        if response_clean.startswith('```json'):
            response_clean = response_clean[7:]
        elif response_clean.startswith('```'):
            response_clean = response_clean[3:]

        if response_clean.endswith('```'):
            response_clean = response_clean[:-3]

        response_clean = response_clean.strip()

        # Try to find JSON object or array in the response
        # Look for { ... } or [ ... ]
        json_match = re.search(r'(\{.*\}|\[.*\])', response_clean, re.DOTALL)

        if json_match:
            json_str = json_match.group(1)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                # If that fails, try to extract just the first complete JSON object
                brace_count = 0
                start_idx = json_str.find('{')
                if start_idx == -1:
                    start_idx = json_str.find('[')
                    if start_idx == -1:
                        raise ValueError(f"No JSON object found in response")

                for i in range(start_idx, len(json_str)):
                    if json_str[i] == '{' or json_str[i] == '[':
                        brace_count += 1
                    elif json_str[i] == '}' or json_str[i] == ']':
                        brace_count -= 1
                        if brace_count == 0:
                            # Found complete JSON object
                            complete_json = json_str[start_idx:i+1]
                            return json.loads(complete_json)

                raise ValueError(f"Could not extract complete JSON object: {str(e)}")
        else:
            # Last resort: try parsing the whole thing
            return json.loads(response_clean)

    def _parse_stage1_results(self, response: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parse Stage 1 classification results from Claude Haiku."""
        try:
            data = self._extract_json_from_response(response)
            results = []

            for result in data.get('results', []):
                chunk_index = result.get('chunk_index', 0)
                if chunk_index < len(chunks):
                    chunk = chunks[chunk_index]

                    if result.get('is_product', False):
                        results.append({
                            'chunk': chunk,
                            'is_product_candidate': True,
                            'confidence': result.get('confidence', 0.5),
                            'product_name': result.get('product_name'),
                            'reasoning': result.get('reasoning', ''),
                            'classification_method': 'claude_haiku'
                        })

            return results

        except Exception as e:
            self.logger.error(f"Failed to parse Stage 1 results: {str(e)}")
            # Fallback to basic filtering
            return [
                {
                    'chunk': chunk,
                    'is_product_candidate': self._is_valid_product_chunk(chunk),
                    'confidence': 0.5,
                    'classification_method': 'fallback'
                }
                for chunk in chunks
                if self._is_valid_product_chunk(chunk)
            ]

    def _parse_stage2_results(self, response: str) -> Dict[str, Any]:
        """Parse Stage 2 enrichment results from Claude Sonnet."""
        try:
            data = self._extract_json_from_response(response)

            # ✅ NEW: Check if Stage 2 rejected this as non-product
            is_valid_product = data.get('is_valid_product', True)

            # Ensure required fields exist
            enrichment_data = {
                'is_valid_product': is_valid_product,
                'rejection_reason': data.get('rejection_reason'),
                'product_name': data.get('product_name', 'Unknown Product'),
                'category': data.get('category', 'Unknown'),
                'collection': data.get('collection'),
                'designer': data.get('designer'),
                'dimensions': data.get('dimensions', []),
                'materials': data.get('materials', []),
                'colors': data.get('colors', []),
                'description': data.get('description', ''),
                'specifications': data.get('specifications', {}),
                'metadata': data.get('metadata', {}),
                'confidence_score': data.get('confidence_score', 0.5),
                'quality_assessment': data.get('quality_assessment', 'medium')
            }

            return enrichment_data

        except Exception as e:
            self.logger.error(f"Failed to parse Stage 2 results: {str(e)}")
            return {
                'is_valid_product': False,
                'rejection_reason': 'Parse error',
                'product_name': 'Unknown Product',
                'category': 'Unknown',
                'confidence_score': 0.3,
                'quality_assessment': 'low'
            }

    def _validate_enrichment_quality(self, enrichment_data: Dict[str, Any]) -> bool:
        """Validate the quality of Stage 2 enrichment results."""
        try:
            # ✅ NEW: Check if Stage 2 rejected this as non-product
            is_valid_product = enrichment_data.get('is_valid_product', True)
            if not is_valid_product:
                rejection_reason = enrichment_data.get('rejection_reason', 'Unknown')
                self.logger.warning(f"Stage 2 rejected as non-product: {rejection_reason}")
                return False

            # Check minimum confidence threshold
            confidence = enrichment_data.get('confidence_score', 0)
            if confidence < 0.4:
                self.logger.debug(f"Rejected: Low confidence ({confidence})")
                return False

            # Check for required fields
            product_name = enrichment_data.get('product_name', '')
            if not product_name or product_name == 'Unknown Product':
                self.logger.debug("Rejected: Missing or unknown product name")
                return False

            # ✅ NEW: Check product name is not a designer/studio name
            designer_indicators = ['studio', 'design', 'architects', 'founded', 'established', 'atelier']
            if any(indicator in product_name.lower() for indicator in designer_indicators):
                self.logger.warning(f"Rejected: Product name looks like designer/studio: {product_name}")
                return False

            # Check quality assessment
            quality = enrichment_data.get('quality_assessment', 'low')
            if quality == 'low' or quality == 'rejected':
                self.logger.debug(f"Rejected: Low quality assessment ({quality})")
                return False

            # Check for meaningful content
            description = enrichment_data.get('description', '')
            if len(description) < 20:
                self.logger.debug("Rejected: Description too short")
                return False

            # ✅ NEW: Check description is not a biography
            bio_indicators = [
                'born in', 'graduated', 'founded in', 'career began', 'based in',
                'studied at', 'education', 'professional background', 'years of experience'
            ]
            if any(indicator in description.lower() for indicator in bio_indicators):
                self.logger.warning("Rejected: Description contains biography content")
                return False

            # ✅ NEW: Check for factory details in description
            factory_indicators = [
                'factory location', 'production capacity', 'manufacturing facility',
                'plant location', 'headquarters', 'production site', 'industrial complex'
            ]
            if any(indicator in description.lower() for indicator in factory_indicators):
                self.logger.warning("Rejected: Description contains factory details")
                return False

            # ✅ NEW: Check for sustainability content in description
            sustainability_indicators = [
                'our commitment to', 'environmental responsibility', 'carbon footprint',
                'sustainability mission', 'green building', 'leed certification'
            ]
            if any(indicator in description.lower() for indicator in sustainability_indicators):
                self.logger.warning("Rejected: Description contains sustainability content")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Enrichment validation failed: {str(e)}")
            return False

    def _build_enriched_product_data(
        self,
        chunk: Dict[str, Any],
        enrichment_data: Dict[str, Any],
        document_id: str,
        workspace_id: str,
        index: int
    ) -> Dict[str, Any]:
        """Build final product data from enrichment results."""
        try:
            chunk_id = chunk.get('id')
            chunk_index = chunk.get('chunk_index', index)
            page_number = chunk.get('page_number', 1)
            content = chunk.get('content', '')

            # Extract enriched data
            product_name = enrichment_data.get('product_name', f'Product from Chunk {chunk_index}')
            description = enrichment_data.get('description', content[:200])

            # Build comprehensive metadata
            metadata = {
                'extracted_from': 'two_stage_classification',
                'stage1_method': 'claude_haiku',
                'stage2_method': 'claude_sonnet',
                'chunk_metadata': chunk.get('metadata', {}),
                'extraction_date': datetime.utcnow().isoformat(),
                'auto_created': True,
                'workspace_id': workspace_id,
                'enrichment_data': enrichment_data,
                'classification_confidence': enrichment_data.get('confidence_score', 0.5),
                'quality_assessment': enrichment_data.get('quality_assessment', 'medium')
            }

            # Add enriched metadata
            if enrichment_data.get('designer'):
                metadata['designer'] = enrichment_data['designer']
            if enrichment_data.get('collection'):
                metadata['collection'] = enrichment_data['collection']
            if enrichment_data.get('dimensions'):
                metadata['dimensions'] = enrichment_data['dimensions']
            if enrichment_data.get('materials'):
                metadata['materials'] = enrichment_data['materials']
            if enrichment_data.get('colors'):
                metadata['colors'] = enrichment_data['colors']

            # Build product data
            product_data = {
                "name": product_name,
                "description": description,
                "long_description": content[:1000] if len(content) > 200 else description,
                "source_document_id": document_id,
                "source_chunks": [chunk_id] if chunk_id else [],
                "properties": {
                    "source_chunk_id": chunk_id,
                    "document_id": document_id,
                    "chunk_index": chunk_index,
                    "page_number": page_number,
                    "content_length": len(content),
                    "auto_generated": True,
                    "generation_timestamp": datetime.utcnow().isoformat(),
                    "classification_method": "two_stage_claude",
                    "enrichment_confidence": enrichment_data.get('confidence_score', 0.5)
                },
                "metadata": metadata,
                "workspace_id": workspace_id,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }

            return product_data

        except Exception as e:
            self.logger.error(f"Failed to build enriched product data: {str(e)}")
            # Fallback to basic product creation
            return self._create_product_from_chunk(
                chunk=chunk,
                document_id=document_id,
                workspace_id=workspace_id,
                index=index
            )

    def _deduplicate_product_chunks(self, product_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        ✅ NEW: Deduplicate product chunks by product name.
        Merges chunks that represent the same product (e.g., multiple PIQUÉ chunks → 1 PIQUÉ product).

        Args:
            product_chunks: List of chunks identified as products

        Returns:
            Deduplicated list of product chunks (one per unique product name)
        """
        try:
            product_map = {}  # product_name -> best_chunk

            for chunk in product_chunks:
                content = chunk.get('content', '')
                product_name = self._extract_product_name(content)

                if not product_name:
                    # If we can't extract a name, use first line as fallback
                    first_line = content.split('\n')[0].strip()[:50]
                    product_name = first_line if first_line else f"Product_{chunk.get('chunk_index', 0)}"

                # Normalize product name (remove extra spaces, convert to uppercase)
                product_name = ' '.join(product_name.upper().split())

                # If this product name already exists, keep the chunk with more content
                if product_name in product_map:
                    existing_chunk = product_map[product_name]
                    existing_length = len(existing_chunk.get('content', ''))
                    current_length = len(content)

                    if current_length > existing_length:
                        self.logger.debug(f"Replacing {product_name}: {existing_length} chars → {current_length} chars")
                        product_map[product_name] = chunk
                    else:
                        self.logger.debug(f"Keeping existing {product_name}: {existing_length} chars (skipping {current_length} chars)")
                else:
                    product_map[product_name] = chunk
                    self.logger.debug(f"New product: {product_name}")

            deduplicated_chunks = list(product_map.values())
            self.logger.info(f"🔄 Deduplication: {len(product_chunks)} chunks → {len(deduplicated_chunks)} unique products")

            return deduplicated_chunks

        except Exception as e:
            self.logger.error(f"Deduplication failed: {e}")
            # Return original chunks if deduplication fails
            return product_chunks

