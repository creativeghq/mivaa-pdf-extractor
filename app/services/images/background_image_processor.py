"""
Background Image Processor Service

Processes deferred AI analysis for images that were uploaded during PDF processing.
This service runs asynchronously to avoid blocking the main PDF processing flow.

Features:
- CLIP visual embeddings (512D)
- Qwen3-VL 17B Vision analysis
- Claude Sonnet 4.5 Vision validation
- Understanding embeddings (1024D) - Qwen vision_analysis → Voyage AI
- Color embeddings (256D)
- Texture embeddings (256D)
- Application embeddings (512D)
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class BackgroundImageProcessor:
    """
    Service to process deferred image AI analysis in background.
    """
    
    def __init__(self, supabase_client):
        """
        Initialize background image processor.
        
        Args:
            supabase_client: Supabase client for database operations
        """
        self.supabase = supabase_client
        self.logger = logger
        
    async def process_pending_images(
        self,
        document_id: str,
        batch_size: int = 20,
        max_concurrent: int = 8
    ) -> Dict[str, Any]:
        """
        ⚡ OPTIMIZED: Process all images needing AI analysis for a document.

        Default max_concurrent increased from 3 to 8 for better throughput.
        Vision APIs can handle higher concurrency without rate limiting issues.

        Args:
            document_id: Document ID to process images for
            batch_size: Number of images to process in each batch (default: 20)
            max_concurrent: Maximum concurrent image processing tasks (default: 8)

        Returns:
            Dictionary with processing statistics
        """
        try:
            self.logger.info(f"🖼️ Starting background image processing for document {document_id}")
            
            # Query images needing analysis
            images_response = self.supabase.client.table('document_images').select('*').eq(
                'document_id', document_id
            ).is_('vision_analysis', 'null').limit(batch_size).execute()
            
            pending_images = images_response.data or []
            
            if not pending_images:
                self.logger.info(f"✅ No pending images for document {document_id}")
                return {
                    "success": True,
                    "images_processed": 0,
                    "message": "No pending images"
                }
            
            self.logger.info(f"📋 Found {len(pending_images)} images needing AI analysis")
            
            # Process images in batches with concurrency control
            processed_count = 0
            failed_count = 0
            
            # Create semaphore for concurrency control
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def process_with_semaphore(image):
                async with semaphore:
                    return await self._process_single_image(image)
            
            # Process all images concurrently (with semaphore limiting)
            tasks = [process_with_semaphore(image) for image in pending_images]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Count successes and failures
            for result in results:
                if isinstance(result, Exception):
                    failed_count += 1
                    self.logger.error(f"Image processing failed: {result}")
                elif result and result.get('success'):
                    processed_count += 1
                else:
                    failed_count += 1
            
            self.logger.info(f"✅ Background image processing complete: {processed_count} processed, {failed_count} failed")
            
            return {
                "success": True,
                "images_processed": processed_count,
                "images_failed": failed_count,
                "total_images": len(pending_images),
                "message": f"Processed {processed_count}/{len(pending_images)} images"
            }
            
        except Exception as e:
            self.logger.error(f"❌ Background image processing failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "images_processed": 0
            }
    
    async def _process_single_image(self, image: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single image with all AI analysis.
        
        Args:
            image: Image record from database
            
        Returns:
            Processing result dictionary
        """
        try:
            image_id = image['id']
            image_url = image.get('image_url')
            
            self.logger.info(f"🔍 Processing image {image_id}")
            
            # Import services
            from .real_image_analysis_service import RealImageAnalysisService
            from .real_embeddings_service import RealEmbeddingsService
            
            # Initialize services
            analysis_service = RealImageAnalysisService(self.supabase)
            embeddings_service = RealEmbeddingsService(self.supabase)
            
            # Run AI analysis
            analysis_result = await analysis_service.analyze_image(
                image_id=image_id,
                image_url=image_url,
                context={
                    "document_id": image.get('document_id'),
                    "page_number": image.get('page_number')
                }
            )
            
            if not analysis_result.get('success'):
                self.logger.warning(f"⚠️ Image analysis failed for {image_id}: {analysis_result.get('error')}")
                return {"success": False, "error": analysis_result.get('error')}
            
            # Generate specialized embeddings (color, texture, application)
            embeddings_result = await embeddings_service.generate_material_embeddings(
                image_url=image_url,
                material_properties=analysis_result.get('vision_analysis', {})
            )

            # Update database with analysis results (all embeddings saved to document_images)
            update_data = {
                "vision_analysis": analysis_result.get('vision_analysis'),
                "claude_validation": analysis_result.get('claude_validation'),
                "material_properties": analysis_result.get('material_properties'),  # ✅ Save extracted material properties
                "quality_score": analysis_result.get('quality_score'),              # ✅ Save quality score
                "confidence_score": analysis_result.get('confidence_score'),        # ✅ Save confidence score
                "processing_status": "completed",
                "updated_at": datetime.utcnow().isoformat()
            }

            # Add CLIP embedding to document_images (512D)
            clip_embedding = analysis_result.get('clip_embedding')
            if clip_embedding:
                update_data["visual_clip_embedding_512"] = clip_embedding
                self.logger.debug(f"✅ Adding CLIP embedding (512D) to document_images for {image_id}")

            # Add specialized embeddings if available
            if embeddings_result and embeddings_result.get('success'):
                if embeddings_result.get('color_embedding'):
                    update_data["color_embedding_256"] = embeddings_result['color_embedding']
                    self.logger.debug(f"✅ Adding color embedding (256D) to document_images for {image_id}")
                if embeddings_result.get('texture_embedding'):
                    update_data["texture_embedding_256"] = embeddings_result['texture_embedding']
                    self.logger.debug(f"✅ Adding texture embedding (256D) to document_images for {image_id}")
                if embeddings_result.get('application_embedding'):
                    update_data["application_embedding_512"] = embeddings_result['application_embedding']
                    self.logger.debug(f"✅ Adding application embedding (512D) to document_images for {image_id}")

            # Update image record with all data
            self.supabase.client.table('document_images').update(update_data).eq('id', image_id).execute()

            # Generate understanding embedding from vision analysis (Qwen → Voyage AI 1024D)
            has_understanding = False
            vision_analysis_data = analysis_result.get('vision_analysis')
            if vision_analysis_data:
                try:
                    understanding_embedding = await embeddings_service.generate_understanding_embedding(
                        vision_analysis=vision_analysis_data,
                        material_properties=analysis_result.get('material_properties')
                    )
                    if understanding_embedding:
                        from app.services.embeddings.vecs_service import get_vecs_service
                        vecs = get_vecs_service()
                        await vecs.upsert_understanding_embedding(
                            image_id=image_id,
                            embedding=understanding_embedding,
                            metadata={
                                'document_id': image.get('document_id'),
                                'workspace_id': image.get('workspace_id'),
                                'page_number': image.get('page_number', 1)
                            }
                        )
                        has_understanding = True
                        self.logger.info(f"✅ Understanding embedding generated and stored for {image_id}")
                except Exception as understanding_error:
                    self.logger.warning(f"⚠️ Understanding embedding failed for {image_id} (non-critical): {understanding_error}")

            # ✨ NEW: Extract product metadata from spec table images
            enriched_from_spec = False
            if vision_analysis_data:
                try:
                    enriched_from_spec = await self._enrich_product_metadata_from_spec_image(
                        image=image,
                        vision_analysis=vision_analysis_data
                    )
                except Exception as spec_err:
                    self.logger.warning(f"⚠️ Spec table enrichment failed for {image_id} (non-critical): {spec_err}")

            self.logger.info(f"✅ Image {image_id} processed successfully")

            return {
                "success": True,
                "image_id": image_id,
                "has_vision": bool(analysis_result.get('vision_analysis')),
                "has_claude": bool(analysis_result.get('claude_validation')),
                "has_clip": bool(analysis_result.get('clip_embedding')),
                "has_understanding": has_understanding,
                "has_color": bool(embeddings_result.get('color_embedding')),
                "has_texture": bool(embeddings_result.get('texture_embedding')),
                "has_application": bool(embeddings_result.get('application_embedding')),
                "enriched_from_spec_table": enriched_from_spec,
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to process image {image.get('id')}: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "image_id": image.get('id')
            }
    
    async def _enrich_product_metadata_from_spec_image(
        self,
        image: Dict[str, Any],
        vision_analysis: Dict[str, Any]
    ) -> bool:
        """
        After Qwen analysis, check if the image is a spec/technical data table.
        If yes, parse the packaging data and upsert it into associated products' metadata.

        Returns True if at least one product was enriched.
        """
        import json
        import anthropic

        # Get raw Qwen text output
        raw_output = vision_analysis.get('raw_qwen_output', '') or ''
        if isinstance(raw_output, dict):
            raw_output = json.dumps(raw_output)
        if not raw_output:
            raw_output = json.dumps(vision_analysis)

        raw_lower = raw_output.lower()

        # Quick heuristic: spec tables contain several packaging-related terms
        spec_keywords = ['pieces', 'pcs', 'weight', 'kg', 'lb', 'coverage', 'm²', 'pallet', 'box', 'sqft', 'boxes']
        keyword_hits = sum(1 for kw in spec_keywords if kw in raw_lower)
        is_spec_table = keyword_hits >= 3

        # Even for non-spec-table images, try to extract dimension data visible in the image
        if not is_spec_table:
            extracted = await self._extract_dimensions_from_image_text(
                image=image,
                raw_text=raw_output
            )
            return extracted

        self.logger.info(f"📊 Spec table detected for image {image['id']} ({keyword_hits} spec keywords found)")

        # Use Claude Haiku to parse structured packaging data from the extracted text
        client = anthropic.Anthropic()
        prompt = f"""The following text was extracted from a product spec table image in a tile/material catalog.
Parse the packaging/technical data and return ONLY valid JSON.

Extracted text:
{raw_output[:4000]}

Return JSON with this exact structure (use null for missing fields):
{{
  "is_spec_table": true,
  "product_variants": ["VARIANT NAME 1", "VARIANT NAME 2"],
  "packaging": {{
    "VARIANT NAME": {{
      "pieces_per_box": "value with unit",
      "weight_per_box_kg": "kg value",
      "weight_per_box_lb": "lb value",
      "boxes_per_pallet": "count",
      "coverage_per_box_m2": "m² value",
      "coverage_per_box_sqft": "sqft value",
      "pallet_dimensions_cm": "LxWxH cm",
      "weight_per_pallet_kg": "kg value"
    }}
  }}
}}

If this is not a spec table, return {{"is_spec_table": false}}.
Return only the JSON object, no markdown."""

        try:
            message = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            result_text = message.content[0].text.strip()
            # Strip markdown code fences if present
            if result_text.startswith('```'):
                result_text = result_text.split('```')[1].lstrip('json\n').strip()
            parsed = json.loads(result_text)
        except Exception as parse_err:
            self.logger.warning(f"⚠️ Failed to parse spec table JSON: {parse_err}")
            return False

        if not parsed.get('is_spec_table') or not parsed.get('packaging'):
            return False

        packaging_by_variant: Dict[str, Any] = parsed['packaging']

        # Find associated products via image_product_associations
        image_id = image['id']
        associations = self.supabase.client.table('image_product_associations').select(
            'product_id'
        ).eq('image_id', image_id).execute()
        product_ids = [a['product_id'] for a in (associations.data or [])]

        # Fallback: all products from the same document
        if not product_ids:
            prods = self.supabase.client.table('products').select('id').eq(
                'source_document_id', image['document_id']
            ).execute()
            product_ids = [p['id'] for p in (prods.data or [])]

        if not product_ids:
            self.logger.info(f"ℹ️ No associated products found for spec table image {image_id}")
            return False

        enriched = 0
        for product_id in product_ids:
            try:
                prod_resp = self.supabase.client.table('products').select(
                    'id, name, metadata'
                ).eq('id', product_id).single().execute()
                if not prod_resp.data:
                    continue

                product_name = (prod_resp.data.get('name') or '').upper()
                existing_metadata = prod_resp.data.get('metadata') or {}

                # Find packaging data for this product (match by name)
                matched_pkg = None
                for variant_key, pkg in packaging_by_variant.items():
                    vk = variant_key.upper()
                    if vk == product_name or product_name in vk or vk in product_name:
                        matched_pkg = pkg
                        break

                # If only one variant and no name match, use it for all products
                if matched_pkg is None and len(packaging_by_variant) == 1:
                    matched_pkg = list(packaging_by_variant.values())[0]

                if not matched_pkg:
                    continue

                # Build structured packaging fields (confidence 0.85 = from spec table image)
                new_pkg: Dict[str, Any] = {}
                field_map = [
                    ('pieces_per_box',    'pieces_per_box'),
                    ('weight_per_box_kg', 'weight_per_box'),
                    ('weight_per_box_lb', 'weight_per_box_lb'),
                    ('boxes_per_pallet',  'boxes_per_pallet'),
                    ('coverage_per_box_m2',   'coverage_per_box'),
                    ('coverage_per_box_sqft', 'coverage_per_box_sqft'),
                    ('pallet_dimensions_cm',  'pallet_dimensions'),
                    ('weight_per_pallet_kg',  'weight_per_pallet'),
                ]
                for src_key, dst_key in field_map:
                    val = matched_pkg.get(src_key)
                    if val:
                        new_pkg[dst_key] = {'value': val, 'confidence': 0.85, 'source': 'spec_table_image'}

                if not new_pkg:
                    continue

                # Merge: only fill gaps — existing values take priority
                existing_pkg = existing_metadata.get('packaging', {})
                merged_pkg = {**new_pkg}
                for k, v in existing_pkg.items():
                    merged_pkg[k] = v  # existing always wins

                updated_metadata = {**existing_metadata, 'packaging': merged_pkg}
                self.supabase.client.table('products').update({
                    'metadata': updated_metadata
                }).eq('id', product_id).execute()

                enriched += 1
                self.logger.info(
                    f"✅ Enriched packaging metadata for product {product_id} ({product_name}) from spec table"
                )

            except Exception as prod_err:
                self.logger.warning(f"⚠️ Failed to enrich product {product_id}: {prod_err}")
                continue

        return enriched > 0

    async def _extract_dimensions_from_image_text(
        self,
        image: Dict[str, Any],
        raw_text: str
    ) -> bool:
        """
        Scan Qwen's raw text output from any image (not just spec tables) for dimension
        patterns (sizes like "10×45 cm", thickness like "6.9 mm") and upsert them into
        the associated products' metadata if those fields are currently empty.

        Returns True if at least one product was enriched.
        """
        import re

        # Fast pre-check: must contain at least one dimension-like token
        dim_hint_pattern = re.compile(r'\d+\s*[×xX]\s*\d+|\d+[\.,]\d+\s*mm|\b\d+\s*mm\b', re.IGNORECASE)
        if not dim_hint_pattern.search(raw_text):
            return False  # No dimension patterns at all — skip quickly

        # Extract size patterns: e.g. "10×45", "15 x 38", "20X60" (cm implied in tile context)
        size_pattern = re.compile(r'(\d{1,4})\s*[×xX]\s*(\d{1,4})(?:\s*cm)?', re.IGNORECASE)
        found_sizes = []
        for m in size_pattern.finditer(raw_text):
            w, h = int(m.group(1)), int(m.group(2))
            # Sanity check: tile sizes are typically between 5 and 300 cm
            if 5 <= w <= 300 and 5 <= h <= 300:
                found_sizes.append(f"{w}×{h} cm")
        found_sizes = list(dict.fromkeys(found_sizes))  # deduplicate, preserve order

        # Extract thickness patterns: e.g. "6.9 mm", "8mm", "10 MM"
        # Look within 60 chars of a thickness keyword for robustness
        thickness_value: Optional[str] = None
        thick_keyword_pattern = re.compile(
            r'(?:thickness|spessore|epaisseur|st[aä]rke|dikte)[^\n]{0,60}?(\d+[\.,]?\d*)\s*mm',
            re.IGNORECASE
        )
        m = thick_keyword_pattern.search(raw_text)
        if m:
            thickness_value = f"{m.group(1).replace(',', '.')}mm"
        else:
            # Fallback: bare "X.Y mm" not near other sizes
            bare_mm = re.findall(r'\b(\d+[\.,]\d+)\s*mm\b', raw_text, re.IGNORECASE)
            if bare_mm:
                thickness_value = f"{bare_mm[0].replace(',', '.')}mm"

        if not found_sizes and not thickness_value:
            return False

        self.logger.info(
            f"📐 Dimension text found in image {image['id']}: "
            f"sizes={found_sizes}, thickness={thickness_value}"
        )

        # Find associated products
        image_id = image['id']
        associations = self.supabase.client.table('image_product_associations').select(
            'product_id'
        ).eq('image_id', image_id).execute()
        product_ids = [a['product_id'] for a in (associations.data or [])]

        if not product_ids:
            prods = self.supabase.client.table('products').select('id').eq(
                'source_document_id', image['document_id']
            ).execute()
            product_ids = [p['id'] for p in (prods.data or [])]

        if not product_ids:
            return False

        enriched = 0
        for product_id in product_ids:
            try:
                prod_resp = self.supabase.client.table('products').select(
                    'id, name, metadata'
                ).eq('id', product_id).single().execute()
                if not prod_resp.data:
                    continue

                existing_metadata = prod_resp.data.get('metadata') or {}
                updated_metadata = {**existing_metadata}
                changed = False

                # Only fill available_sizes if currently empty
                if found_sizes and not existing_metadata.get('available_sizes'):
                    updated_metadata['available_sizes'] = found_sizes
                    changed = True

                # Only fill thickness if currently empty
                if thickness_value:
                    mat_props = dict(existing_metadata.get('material_properties') or {})
                    if not mat_props.get('thickness'):
                        mat_props['thickness'] = {
                            'value': thickness_value,
                            'confidence': 0.70,
                            'source': 'image_text'
                        }
                        updated_metadata['material_properties'] = mat_props
                        changed = True

                if changed:
                    self.supabase.client.table('products').update({
                        'metadata': updated_metadata
                    }).eq('id', product_id).execute()
                    enriched += 1
                    self.logger.info(
                        f"✅ Enriched dimensions for product {product_id} from image text"
                    )

            except Exception as prod_err:
                self.logger.warning(f"⚠️ Failed to enrich product {product_id} with dimensions: {prod_err}")
                continue

        return enriched > 0

    async def process_all_pending_images_for_document(
        self,
        document_id: str
    ) -> Dict[str, Any]:
        """
        Process ALL pending images for a document (not just one batch).
        Continues processing until no more pending images remain.
        
        Args:
            document_id: Document ID to process images for
            
        Returns:
            Dictionary with total processing statistics
        """
        total_processed = 0
        total_failed = 0
        batch_count = 0
        
        while True:
            batch_count += 1
            self.logger.info(f"📦 Processing batch {batch_count} for document {document_id}")
            
            result = await self.process_pending_images(
                document_id=document_id,
                batch_size=10,
                max_concurrent=3
            )
            
            if not result.get('success'):
                break
            
            processed = result.get('images_processed', 0)
            failed = result.get('images_failed', 0)
            
            total_processed += processed
            total_failed += failed
            
            # If no images were processed, we're done
            if processed == 0:
                break
            
            # Small delay between batches to avoid overwhelming the system
            await asyncio.sleep(1)
        
        self.logger.info(f"✅ All batches complete: {total_processed} processed, {total_failed} failed across {batch_count} batches")
        
        return {
            "success": True,
            "total_processed": total_processed,
            "total_failed": total_failed,
            "batches_processed": batch_count,
            "message": f"Processed {total_processed} images across {batch_count} batches"
        }


# Convenience function to start background processing
async def start_background_image_processing(document_id: str, supabase_client):
    """
    Start background image processing for a document.
    
    Args:
        document_id: Document ID to process images for
        supabase_client: Supabase client instance
    """
    processor = BackgroundImageProcessor(supabase_client)
    return await processor.process_all_pending_images_for_document(document_id)


