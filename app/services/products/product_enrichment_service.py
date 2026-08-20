"""
Product Enrichment Service - Stage 5 Implementation

This service enriches products with real data from:
1. Image analysis results (Claude Vision, SLIG embeddings)
2. Material properties extracted from images
3. Product embeddings for semantic search
4. Related product linking
5. Product descriptions from image analysis

Replaces mock enrichment with real AI-powered data extraction.
"""

import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import os

logger = logging.getLogger(__name__)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Import real embeddings service
from app.services.embeddings.real_embeddings_service import RealEmbeddingsService

# Import real quality scoring service (Step 5)
from app.services.ai_validation.real_quality_scoring_service import RealQualityScoringService

# Anthropic shim (httpx-backed, no SDK dependency)
from app.services.core.ai_client_service import get_ai_client_service
from app.services.utilities.prompt_registry import load_prompt, render


class ProductEnrichmentService:
    """
    Enriches products with real data from image analysis and AI models.
    
    This service:
    - Links products to images based on semantic similarity
    - Extracts material properties from image analysis
    - Generates product descriptions
    - Creates product embeddings
    - Links related products
    """
    
    def __init__(self, supabase_client):
        """
        Initialize product enrichment service.

        Args:
            supabase_client: Supabase client for database operations
        """
        self.supabase = supabase_client
        self.logger = logger
        self.quality_scoring_service = RealQualityScoringService(supabase_client)  # Step 5
        self.anthropic_client = get_ai_client_service().anthropic if ANTHROPIC_API_KEY else None
    
    async def enrich_product(
        self,
        product_id: str,
        product_data: Dict[str, Any],
        document_id: str,
        workspace_id: str
    ) -> Dict[str, Any]:
        """
        Enrich a single product with real data.
        
        Args:
            product_id: UUID of product to enrich
            product_data: Current product data
            document_id: Source document ID
            workspace_id: Workspace ID
            
        Returns:
            Dictionary with enrichment results
        """
        try:
            self.logger.info(f"🎯 Starting product enrichment for {product_id}")
            
            # Step 1: Find related images
            related_images = await self._find_related_images(
                product_id=product_id,
                product_data=product_data,
                document_id=document_id
            )
            
            if not related_images:
                self.logger.warning(f"No related images found for product {product_id}")
                return {"success": False, "error": "No related images found"}
            
            self.logger.info(f"✅ Found {len(related_images)} related images")
            
            # Step 2: Extract material properties from images
            material_properties = await self._extract_material_properties_from_images(
                related_images=related_images
            )
            
            # Step 3: Generate enhanced description
            enhanced_description = await self._generate_enhanced_description(
                product_data=product_data,
                material_properties=material_properties,
                image_analysis=related_images[0].get('analysis', {}) if related_images else {}
            )
            
            # Step 4: Create product embedding
            embedding = await self._create_product_embedding(
                product_name=product_data.get('name', ''),
                description=enhanced_description,
                material_properties=material_properties,
                product_id=product_id,
                workspace_id=workspace_id,
            )
            product_embedding = embedding.get('embeddings') or {}

            # Step 5: Find related products.
            # The RPC ranks on text_embedding_1024, so it wants THE TEXT VECTOR.
            # The whole six-key embeddings MAP was being handed over as
            # `query_embedding`.
            related_products = await self._find_related_products(
                product_id=product_id,
                query_vector=product_embedding.get('text_1024'),
                workspace_id=workspace_id
            )

            # Step 6: Update product in database
            enrichment_result = {
                "product_id": product_id,
                "material_properties": material_properties,
                "enhanced_description": enhanced_description,
                "product_embedding": product_embedding,
                # The envelope, not just the vectors. "This product has no
                # embeddings" has to be answerable as failed-vs-never-attempted
                # from the row afterwards, which an empty map cannot do (#348,
                # pipeline convention 1).
                "embedding": embedding,
                "embedding_status": embedding.get('status'),
                "related_images": [img['id'] for img in related_images],
                "related_products": related_products,
                "enrichment_timestamp": datetime.utcnow().isoformat(),
                "success": True
            }
            
            # Store enrichment in database
            await self._store_enrichment_results(
                product_id=product_id,
                enrichment_result=enrichment_result
            )
            
            self.logger.info(f"✅ Product enrichment complete for {product_id}")
            return enrichment_result
            
        except Exception as e:
            self.logger.error(f"❌ Product enrichment failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _find_related_images(
        self,
        product_id: str,
        product_data: Dict[str, Any],
        document_id: str
    ) -> List[Dict[str, Any]]:
        """Find images related to a product."""
        try:
            # Get all images from the document
            response = self.supabase.client.table('document_images').select(
                '*'
            ).eq('document_id', document_id).execute()
            
            if not response.data:
                return []
            
            images = response.data
            
            # Filter images with real analysis data
            related_images = []
            for image in images:
                # Check if image has real analysis (not mock)
                if image.get('vision_analysis') or image.get('claude_validation'):
                    related_images.append(image)
            
            return related_images[:5]  # Limit to 5 most relevant images
            
        except Exception as e:
            self.logger.error(f"Error finding related images: {e}")
            return []
    
    async def _extract_material_properties_from_images(
        self,
        related_images: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract material properties from image analysis results."""
        try:
            combined_properties = {
                "colors": [],
                "finishes": [],
                "patterns": [],
                "textures": [],
                "materials": [],
                "confidence": 0.0
            }
            
            confidences = []
            
            for image in related_images:
                # Get material properties from image analysis
                material_props = image.get('material_properties', {})
                
                if material_props:
                    # Collect unique values
                    if material_props.get('color'):
                        combined_properties["colors"].append(material_props['color'])
                    if material_props.get('finish'):
                        combined_properties["finishes"].append(material_props['finish'])
                    if material_props.get('pattern'):
                        combined_properties["patterns"].append(material_props['pattern'])
                    if material_props.get('texture'):
                        combined_properties["textures"].append(material_props['texture'])
                    if material_props.get('composition'):
                        combined_properties["materials"].append(material_props['composition'])
                    
                    confidences.append(material_props.get('confidence', 0.0))
            
            # Remove duplicates and calculate average confidence
            combined_properties["colors"] = list(set(combined_properties["colors"]))
            combined_properties["finishes"] = list(set(combined_properties["finishes"]))
            combined_properties["patterns"] = list(set(combined_properties["patterns"]))
            combined_properties["textures"] = list(set(combined_properties["textures"]))
            combined_properties["materials"] = list(set(combined_properties["materials"]))
            
            if confidences:
                combined_properties["confidence"] = sum(confidences) / len(confidences)
            
            return combined_properties
            
        except Exception as e:
            self.logger.error(f"Error extracting material properties: {e}")
            return {}
    
    async def _generate_enhanced_description(
        self,
        product_data: Dict[str, Any],
        material_properties: Dict[str, Any],
        image_analysis: Dict[str, Any]
    ) -> str:
        """Generate enhanced product description using Claude."""
        try:
            if not self.anthropic_client:
                self.logger.warning("Anthropic client not available, using basic description")
                return product_data.get('description', '')
            
            prompt = render(
                await load_prompt("extraction", "product_description_writer",
                                  stage="entity_creation"),
                product_name=product_data.get('name', 'Unknown'),
                current_description=product_data.get('description', 'N/A'),
                colors=', '.join(material_properties.get('colors', [])),
                finishes=', '.join(material_properties.get('finishes', [])),
                patterns=', '.join(material_properties.get('patterns', [])),
                textures=', '.join(material_properties.get('textures', [])),
                materials=', '.join(material_properties.get('materials', [])),
                image_analysis=json.dumps(image_analysis, indent=2)[:500],
            )

            from app.services.core.claude_helper import tracked_claude_call
            response = tracked_claude_call(
                task="product_enrichment_description",
                model="claude-opus-4-8",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )

            return response.content[0].text if response.content else product_data.get('description', '')
            
        except Exception as e:
            self.logger.error(f"Error generating enhanced description: {e}")
            return product_data.get('description', '')
    
    async def _create_product_embedding(
        self,
        product_name: str,
        description: str,
        material_properties: Dict[str, Any],
        product_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate the product's embeddings and report whether they LANDED.

        Returns an envelope, never a bare map::

            {"status": "ok" | "failed", "embeddings": {...},
             "model_versions": {...}, "reason": str | None, "error": str | None}

        `{}` used to be the answer to three different questions -- the embedder
        raised, the embedder ran and produced nothing, and there was nothing worth
        embedding -- so a product could be persisted un-embedded with nothing on the
        row marking it for replay (#348). `generate_all_embeddings` already tells
        those apart via `success` / `error`; this method discarded that with
        `.get("embeddings", {})` and re-flattened it to `{}` two lines later.

        `status: "ok"` means at least one vector came back, NOT that it was stored.
        Storing is `_store_enrichment_results`, which stamps the failure marker when
        the text vector does not reach the column.
        """
        try:
            materials = ', '.join(material_properties.get('materials', []) or [])
            colors = ', '.join(material_properties.get('colors', []) or [])
            embedding_text = (
                f"{product_name}. {description}. "
                f"Materials: {materials}. Colors: {colors}"
            )
            # The f-string is never empty -- it carries its own punctuation and
            # labels -- so measure the PARTS, not the result. Embedding
            # ". . Materials: . Colors:" produces a vector of the template, which
            # then sits near every query in the catalog.
            if not any(part.strip() for part in (product_name, description, materials, colors)):
                self.logger.error(
                    f"Nothing to embed for product {product_id or '(unknown)'} -- "
                    f"name, description, materials and colors are all empty"
                )
                return {
                    "status": "failed",
                    "embeddings": {},
                    "model_versions": {},
                    "reason": "no_embeddable_text",
                    "error": None,
                }

            # Use RealEmbeddingsService to generate all embedding types
            embeddings_service = RealEmbeddingsService(self.supabase)

            all_embeddings = await embeddings_service.generate_all_embeddings(
                # entity_id was the literal "temp", with a comment saying it would be
                # updated with the product id. Nothing updated it. The ids are also
                # what attributes the Voyage spend: an untenanted ai_usage_logs row is
                # invisible to per-workspace cost views AND to that table's own
                # is_workspace_admin(workspace_id) policy, which cannot match NULL.
                entity_id=product_id or "unknown",
                entity_type="product",
                text_content=embedding_text,
                material_properties=material_properties,
                workspace_id=workspace_id,
                product_id=product_id,
            )

            produced = all_embeddings.get("embeddings") or {}
            if not all_embeddings.get("success") or not produced:
                reason = all_embeddings.get("error") or "no_vectors_generated"
                self.logger.error(
                    f"Product embedding produced no vectors for "
                    f"{product_id or '(unknown)'}: {reason}"
                )
                return {
                    "status": "failed",
                    "embeddings": produced,
                    "model_versions": {},
                    "reason": reason,
                    "error": all_embeddings.get("error"),
                }

            return {
                "status": "ok",
                "embeddings": produced,
                # Provenance travels with the vector, so the row records the model
                # that ACTUALLY produced it rather than the one configured at write
                # time.
                "model_versions": (all_embeddings.get("metadata") or {}).get("model_versions", {}),
                "reason": None,
                "error": None,
            }

        except Exception as e:
            self.logger.error(f"Error creating product embedding: {e}")
            return {
                "status": "failed",
                "embeddings": {},
                "model_versions": {},
                "reason": "exception",
                "error": str(e),
            }
    async def _find_related_products(
        self,
        product_id: str,
        query_vector: Optional[List[float]],
        workspace_id: str
    ) -> List[str]:
        """Find related products by text-embedding similarity.

        `search_similar_products` is not a function in this database and never has
        been, so every call landed in the warning handler below and the answer was a
        permanent `[]` -- the silent-zero shape wearing "related products are
        genuinely rare" as a disguise. The real RPC is
        `search_products_by_embedding(query_embedding, p_workspace_id, p_limit)`; it
        returns `product_id` (not `id`), and it has no exclude argument, so the
        source product is filtered out here.
        """
        if not query_vector:
            # No vector is not "no neighbours". Asking anyway would rank against
            # whatever a malformed argument coerces to.
            return []
        try:
            response = self.supabase.client.rpc(
                'search_products_by_embedding',
                {
                    'query_embedding': query_vector,
                    'p_workspace_id': workspace_id,
                    # +1: the product ranks first against its own vector.
                    'p_limit': 6,
                }
            ).execute()

            rows = response.data or []
            return [
                r['product_id'] for r in rows
                if r.get('product_id') and r['product_id'] != product_id
            ][:5]

        except Exception as e:
            self.logger.warning(f"Error finding related products: {e}")
            return []
    async def _store_enrichment_results(
        self,
        product_id: str,
        enrichment_result: Dict[str, Any]
    ) -> bool:
        """
        Store enrichment results in database.

        Text embedding: the 1024D product vector from step 4 is written to
        `products.text_embedding_1024` -- the column `search_products_by_embedding`
        ranks on, and the one `text_embedding_backfill` reads as "needs embedding".
        It used to be generated and dropped on the floor, so every product this
        service enriched was billed for a Voyage call and stayed invisible to product
        vector search regardless of whether that call succeeded.

        Per-aspect and visual vectors are NOT written here: they belong to VECS
        collections keyed by IMAGE, and this path has no image entity to key them to.

        Both jsonb columns are read-modify-written. They are not this service's to
        own -- stage 4 writes `metadata.facet_canonicalization`, stage 0 writes
        `metadata.embedding_failure`, and `_create_product_from_candidate` writes the
        layout provenance into `properties`. A whole-object update erased all of it,
        including the neighbours of the marker this method now stamps itself.
        """
        try:
            # Calculate real quality score (not hardcoded)
            product_data = {
                "name": enrichment_result.get('name', ''),
                "description": enrichment_result.get('enhanced_description', ''),
                "long_description": enrichment_result.get('enhanced_description', ''),
                "properties": enrichment_result.get('material_properties', {}),
                "metadata": {
                    "related_images": enrichment_result.get('related_images', []),
                    "related_products": enrichment_result.get('related_products', [])
                }
            }

            quality_score, quality_metrics = self.quality_scoring_service.calculate_product_quality_score(product_data)

            update_data: Dict[str, Any] = {
                "long_description": enrichment_result.get('enhanced_description', ''),
                "quality_score": quality_score,
                "quality_metrics": quality_metrics,
                "updated_at": datetime.utcnow().isoformat()
            }

            embedding = enrichment_result.get('embedding') or {}
            embedding_status = embedding.get('status') or 'failed'
            failure_reason = embedding.get('reason') or 'unknown'
            text_vector = (embedding.get('embeddings') or {}).get('text_1024')

            if text_vector and len(text_vector) == 1024:
                update_data["text_embedding_1024"] = "[" + ",".join(str(x) for x in text_vector) + "]"
                update_data["text_embedding_schema_version"] = 1
                model = (embedding.get('model_versions') or {}).get('text')
                if model:
                    update_data["text_embedding_1024_model"] = model
            elif text_vector:
                # A vector of the wrong width is not a vector of this space. The
                # column would accept it and then rank it against every query -- the
                # "same shape, different space" failure the no-fallback-embedder rule
                # exists to prevent.
                self.logger.error(
                    f"Refusing to store a {len(text_vector)}D product embedding for "
                    f"{product_id} -- text_embedding_1024 is 1024D"
                )
                embedding_status = 'failed'
                failure_reason = f'dim_mismatch_{len(text_vector)}'
            elif embedding_status == 'ok':
                # Vectors came back, but none of them was the text one, so nothing
                # reaches the column the product search ranks on.
                embedding_status = 'failed'
                failure_reason = 'no_text_vector'

            existing = await self._read_product_jsonb(product_id)
            if existing is None:
                # Merging is impossible, so writing either jsonb column would DESTROY
                # markers rather than add one. Skip both and say so -- the scalar half
                # of the enrichment is still worth storing.
                self.logger.error(
                    f"Could not read existing jsonb for product {product_id} -- "
                    f"metadata/properties left untouched to avoid clobbering markers; "
                    f"embedding_status={embedding_status} NOT recorded"
                )
            else:
                existing_metadata, existing_properties = existing

                metadata = dict(existing_metadata)
                metadata.update({
                    "enriched": True,
                    "enrichment_timestamp": enrichment_result.get('enrichment_timestamp'),
                    "related_images": enrichment_result.get('related_images', []),
                    "related_products": enrichment_result.get('related_products', []),
                    "quality_score": quality_score,
                    "quality_metrics": quality_metrics
                })

                if embedding_status == 'ok':
                    prior = metadata.pop('embedding_failure', None)
                    if prior is not None:
                        # Same clearing ritual as text_embedding_backfill: keep the
                        # history, stop the sweep re-picking the row.
                        metadata['embedding_failure_resolved'] = {
                            **(prior if isinstance(prior, dict) else {'previous': prior}),
                            'resolved_at': datetime.utcnow().isoformat(),
                            'resolved_by': 'product_enrichment',
                        }
                else:
                    # The SAME key, shape and consumer as the Stage 0 marker --
                    # text_embedding_backfill already finds `embedding_failure` and
                    # clears it on success. A second vocabulary for "this product's
                    # vector never landed" would need a second sweep to go with it.
                    metadata['embedding_failure'] = {
                        'failed_at': datetime.utcnow().isoformat(),
                        'stage': 'product_enrichment',
                        'reason': failure_reason,
                        'error': embedding.get('error'),
                    }
                    self.logger.error(
                        f"Product {product_id} enriched WITHOUT a text embedding "
                        f"({failure_reason}) -- flagged via metadata.embedding_failure "
                        f"for backfill; it is invisible to product vector search "
                        f"until that runs"
                    )

                update_data["metadata"] = metadata

                # An empty material_properties map is exactly what the extraction
                # step returns when it FAILS, so merging rather than replacing is
                # also what stops a failed extraction wiping the layout provenance.
                properties = dict(existing_properties)
                properties.update(enrichment_result.get('material_properties') or {})
                update_data["properties"] = properties

            response = self.supabase.client.table('products').update(
                update_data
            ).eq('id', product_id).execute()

            if response.data:
                self.logger.info(f"Stored enrichment results for product {product_id}")
            else:
                self.logger.error(
                    f"Enrichment update matched no row for product {product_id} -- "
                    f"nothing was stored"
                )

            return bool(response.data)

        except Exception as e:
            self.logger.error(f"Error storing enrichment results: {e}")
            return False

    async def _read_product_jsonb(
        self,
        product_id: str
    ) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Current (metadata, properties) for a product, or None if unreadable.

        None is deliberately distinct from ({}, {}): the first means "do not write
        these columns", the second means "they are genuinely empty".
        """
        try:
            current = (
                self.supabase.client.table('products')
                .select('metadata, properties')
                .eq('id', product_id)
                .maybe_single()
                .execute()
            )
            row = current.data
            if not row:
                return None
            return (row.get('metadata') or {}), (row.get('properties') or {})
        except Exception as e:
            self.logger.warning(f"Could not read product jsonb for {product_id}: {e}")
            return None
