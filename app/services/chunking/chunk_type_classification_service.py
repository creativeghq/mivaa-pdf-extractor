"""
Chunk Type Classification Service

Provides intelligent semantic classification of document chunks into predefined types
with structured metadata extraction for each type. Uses pattern recognition and
content analysis to determine the most appropriate classification.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import asyncio
import httpx
from app.config import settings

logger = logging.getLogger(__name__)

class ChunkType(Enum):
    """Chunk type enumeration for semantic classification"""
    PRODUCT_DESCRIPTION = 'product_description'
    TECHNICAL_SPECS = 'technical_specs'
    VISUAL_SHOWCASE = 'visual_showcase'
    DESIGNER_STORY = 'designer_story'
    COLLECTION_OVERVIEW = 'collection_overview'
    SUPPORTING_CONTENT = 'supporting_content'
    INDEX_CONTENT = 'index_content'
    SUSTAINABILITY_INFO = 'sustainability_info'
    CERTIFICATION_INFO = 'certification_info'
    UNCLASSIFIED = 'unclassified'

class ChunkClassificationResult:
    """Classification result with confidence and metadata"""
    def __init__(self, chunk_type: ChunkType, confidence: float, metadata: Dict[str, Any], reasoning: str):
        self.chunk_type = chunk_type
        self.confidence = confidence
        self.metadata = metadata
        self.reasoning = reasoning

class ChunkTypeClassificationService:
    """
    Chunk Type Classification Service
    
    Provides intelligent semantic classification of document chunks into predefined types
    with structured metadata extraction for each type. Uses pattern recognition and
    content analysis to determine the most appropriate classification.
    """
    
    def __init__(self):
        self.supabase_url = settings.supabase_url
        self.supabase_service_key = settings.supabase_service_role_key
        
    async def classify_chunk(self, content: str) -> ChunkClassificationResult:
        """
        Classify a single chunk and extract structured metadata.

        Uses pattern matching first. For low-confidence results (SUPPORTING_CONTENT
        or UNCLASSIFIED), falls back to Qwen for better accuracy.

        Args:
            content: The text content of the chunk

        Returns:
            ChunkClassificationResult with type, confidence, metadata, and reasoning
        """
        logger.info(f"🎯 Classifying chunk with {len(content)} characters")

        try:
            # Primary: pattern-based classification (fast, no API cost)
            classification = self._analyze_content_patterns(content)

            # For ambiguous cases, upgrade with Qwen if endpoint is available
            is_ambiguous = (
                classification['chunk_type'] in (ChunkType.SUPPORTING_CONTENT, ChunkType.UNCLASSIFIED)
                and len(content) >= 50  # Not worth LLM call for tiny chunks
            )
            if is_ambiguous:
                qwen_result = await self._classify_with_qwen(content)
                if qwen_result and qwen_result.confidence > classification['confidence']:
                    logger.info(f"   🤖 Qwen upgraded classification: {classification['chunk_type'].value} → {qwen_result.chunk_type.value} ({qwen_result.confidence:.2f})")
                    return qwen_result

            # Extract structured metadata based on final classification
            metadata = self._extract_structured_metadata(content, classification['chunk_type'])

            return ChunkClassificationResult(
                chunk_type=classification['chunk_type'],
                confidence=classification['confidence'],
                metadata=metadata,
                reasoning=classification['reasoning']
            )
        except Exception as error:
            logger.error(f"❌ Failed to classify chunk: {error}")

            return ChunkClassificationResult(
                chunk_type=ChunkType.UNCLASSIFIED,
                confidence=0.0,
                metadata={},
                reasoning=f"Classification failed: {error}"
            )

    async def _classify_with_qwen(self, content: str) -> Optional[ChunkClassificationResult]:
        """
        Use Qwen to classify chunks that pattern matching couldn't confidently resolve.
        Returns None if Qwen endpoint is unavailable (caller keeps pattern result).
        """
        try:
            import json
            import httpx
            from app.config import settings as app_settings

            qwen_config = app_settings.get_qwen_config()
            if not qwen_config.get("enabled") or not qwen_config.get("endpoint_url"):
                return None

            chunk_types = ", ".join(ct.value for ct in ChunkType if ct != ChunkType.UNCLASSIFIED)
            system_prompt = f"""You are a document chunk classifier for a material catalog platform.
Classify the given text into exactly one of these types: {chunk_types}

Rules:
- product_description: describes a specific product (name, features, colors, sizes, finishes)
- technical_specs: measurements, dimensions, technical properties, installation data
- visual_showcase: describes aesthetics, visual appearance, mood, atmosphere
- designer_story: about a designer, studio, creative process, or inspiration
- collection_overview: introduces a product collection, range, or theme
- supporting_content: general brand/company text, marketing copy, introductory content
- index_content: table of contents, page numbers, navigation
- sustainability_info: environmental, eco, recycled, carbon, sustainability
- certification_info: ISO, CE, certifications, standards compliance

Respond with ONLY this JSON: {{"chunk_type": "<type>", "confidence": <0.0-1.0>, "reasoning": "<brief>"}}"""

            async with httpx.AsyncClient(timeout=12.0) as http:
                response = await http.post(
                    qwen_config["endpoint_url"],
                    headers={
                        "Authorization": f"Bearer {qwen_config['endpoint_token']}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": qwen_config["model"],
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"Classify this chunk:\n\n{content[:1500]}"},
                        ],
                        "max_tokens": 128,
                        "temperature": 0.1,
                    },
                )

            if response.status_code != 200:
                return None

            raw = response.json()["choices"][0]["message"]["content"].strip()
            # Strip markdown code fences
            if raw.startswith("```json"):
                raw = raw[7:]
            if raw.startswith("```"):
                raw = raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]

            data = json.loads(raw.strip())
            chunk_type_str = data.get("chunk_type", "")
            confidence = float(data.get("confidence", 0.0))

            # Map string back to enum
            try:
                chunk_type = ChunkType(chunk_type_str)
            except ValueError:
                return None

            metadata = self._extract_structured_metadata(content, chunk_type)
            return ChunkClassificationResult(
                chunk_type=chunk_type,
                confidence=confidence,
                metadata=metadata,
                reasoning=data.get("reasoning", "Qwen classification")
            )

        except Exception as e:
            logger.debug(f"Qwen chunk classification skipped: {e}")
            return None
    
    async def classify_chunks_batch(self, chunks: List[Dict[str, str]]) -> List[ChunkClassificationResult]:
        """
        Classify multiple chunks in batch
        
        Args:
            chunks: List of dictionaries with 'id' and 'content' keys
            
        Returns:
            List of ChunkClassificationResult objects
        """
        logger.info(f"🎯 Batch classifying {len(chunks)} chunks")
        
        results = []
        
        # Process chunks in parallel batches of 10
        batch_size = 10
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            # Create tasks for parallel processing
            tasks = [self.classify_chunk(chunk['content']) for chunk in batch]
            
            # Execute batch in parallel
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle results and exceptions
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"❌ Failed to classify chunk {batch[j]['id']}: {result}")
                    results.append(ChunkClassificationResult(
                        chunk_type=ChunkType.UNCLASSIFIED,
                        confidence=0.0,
                        metadata={},
                        reasoning=f"Classification failed: {result}"
                    ))
                else:
                    results.append(result)
            
            # Small delay between batches to avoid overwhelming the system
            if i + batch_size < len(chunks):
                await asyncio.sleep(0.1)
        
        logger.info(f"✅ Completed batch classification of {len(chunks)} chunks")
        return results
    
    def _analyze_content_patterns(self, content: str) -> Dict[str, Any]:
        """
        Analyze content patterns to determine chunk type
        
        Args:
            content: The text content to analyze
            
        Returns:
            Dictionary with chunk_type, confidence, and reasoning
        """
        content_lower = content.lower()
        content_length = len(content)
        
        # Product Description patterns
        if self._is_product_description(content):
            return {
                'chunk_type': ChunkType.PRODUCT_DESCRIPTION,
                'confidence': 0.85,
                'reasoning': 'Contains product name, description, and key features'
            }
        
        # Technical Specs patterns
        if self._is_technical_specs(content):
            return {
                'chunk_type': ChunkType.TECHNICAL_SPECS,
                'confidence': 0.90,
                'reasoning': 'Contains technical specifications, measurements, or detailed properties'
            }
        
        # Visual Showcase patterns
        if self._is_visual_showcase(content):
            return {
                'chunk_type': ChunkType.VISUAL_SHOWCASE,
                'confidence': 0.80,
                'reasoning': 'Contains visual descriptions, image references, or style elements'
            }
        
        # Designer Story patterns
        if self._is_designer_story(content):
            return {
                'chunk_type': ChunkType.DESIGNER_STORY,
                'confidence': 0.85,
                'reasoning': 'Contains designer information, philosophy, or creative process'
            }
        
        # Collection Overview patterns
        if self._is_collection_overview(content):
            return {
                'chunk_type': ChunkType.COLLECTION_OVERVIEW,
                'confidence': 0.80,
                'reasoning': 'Contains collection information, themes, or overview content'
            }
        
        # Index Content patterns
        if self._is_index_content(content):
            return {
                'chunk_type': ChunkType.INDEX_CONTENT,
                'confidence': 0.95,
                'reasoning': 'Contains table of contents, index, or navigation elements'
            }
        
        # Sustainability Info patterns
        if self._is_sustainability_info(content):
            return {
                'chunk_type': ChunkType.SUSTAINABILITY_INFO,
                'confidence': 0.90,
                'reasoning': 'Contains sustainability, environmental, or eco-friendly information'
            }
        
        # Certification Info patterns
        if self._is_certification_info(content):
            return {
                'chunk_type': ChunkType.CERTIFICATION_INFO,
                'confidence': 0.90,
                'reasoning': 'Contains certification, compliance, or quality assurance information'
            }
        
        # Supporting Content (default for other content)
        if content_length > 50:
            return {
                'chunk_type': ChunkType.SUPPORTING_CONTENT,
                'confidence': 0.60,
                'reasoning': 'General content that supports the document but doesn\'t fit specific categories'
            }
        
        # Unclassified (very short or unclear content)
        return {
            'chunk_type': ChunkType.UNCLASSIFIED,
            'confidence': 0.30,
            'reasoning': 'Content too short or unclear for classification'
        }
    
    def _is_product_description(self, content: str) -> bool:
        """Check if content represents a product description"""
        content_lower = content.lower()
        
        # Product name patterns (UPPERCASE words)
        has_product_name = bool(re.search(r'\b[A-Z]{2,}\b', content))
        
        # Product description keywords
        product_keywords = [
            'product', 'design', 'collection', 'series', 'line',
            'available in', 'comes in', 'features', 'includes',
            'material', 'finish', 'color', 'size', 'dimension'
        ]
        
        keyword_matches = sum(1 for keyword in product_keywords if keyword in content_lower)
        
        # Dimension patterns (e.g., "15×38", "20×40")
        has_dimensions = bool(re.search(r'\d+\s*[×x]\s*\d+', content))
        
        return has_product_name and (keyword_matches >= 2 or has_dimensions)
    
    def _is_technical_specs(self, content: str) -> bool:
        """Check if content represents technical specifications"""
        content_lower = content.lower()
        
        # Technical specification keywords
        tech_keywords = [
            'specification', 'specs', 'technical', 'properties',
            'dimensions', 'weight', 'capacity', 'performance',
            'material composition', 'thickness', 'density',
            'resistance', 'durability', 'compliance'
        ]
        
        keyword_matches = sum(1 for keyword in tech_keywords if keyword in content_lower)
        
        # Measurement patterns
        has_measurements = bool(re.search(r'\d+\s*(mm|cm|m|kg|g|%|°C|°F)', content))
        
        # Technical formatting (lists, specifications)
        has_list_format = '•' in content or '-' in content or ':' in content
        
        return keyword_matches >= 2 or (has_measurements and has_list_format)
    
    def _is_visual_showcase(self, content: str) -> bool:
        """Check if content represents visual showcase"""
        content_lower = content.lower()
        
        # Visual keywords
        visual_keywords = [
            'image', 'photo', 'visual', 'showcase', 'gallery',
            'moodboard', 'style', 'aesthetic', 'look', 'appearance',
            'color palette', 'texture', 'pattern', 'finish'
        ]
        
        keyword_matches = sum(1 for keyword in visual_keywords if keyword in content_lower)
        
        # Image references
        has_image_refs = ('![' in content or '<img' in content or 
                         'see image' in content_lower or 'shown in' in content_lower)
        
        return keyword_matches >= 2 or has_image_refs
    
    def _is_designer_story(self, content: str) -> bool:
        """Check if content represents designer story"""
        content_lower = content.lower()
        
        # Designer keywords
        designer_keywords = [
            'designer', 'design', 'studio', 'architect', 'creative',
            'inspiration', 'philosophy', 'vision', 'concept',
            'process', 'approach', 'methodology', 'story'
        ]
        
        keyword_matches = sum(1 for keyword in designer_keywords if keyword in content_lower)
        
        # Designer name patterns (often in caps or with studio)
        has_designer_name = (bool(re.search(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', content)) or
                           'studio' in content_lower or
                           'design by' in content_lower)
        
        return keyword_matches >= 3 or (keyword_matches >= 2 and has_designer_name)
    
    def _is_collection_overview(self, content: str) -> bool:
        """Check if content represents collection overview"""
        content_lower = content.lower()
        
        # Collection keywords
        collection_keywords = [
            'collection', 'series', 'line', 'range', 'family',
            'overview', 'introduction', 'presents', 'featuring',
            'includes', 'comprises', 'consists of'
        ]
        
        keyword_matches = sum(1 for keyword in collection_keywords if keyword in content_lower)
        
        # Collection structure indicators
        has_structure = ('•' in content or '-' in content or
                        bool(re.search(r'\d+\s+(products|items|pieces)', content_lower)))
        
        return keyword_matches >= 2 or (keyword_matches >= 1 and has_structure)
    
    def _is_index_content(self, content: str) -> bool:
        """Check if content represents index/navigation content"""
        content_lower = content.lower()

        # Detect multiple product names listed together (index pattern):
        # UPPERCASE words (product names) with minimal text between them.
        uppercase_words = re.findall(r'\b[A-Z]{2,}\b', content)
        if len(uppercase_words) >= 3:  # 3+ product names = likely index
            # Check if they're close together (index pattern)
            lines = content.split('\n')
            short_lines = [l for l in lines if len(l.strip()) < 50 and len(l.strip()) > 0]
            if len(short_lines) >= 3:
                # This looks like an index page with product listings
                return True

        # Detect "by DESIGNER" pattern repeated — multiple "by UPPERCASE"
        # occurrences indicate a product listing.
        by_pattern_count = len(re.findall(r'by\s+[A-Z]+', content))
        if by_pattern_count >= 3:
            return True

        # Detect "COLLECTIONS INDEX" or similar text
        if 'collections index' in content_lower or 'product index' in content_lower:
            return True

        # Detect multiple size patterns (e.g., "20×20 cm") without descriptions —
        # this indicates a spec list rather than actual content.
        size_patterns = re.findall(r'\d+[×x]\d+\s*cm', content)
        if len(size_patterns) >= 3:
            # Check if there's minimal descriptive text
            words = content.split()
            if len(words) < 100:  # Short text with multiple sizes = index
                return True

        # Index keywords
        index_keywords = [
            'table of contents', 'index', 'contents', 'navigation',
            'page', 'section', 'chapter', 'part'
        ]

        keyword_matches = sum(1 for keyword in index_keywords if keyword in content_lower)

        # Page number patterns
        has_page_numbers = (bool(re.search(r'\.\.\.\s*\d+', content)) or
                           bool(re.search(r'page\s+\d+', content, re.IGNORECASE)))

        # List structure with numbers
        has_numbered_list = bool(re.search(r'^\d+\.', content.strip())) or '...' in content

        return keyword_matches >= 1 or has_page_numbers or has_numbered_list
    
    def _is_sustainability_info(self, content: str) -> bool:
        """Check if content represents sustainability information"""
        content_lower = content.lower()
        
        # Sustainability keywords
        sustainability_keywords = [
            'sustainability', 'sustainable', 'eco', 'environmental',
            'green', 'renewable', 'recycled', 'recyclable',
            'carbon footprint', 'eco-friendly', 'biodegradable',
            'energy efficient', 'responsible sourcing'
        ]
        
        keyword_matches = sum(1 for keyword in sustainability_keywords if keyword in content_lower)
        
        return keyword_matches >= 2
    
    def _is_certification_info(self, content: str) -> bool:
        """Check if content represents certification information"""
        content_lower = content.lower()
        
        # Certification keywords
        certification_keywords = [
            'certification', 'certified', 'standard', 'compliance',
            'iso', 'ce mark', 'quality assurance', 'tested',
            'approved', 'meets standards', 'conforms to'
        ]
        
        keyword_matches = sum(1 for keyword in certification_keywords if keyword in content_lower)
        
        # Certification codes (ISO, CE, etc.)
        has_cert_codes = bool(re.search(r'\b(ISO|CE|EN|ASTM|ANSI)\s*\d+', content))
        
        return keyword_matches >= 2 or has_cert_codes

    def _extract_structured_metadata(self, content: str, chunk_type: ChunkType) -> Dict[str, Any]:
        """
        Extract structured metadata based on chunk type

        Args:
            content: The text content to analyze
            chunk_type: The classified chunk type

        Returns:
            Dictionary with extracted metadata
        """
        metadata = {}

        if chunk_type == ChunkType.PRODUCT_DESCRIPTION:
            return self._extract_product_metadata(content)
        elif chunk_type == ChunkType.TECHNICAL_SPECS:
            return self._extract_technical_metadata(content)
        elif chunk_type == ChunkType.VISUAL_SHOWCASE:
            return self._extract_visual_metadata(content)
        elif chunk_type == ChunkType.DESIGNER_STORY:
            return self._extract_designer_metadata(content)
        elif chunk_type == ChunkType.COLLECTION_OVERVIEW:
            return self._extract_collection_metadata(content)
        else:
            return metadata

    def _extract_product_metadata(self, content: str) -> Dict[str, Any]:
        """Extract product-specific metadata"""
        metadata = {}

        # Extract product name (UPPERCASE words)
        product_name_match = re.search(r'\b[A-Z]{2,}(?:\s+[A-Z]{2,})*\b', content)
        if product_name_match:
            metadata['productName'] = product_name_match.group(0)

        # Extract dimensions (e.g., "15×38", "20×40")
        dimension_match = re.search(r'\d+\s*[×x]\s*\d+(?:\s*[×x]\s*\d+)?', content)
        if dimension_match:
            metadata['dimensions'] = dimension_match.group(0)

        # Extract materials
        material_keywords = ['wood', 'metal', 'glass', 'ceramic', 'fabric', 'leather', 'plastic', 'stone', 'concrete']
        found_materials = [material for material in material_keywords if material in content.lower()]
        if found_materials:
            metadata['materials'] = found_materials

        # Extract colors
        color_keywords = ['white', 'black', 'red', 'blue', 'green', 'yellow', 'brown', 'gray', 'grey', 'beige', 'natural']
        found_colors = [color for color in color_keywords if color in content.lower()]
        if found_colors:
            metadata['colors'] = found_colors

        # Extract key features (bullet points or listed items)
        features = self._extract_list_items(content)
        if features:
            metadata['keyFeatures'] = features

        return metadata

    def _extract_technical_metadata(self, content: str) -> Dict[str, Any]:
        """Extract technical specifications metadata"""
        metadata = {}

        # Extract specifications (key: value pairs)
        specifications = {}
        spec_lines = [line for line in content.split('\n') if ':' in line]

        for line in spec_lines:
            parts = line.split(':', 1)
            if len(parts) == 2:
                key, value = parts[0].strip(), parts[1].strip()
                if key and value:
                    specifications[key] = value

        if specifications:
            metadata['specifications'] = specifications

        # Extract measurements
        measurements = {}
        measurement_matches = re.findall(r'\d+\s*(mm|cm|m|kg|g|%|°C|°F)', content)
        if measurement_matches:
            for i, match in enumerate(measurement_matches):
                measurements[f'measurement_{i + 1}'] = match
            metadata['measurements'] = measurements

        # Extract technical details
        technical_details = self._extract_list_items(content)
        if technical_details:
            metadata['technicalDetails'] = technical_details

        return metadata

    def _extract_visual_metadata(self, content: str) -> Dict[str, Any]:
        """Extract visual showcase metadata"""
        metadata = {}

        # Extract image references
        image_refs = []
        img_matches = re.findall(r'!\[([^\]]*)\]', content)
        if img_matches:
            image_refs.extend(img_matches)

        if 'image' in content.lower() or 'photo' in content.lower():
            image_refs.append('Referenced in text')

        if image_refs:
            metadata['imageReferences'] = image_refs

        # Extract visual elements
        visual_keywords = ['color', 'texture', 'pattern', 'finish', 'style', 'aesthetic']
        found_elements = [element for element in visual_keywords if element in content.lower()]
        if found_elements:
            metadata['visualElements'] = found_elements

        # Extract style description
        style_match = re.search(r'style[:\s]+([^.!?]+)', content, re.IGNORECASE)
        if style_match:
            metadata['styleDescription'] = style_match.group(1).strip()

        return metadata

    def _extract_designer_metadata(self, content: str) -> Dict[str, Any]:
        """Extract designer story metadata"""
        metadata = {}

        # Extract designer name
        designer_match = re.search(r'(?:designer?|design by|created by)[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', content, re.IGNORECASE)
        if designer_match:
            metadata['designerName'] = designer_match.group(1)

        # Extract studio name
        studio_match = re.search(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+studio', content, re.IGNORECASE)
        if studio_match:
            metadata['studioName'] = studio_match.group(1) + ' Studio'

        # Extract design philosophy
        philosophy_match = re.search(r'(?:philosophy|vision|approach)[:\s]+([^.!?]+)', content, re.IGNORECASE)
        if philosophy_match:
            metadata['designPhilosophy'] = philosophy_match.group(1).strip()

        # Extract inspiration sources
        inspiration_keywords = ['inspired by', 'inspiration', 'influenced by']
        inspiration_sources = []
        for keyword in inspiration_keywords:
            match = re.search(f'{keyword}[:\s]+([^.!?]+)', content, re.IGNORECASE)
            if match:
                inspiration_sources.append(match.group(1).strip())
        if inspiration_sources:
            metadata['inspirationSources'] = inspiration_sources

        return metadata

    def _extract_collection_metadata(self, content: str) -> Dict[str, Any]:
        """Extract collection overview metadata"""
        metadata = {}

        # Extract collection name
        collection_match = re.search(r'(?:collection|series|line)[:\s]+([A-Z][a-zA-Z\s]+)', content, re.IGNORECASE)
        if collection_match:
            metadata['collectionName'] = collection_match.group(1).strip()

        # Extract collection theme
        theme_match = re.search(r'(?:theme|concept)[:\s]+([^.!?]+)', content, re.IGNORECASE)
        if theme_match:
            metadata['collectionTheme'] = theme_match.group(1).strip()

        # Extract product count
        count_match = re.search(r'(\d+)\s+(?:products|items|pieces)', content, re.IGNORECASE)
        if count_match:
            metadata['productCount'] = int(count_match.group(1))

        # Extract season/year
        season_match = re.search(r'(spring|summer|fall|autumn|winter)\s+(\d{4})', content, re.IGNORECASE)
        if season_match:
            metadata['seasonYear'] = f"{season_match.group(1)} {season_match.group(2)}"

        return metadata

    def _extract_list_items(self, content: str) -> List[str]:
        """Extract list items from content (bullet points, numbered lists, etc.)"""
        items = []

        # Extract bullet points
        bullet_matches = re.findall(r'[•\-\*]\s*([^\n]+)', content)
        if bullet_matches:
            items.extend([match.strip() for match in bullet_matches])

        # Extract numbered lists
        numbered_matches = re.findall(r'\d+\.\s*([^\n]+)', content)
        if numbered_matches:
            items.extend([match.strip() for match in numbered_matches])

        return [item for item in items if item]

    async def classify_document_chunks(self, document_id: str, workspace_id: str) -> Dict[str, Any]:
        """
        Classify all chunks for a document using the Edge Function

        Args:
            document_id: The document ID to classify chunks for
            workspace_id: The workspace ID

        Returns:
            Dictionary with classification results and statistics
        """
        logger.info(f"🎯 Classifying chunks for document: {document_id}")

        try:
            # Call the Edge Function for chunk classification
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.supabase_url}/functions/v1/chunk-type-classification",
                    headers={
                        "Authorization": f"Bearer {self.supabase_service_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "document_id": document_id,
                        "reclassify_all": True
                    },
                    timeout=300.0  # 5 minute timeout for large documents
                )

                if response.status_code != 200:
                    error_text = response.text
                    logger.error(f"❌ Edge Function error: {response.status_code} - {error_text}")
                    return {
                        "success": False,
                        "error": f"Edge Function error: {response.status_code} - {error_text}",
                        "stats": {"classified": 0, "errors": 1, "total": 0}
                    }

                result = response.json()
                logger.info(f"✅ Classification complete: {result.get('stats', {})}")

                return result

        except Exception as error:
            logger.error(f"❌ Error classifying document chunks: {error}")
            return {
                "success": False,
                "error": str(error),
                "stats": {"classified": 0, "errors": 1, "total": 0}
            }

