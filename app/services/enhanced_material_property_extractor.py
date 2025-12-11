"""
Enhanced Material Property Extractor for MIVAA

This service provides sophisticated LLM-based extraction of comprehensive 
material functional properties to match frontend capabilities (60+ properties 
across 9 categories), replacing basic keyword matching with semantic analysis.

Key Features:
- LLM-powered semantic understanding using TogetherAI LLaMA Vision
- Comprehensive property extraction across 9 functional categories
- Structured output matching frontend filter system requirements
- Confidence scoring based on extraction quality
- Integration with existing MIVAA architecture

Author: Roo 🐍 Python Developer
Created: 2025-09-04
"""

import json
import logging
import re
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum

from app.services.supabase_client import get_supabase_client

# Set up logging
logger = logging.getLogger(__name__)


class PropertyExtractionCategory(Enum):
    """Enumeration of functional property categories matching frontend filters."""
    SLIP_SAFETY_RATINGS = "slipSafetyRatings"
    SURFACE_GLOSS_REFLECTIVITY = "surfaceGlossReflectivity" 
    MECHANICAL_PROPERTIES_EXTENDED = "mechanicalPropertiesExtended"
    THERMAL_PROPERTIES = "thermalProperties"
    WATER_MOISTURE_RESISTANCE = "waterMoistureResistance"
    CHEMICAL_HYGIENE_RESISTANCE = "chemicalHygieneResistance"
    ACOUSTIC_ELECTRICAL_PROPERTIES = "acousticElectricalProperties"
    ENVIRONMENTAL_SUSTAINABILITY = "environmentalSustainability"
    DIMENSIONAL_AESTHETIC = "dimensionalAesthetic"


@dataclass
class EnhancedMaterialProperties:
    """Enhanced material properties structure matching frontend filtering capabilities."""
    
    # 🦶 Slip/Safety Ratings
    slip_safety_ratings: Optional[Dict[str, Any]] = None  # R-values, DCOF, certifications
    
    # ✨ Surface Gloss/Reflectivity  
    surface_gloss_reflectivity: Optional[Dict[str, Any]] = None  # Gloss levels, value ranges
    
    # 🔧 Mechanical Properties Extended
    mechanical_properties_extended: Optional[Dict[str, Any]] = None  # Mohs, PEI ratings
    
    # 🌡️ Thermal Properties
    thermal_properties: Optional[Dict[str, Any]] = None  # Conductivity, heat resistance
    
    # 💧 Water/Moisture Resistance
    water_moisture_resistance: Optional[Dict[str, Any]] = None  # Absorption, frost resistance
    
    # 🧪 Chemical/Hygiene Resistance
    chemical_hygiene_resistance: Optional[Dict[str, Any]] = None  # Acid/alkali resistance
    
    # 🔊 Acoustic/Electrical Properties
    acoustic_electrical_properties: Optional[Dict[str, Any]] = None  # NRC, conductivity
    
    # 🌱 Environmental/Sustainability
    environmental_sustainability: Optional[Dict[str, Any]] = None  # LEED, recycled content
    
    # 📏 Dimensional/Aesthetic
    dimensional_aesthetic: Optional[Dict[str, Any]] = None  # Edge types, shade variation
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "slipSafetyRatings": self.slip_safety_ratings,
            "surfaceGlossReflectivity": self.surface_gloss_reflectivity,
            "mechanicalPropertiesExtended": self.mechanical_properties_extended,
            "thermalProperties": self.thermal_properties,
            "waterMoistureResistance": self.water_moisture_resistance,
            "chemicalHygieneResistance": self.chemical_hygiene_resistance,
            "acousticElectricalProperties": self.acoustic_electrical_properties,
            "environmentalSustainability": self.environmental_sustainability,
            "dimensionalAesthetic": self.dimensional_aesthetic
        }


@dataclass 
class PropertyExtractionResult:
    """Result of enhanced material property extraction."""
    enhanced_properties: EnhancedMaterialProperties
    extraction_confidence: float
    property_coverage_percentage: float  # % of expected properties extracted
    processing_time: float
    extraction_method: str
    raw_llm_response: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "enhanced_properties": self.enhanced_properties.to_dict(),
            "extraction_confidence": self.extraction_confidence,
            "property_coverage_percentage": self.property_coverage_percentage,
            "processing_time": self.processing_time,
            "extraction_method": self.extraction_method,
            "raw_llm_response": self.raw_llm_response,
            "timestamp": self.timestamp.isoformat()
        }


class EnhancedMaterialPropertyExtractor:
    """
    Enhanced material property extractor using sophisticated LLM-based semantic analysis.
    
    This class replaces basic keyword matching with comprehensive property extraction
    that leverages TogetherAI's LLaMA Vision model for sophisticated document understanding.
    """
    
    def __init__(self, together_ai_client=None, confidence_threshold: float = 0.7, workspace_id: str = "ffafc28b-1b8b-4b0d-b226-9f9a6154004e"):
        """Initialize the enhanced property extractor.

        Args:
            together_ai_client: TogetherAI service client
            confidence_threshold: Minimum confidence for property extraction
            workspace_id: Workspace ID for loading custom prompts
        """
        self.together_ai_client = together_ai_client
        self.confidence_threshold = confidence_threshold
        self.workspace_id = workspace_id
        self.supabase = get_supabase_client()
        self._setup_property_extractors()
        
    def _setup_property_extractors(self) -> None:
        """Set up category-specific property extraction patterns and prompts."""
        # Try to load prompts from database first
        db_prompts = self._load_prompts_from_database()

        # Map database prompts (by version) to PropertyExtractionCategory enums
        # Version 1 = SLIP_SAFETY_RATINGS, Version 2 = SURFACE_GLOSS_REFLECTIVITY, etc.
        version_to_category = {
            1: PropertyExtractionCategory.SLIP_SAFETY_RATINGS,
            2: PropertyExtractionCategory.SURFACE_GLOSS_REFLECTIVITY,
            3: PropertyExtractionCategory.MECHANICAL_PROPERTIES_EXTENDED,
            4: PropertyExtractionCategory.THERMAL_PROPERTIES,
            5: PropertyExtractionCategory.WATER_MOISTURE_RESISTANCE,
            6: PropertyExtractionCategory.CHEMICAL_HYGIENE_RESISTANCE,
            7: PropertyExtractionCategory.ACOUSTIC_ELECTRICAL_PROPERTIES,
            8: PropertyExtractionCategory.ENVIRONMENTAL_SUSTAINABILITY,
            9: PropertyExtractionCategory.DIMENSIONAL_AESTHETIC
        }

        # Build extraction_prompts dict with database prompts or hardcoded fallbacks
        self.extraction_prompts = {}
        for version, category in version_to_category.items():
            if version in db_prompts:
                self.extraction_prompts[category] = db_prompts[version]
                logger.info(f"✅ Using DATABASE prompt for {category.value} (version {version})")
            else:
                # Fallback to hardcoded prompts
                logger.info(f"⚠️ Using HARDCODED fallback prompt for {category.value}")
                if category == PropertyExtractionCategory.SLIP_SAFETY_RATINGS:
                    self.extraction_prompts[category] = self._create_slip_safety_prompt()
                elif category == PropertyExtractionCategory.SURFACE_GLOSS_REFLECTIVITY:
                    self.extraction_prompts[category] = self._create_gloss_prompt()
                elif category == PropertyExtractionCategory.MECHANICAL_PROPERTIES_EXTENDED:
                    self.extraction_prompts[category] = self._create_mechanical_prompt()
                elif category == PropertyExtractionCategory.THERMAL_PROPERTIES:
                    self.extraction_prompts[category] = self._create_thermal_prompt()
                elif category == PropertyExtractionCategory.WATER_MOISTURE_RESISTANCE:
                    self.extraction_prompts[category] = self._create_water_resistance_prompt()
                elif category == PropertyExtractionCategory.CHEMICAL_HYGIENE_RESISTANCE:
                    self.extraction_prompts[category] = self._create_chemical_prompt()
                elif category == PropertyExtractionCategory.ACOUSTIC_ELECTRICAL_PROPERTIES:
                    self.extraction_prompts[category] = self._create_acoustic_prompt()
                elif category == PropertyExtractionCategory.ENVIRONMENTAL_SUSTAINABILITY:
                    self.extraction_prompts[category] = self._create_environmental_prompt()
                elif category == PropertyExtractionCategory.DIMENSIONAL_AESTHETIC:
                    self.extraction_prompts[category] = self._create_aesthetic_prompt()

    def _load_prompts_from_database(self) -> Dict[int, str]:
        """Load all material property prompts from database.

        Returns:
            Dict mapping version number to prompt template
        """
        try:
            result = self.supabase.client.table('prompts')\
                .select('prompt_text, version')\
                .eq('workspace_id', self.workspace_id)\
                .eq('prompt_type', 'extraction')\
                .eq('stage', 'entity_creation')\
                .eq('category', 'material_properties')\
                .eq('is_custom', False)\
                .execute()

            if result.data and len(result.data) > 0:
                prompts_by_version = {}
                for row in result.data:
                    version = row['version']
                    prompt = row['prompt_text']
                    prompts_by_version[version] = prompt

                logger.info(f"✅ Loaded {len(prompts_by_version)} material property prompts from database")
                return prompts_by_version
            else:
                logger.warning("⚠️ No material property prompts found in database, using hardcoded fallbacks")
                return {}

        except Exception as e:
            logger.error(f"❌ Failed to load prompts from database: {e}")
            return {}
        
    def _create_slip_safety_prompt(self) -> str:
        """Create specialized prompt for slip/safety property extraction."""
        return """
        Analyze this material document for slip resistance and safety properties. Extract the following:
        
        SLIP SAFETY RATINGS:
        1. R-Value (DIN 51130): Look for R9, R10, R11, R12, R13 ratings
        2. Barefoot Ramp Test (DIN 51097): Look for Class A, B, or C ratings  
        3. DCOF Range: Dynamic Coefficient of Friction values (0.0-1.0, ≥0.42 recommended)
        4. Pendulum Test Range (PTV): Pendulum Test Values (0-100)
        5. Safety Certifications: ANSI A137.1, DIN 51130, DIN 51097, BS 7976, AS/NZS 4586
        
        Return as JSON:
        {
            "rValue": ["R10", "R11"], // found R-values
            "barefootRampTest": ["A", "B"], // found classifications  
            "dcofRange": [0.45, 0.62], // min/max values if found
            "pendulumTestRange": [25, 45], // min/max PTV if found
            "safetyCertifications": ["DIN 51130"], // found certifications
            "confidence": 0.85 // extraction confidence 0.0-1.0
        }
        """
        
    def _create_gloss_prompt(self) -> str:
        """Create specialized prompt for surface gloss/reflectivity extraction."""
        return """
        Analyze this material document for surface gloss and reflectivity properties:
        
        SURFACE GLOSS/REFLECTIVITY:
        1. Gloss Level: super-polished, polished, satin, semi-polished, matte, velvet, anti-glare
        2. Gloss Value Range: Numerical values 0-100 (gloss meter readings)
        3. Surface Finish descriptions and specifications
        
        Return as JSON:
        {
            "glossLevel": ["polished", "satin"], // detected gloss levels
            "glossValueRange": [15, 35], // min/max gloss values if specified
            "confidence": 0.90
        }
        """
        
    def _create_mechanical_prompt(self) -> str:
        """Create specialized prompt for mechanical properties extraction."""  
        return """
        Analyze this material document for mechanical properties:
        
        MECHANICAL PROPERTIES:
        1. Mohs Hardness: Scale 1-10 (1=talc, 10=diamond)
        2. PEI Rating: Abrasion resistance Class 0-5 (porcelain enamel institute)
        3. Tensile strength, compressive strength, elastic modulus if mentioned
        4. Durability ratings and wear resistance classifications
        
        Return as JSON:
        {
            "mohsHardnessRange": [6.5, 7.0], // hardness range if found
            "peiRating": [3, 4], // PEI classes if found  
            "tensileStrength": 45.5, // MPa if specified
            "compressiveStrength": 120.0, // MPa if specified
            "confidence": 0.88
        }
        """
        
    def _create_thermal_prompt(self) -> str:
        """Create specialized prompt for thermal properties extraction."""
        return """
        Analyze this material document for thermal properties:
        
        THERMAL PROPERTIES:
        1. Thermal Conductivity: W/mK values (0-10+ range)
        2. Heat Resistance: Temperature range in °C (0-500°C+)
        3. Radiant Heating Compatibility: Yes/No/Compatible
        4. Thermal expansion coefficient if mentioned
        5. Fire resistance ratings or classifications
        
        Return as JSON:
        {
            "thermalConductivityRange": [0.8, 1.2], // W/mK if found
            "heatResistanceRange": [200, 300], // °C min/max if found
            "radiantHeatingCompatible": true, // compatibility
            "thermalExpansion": 8.6e-6, // coefficient if found
            "confidence": 0.82
        }
        """
        
    def _create_water_resistance_prompt(self) -> str:
        """Create specialized prompt for water/moisture resistance extraction."""
        return """
        Analyze this material document for water and moisture resistance:
        
        WATER/MOISTURE RESISTANCE:
        1. Water Absorption Range: Percentage 0-20%
        2. Frost Resistance: Yes/No/Rated
        3. Mold/Mildew Resistance: Yes/No/Treated
        4. Porosity levels and permeability ratings
        5. Waterproof vs water-resistant classifications
        
        Return as JSON:
        {
            "waterAbsorptionRange": [0.1, 0.5], // % range if found
            "frostResistance": true, // frost resistant
            "moldMildewResistant": true, // mold resistant  
            "porosity": "low", // porosity level if mentioned
            "confidence": 0.91
        }
        """
        
    def _create_chemical_prompt(self) -> str:
        """Create specialized prompt for chemical/hygiene resistance extraction."""
        return """
        Analyze this material document for chemical and hygiene resistance:
        
        CHEMICAL/HYGIENE RESISTANCE:
        1. Acid Resistance Level: low, medium, high, excellent
        2. Alkali Resistance Level: low, medium, high, excellent  
        3. Stain Resistance Class: Class 1-5 ratings
        4. Food Safe Certification: FDA approved, food contact safe
        5. Chemical compatibility with cleaners, solvents
        
        Return as JSON:
        {
            "acidResistance": ["high"], // resistance levels
            "alkaliResistance": ["medium", "high"], // resistance levels
            "stainResistanceClass": [4, 5], // class ratings if found
            "foodSafeCertified": true, // food safety status
            "confidence": 0.86
        }
        """
        
    def _create_acoustic_prompt(self) -> str:
        """Create specialized prompt for acoustic/electrical properties extraction."""
        return """
        Analyze this material document for acoustic and electrical properties:
        
        ACOUSTIC/ELECTRICAL PROPERTIES:
        1. NRC Range: Noise Reduction Coefficient 0.0-1.0
        2. Anti-Static Properties: Yes/No/ESD safe
        3. Electrical Conductivity: Conductive/Non-conductive/Semi-conductive
        4. Sound absorption coefficients and ratings
        5. EMI/EMC shielding properties if applicable
        
        Return as JSON:
        {
            "nrcRange": [0.15, 0.25], // NRC values if found
            "antiStatic": true, // anti-static properties
            "conductive": false, // electrical conductivity
            "soundAbsorption": 0.22, // coefficient if specified
            "confidence": 0.79
        }
        """
        
    def _create_environmental_prompt(self) -> str:
        """Create specialized prompt for environmental/sustainability extraction."""
        return """
        Analyze this material document for environmental and sustainability properties:
        
        ENVIRONMENTAL/SUSTAINABILITY:
        1. Greenguard Certification: certified, gold, none
        2. Total Recycled Content Range: Percentage 0-100%
        3. LEED Credits Range: Available credits 0-10
        4. VOC emissions, off-gassing properties
        5. Carbon footprint, lifecycle assessment data
        6. Recyclability and end-of-life properties
        
        Return as JSON:
        {
            "greenguardLevel": ["certified"], // certification levels
            "totalRecycledContentRange": [25, 40], // % recycled content
            "leedCreditsRange": [2, 4], // LEED credits available
            "vocEmissions": "low", // emission levels
            "confidence": 0.84
        }
        """
        
    def _create_aesthetic_prompt(self) -> str:
        """Create specialized prompt for dimensional/aesthetic properties extraction."""
        return """
        Analyze this material document for dimensional and aesthetic properties:
        
        DIMENSIONAL/AESTHETIC:
        1. Rectified Edges: Yes/No/Available
        2. Texture Rating Range: If texture is rated/classified
        3. Shade Variation: V1 (Uniform), V2 (Slight), V3 (Moderate), V4 (Substantial)
        4. Dimensional stability and tolerances
        5. Color consistency and variation specifications
        
        Return as JSON:
        {
            "rectifiedEdges": true, // edge treatment
            "textureRatingRange": true, // if texture is rated
            "shadeVariation": ["V2", "V3"], // variation classifications
            "dimensionalStability": "high", // stability rating
            "confidence": 0.88
        }
        """

    async def extract_comprehensive_properties(
        self, 
        analysis_text: str,
        document_context: Optional[str] = None,
        extraction_categories: Optional[List[PropertyExtractionCategory]] = None
    ) -> PropertyExtractionResult:
        """
        Extract comprehensive material properties using enhanced LLM-based analysis.
        
        This method replaces the basic keyword matching in _parse_analysis_response
        with sophisticated semantic understanding across all property categories.
        
        Args:
            analysis_text: Raw text from document analysis
            document_context: Additional context about the document
            extraction_categories: Specific categories to extract (None = all)
            
        Returns:
            PropertyExtractionResult with comprehensive property data
        """
        start_time = time.time()
        
        # Default to extracting all categories
        if extraction_categories is None:
            extraction_categories = list(PropertyExtractionCategory)
            
        logger.info(f"Starting enhanced property extraction for {len(extraction_categories)} categories")
        
        enhanced_properties = EnhancedMaterialProperties()
        category_confidences = []
        successful_extractions = 0
        total_expected_properties = len(extraction_categories)
        
        try:
            # Process each category with specialized extraction
            for category in extraction_categories:
                try:
                    category_result = await self._extract_category_properties(
                        category, analysis_text, document_context
                    )
                    
                    if category_result and category_result.get("confidence", 0) >= self.confidence_threshold:
                        self._apply_category_result(enhanced_properties, category, category_result)
                        category_confidences.append(category_result["confidence"])
                        successful_extractions += 1
                        
                        logger.debug(f"Successfully extracted {category.value} properties")
                    else:
                        logger.warning(f"Low confidence extraction for {category.value}")
                        category_confidences.append(0.0)
                        
                except Exception as e:
                    logger.error(f"Failed to extract {category.value} properties: {e}")
                    category_confidences.append(0.0)
                    
            # Calculate overall metrics
            processing_time = time.time() - start_time
            overall_confidence = sum(category_confidences) / len(category_confidences) if category_confidences else 0.0
            coverage_percentage = (successful_extractions / total_expected_properties) * 100
            
            logger.info(f"Enhanced extraction completed: {coverage_percentage:.1f}% coverage, {overall_confidence:.3f} confidence")
            
            return PropertyExtractionResult(
                enhanced_properties=enhanced_properties,
                extraction_confidence=overall_confidence,
                property_coverage_percentage=coverage_percentage,
                processing_time=processing_time,
                extraction_method="llm_semantic_analysis",
                raw_llm_response=analysis_text[:1000] + "..." if len(analysis_text) > 1000 else analysis_text
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Enhanced property extraction failed after {processing_time:.2f}s: {e}")
            raise
            
    async def _extract_category_properties(
        self, 
        category: PropertyExtractionCategory,
        analysis_text: str, 
        context: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Extract properties for a specific category using targeted LLM analysis.
        
        This method uses category-specific prompts to guide the LLM toward 
        extracting relevant technical specifications and properties.
        """
        try:
            # Prepare category-specific extraction prompt
            system_prompt = self.extraction_prompts.get(category)
            if not system_prompt:
                logger.warning(f"No extraction prompt found for category: {category.value}")
                return None
                
            # Combine document text with context for comprehensive analysis
            full_context = f"Document Content:\n{analysis_text}"
            if context:
                full_context += f"\n\nAdditional Context:\n{context}"
                
            # Use TogetherAI for sophisticated semantic analysis
            if self.together_ai_client:
                extraction_result = await self._llm_property_extraction(
                    system_prompt, full_context, category
                )
            else:
                # Fallback to enhanced rule-based extraction
                extraction_result = self._enhanced_rule_based_extraction(
                    category, analysis_text, context
                )
                
            return extraction_result
            
        except Exception as e:
            logger.error(f"Category extraction failed for {category.value}: {e}")
            return None
            
    async def _llm_property_extraction(
        self,
        system_prompt: str,
        document_content: str,
        category: PropertyExtractionCategory
    ) -> Optional[Dict[str, Any]]:
        """
        Perform LLM-based property extraction using TogetherAI.
        
        This method leverages the LLaMA Vision model for sophisticated
        semantic understanding of technical material specifications.
        """
        try:
            # Construct focused analysis prompt
            analysis_prompt = f"""
            {system_prompt}
            
            DOCUMENT TO ANALYZE:
            {document_content}
            
            IMPORTANT: 
            - Return ONLY valid JSON as specified above
            - If a property is not clearly mentioned, omit it from the JSON
            - Set confidence based on clarity and specificity of found information
            - Higher confidence (0.8-1.0) for explicit technical specifications
            - Lower confidence (0.4-0.7) for implied or general mentions
            - Very low confidence (0.1-0.3) for uncertain or ambiguous references
            """
            
            # Call TogetherAI for semantic analysis
            if hasattr(self.together_ai_client, 'analyze_semantic_content'):
                response = await self.together_ai_client.analyze_semantic_content({
                    "content": analysis_prompt,
                    "analysis_type": "property_extraction",
                    "category": category.value
                })
                
                # Parse LLM response into structured format
                return self._parse_llm_response(response, category)
            else:
                logger.warning("TogetherAI client not properly configured for property extraction")
                return None
                
        except Exception as e:
            logger.error(f"LLM property extraction failed for {category.value}: {e}")
            return None
            
    def _parse_llm_response(
        self, 
        llm_response: Any, 
        category: PropertyExtractionCategory
    ) -> Optional[Dict[str, Any]]:
        """
        Parse LLM response into structured property data.
        
        This method handles the conversion of LLM output into the standardized
        property format expected by the frontend filtering system.
        """
        try:
            # Handle different response formats from TogetherAI
            response_text = ""
            if hasattr(llm_response, 'description'):
                response_text = llm_response.description
            elif isinstance(llm_response, dict) and 'content' in llm_response:
                response_text = llm_response['content']
            elif isinstance(llm_response, str):
                response_text = llm_response
            else:
                logger.warning(f"Unexpected LLM response format for {category.value}")
                return None
                
            # Extract JSON from LLM response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    parsed_data = json.loads(json_str)
                    
                    # Validate and sanitize the parsed data
                    validated_data = self._validate_category_data(parsed_data, category)
                    return validated_data
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON for {category.value}: {e}")
                    # Fallback to text-based extraction
                    return self._extract_from_text(response_text, category)
            else:
                logger.warning(f"No JSON found in LLM response for {category.value}")
                return self._extract_from_text(response_text, category)
                
        except Exception as e:
            logger.error(f"LLM response parsing failed for {category.value}: {e}")
            return None
            
    def _validate_category_data(
        self, 
        data: Dict[str, Any], 
        category: PropertyExtractionCategory
    ) -> Dict[str, Any]:
        """
        Validate and sanitize extracted property data for a specific category.
        
        This ensures the data format matches frontend expectations and 
        handles edge cases or malformed extractions.
        """
        validated = {}
        confidence = data.get("confidence", 0.5)
        
        if category == PropertyExtractionCategory.SLIP_SAFETY_RATINGS:
            # Validate R-Values
            if "rValue" in data and isinstance(data["rValue"], list):
                valid_r_values = [r for r in data["rValue"] if r in ["R9", "R10", "R11", "R12", "R13"]]
                if valid_r_values:
                    validated["rValue"] = valid_r_values
                    
            # Validate DCOF range
            if "dcofRange" in data and isinstance(data["dcofRange"], list) and len(data["dcofRange"]) == 2:
                dcof_min, dcof_max = data["dcofRange"]
                if 0 <= dcof_min <= dcof_max <= 1:
                    validated["dcofRange"] = [float(dcof_min), float(dcof_max)]
                    
            # Validate other slip safety properties...
            
        elif category == PropertyExtractionCategory.MECHANICAL_PROPERTIES_EXTENDED:
            # Validate Mohs hardness
            if "mohsHardnessRange" in data and isinstance(data["mohsHardnessRange"], list):
                mohs_values = data["mohsHardnessRange"]
                if len(mohs_values) == 2 and 1 <= min(mohs_values) <= max(mohs_values) <= 10:
                    validated["mohsHardnessRange"] = [float(v) for v in mohs_values]
                    
            # Validate PEI ratings
            if "peiRating" in data and isinstance(data["peiRating"], list):
                valid_pei = [p for p in data["peiRating"] if isinstance(p, int) and 0 <= p <= 5]
                if valid_pei:
                    validated["peiRating"] = valid_pei
                    
        # Add validation for other categories...
        
        validated["confidence"] = max(0.0, min(1.0, float(confidence)))
        return validated
        
    def _enhanced_rule_based_extraction(
        self, 
        category: PropertyExtractionCategory,
        analysis_text: str, 
        context: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Enhanced rule-based extraction as fallback when LLM is unavailable.
        
        This provides more sophisticated pattern matching than the original
        keyword lists, using regex and contextual analysis.
        """
        text_lower = analysis_text.lower()
        result = {"confidence": 0.0}
        
        if category == PropertyExtractionCategory.SLIP_SAFETY_RATINGS:
            # Enhanced R-value detection
            r_pattern = r'r[-\s]?(?:value|rating)?\s*[:\-]?\s*(r?(?:9|10|11|12|13))'
            r_matches = re.findall(r_pattern, text_lower, re.IGNORECASE)
            if r_matches:
                r_values = [f"R{m.replace('r', '')}" for m in r_matches]
                result["rValue"] = list(set(r_values))  # Remove duplicates
                result["confidence"] = 0.75
                
            # DCOF pattern detection
            dcof_pattern = r'dcof[:\s]*([0-9]+\.?[0-9]*)'
            dcof_matches = re.findall(dcof_pattern, text_lower)
            if dcof_matches:
                dcof_values = [float(m) for m in dcof_matches if 0 <= float(m) <= 1]
                if dcof_values:
                    result["dcofRange"] = [min(dcof_values), max(dcof_values)]
                    result["confidence"] = max(result["confidence"], 0.7)
                    
        elif category == PropertyExtractionCategory.MECHANICAL_PROPERTIES_EXTENDED:
            # Mohs hardness pattern
            mohs_pattern = r'mohs[:\s]+(?:hardness[:\s]+)?([0-9]+\.?[0-9]*)'
            mohs_matches = re.findall(mohs_pattern, text_lower)
            if mohs_matches:
                mohs_values = [float(m) for m in mohs_matches if 1 <= float(m) <= 10]
                if mohs_values:
                    result["mohsHardnessRange"] = [min(mohs_values), max(mohs_values)]
                    result["confidence"] = 0.8
                    
            # PEI rating pattern
            pei_pattern = r'pei[:\s]+(?:rating[:\s]+)?(?:class[:\s]+)?([0-5])'
            pei_matches = re.findall(pei_pattern, text_lower)
            if pei_matches:
                pei_values = [int(m) for m in pei_matches if 0 <= int(m) <= 5]
                if pei_values:
                    result["peiRating"] = max(pei_values)  # Use highest PEI rating found
                    result["confidence"] = 0.8

        return result


# Utility functions for integration with existing MIVAA services

def create_enhanced_extractor(together_ai_client=None) -> EnhancedMaterialPropertyExtractor:
    """Factory function to create an enhanced material property extractor."""
    return EnhancedMaterialPropertyExtractor(
        together_ai_client=together_ai_client,
        confidence_threshold=0.7
    )


async def extract_enhanced_properties_from_analysis(
    analysis_text: str,
    together_ai_client=None,
    document_context: Optional[str] = None
) -> PropertyExtractionResult:
    """
    Convenience function for extracting enhanced properties from analysis text.
    
    This provides a simple interface for integrating enhanced property 
    extraction into existing MIVAA workflows.
    """
    extractor = create_enhanced_extractor(together_ai_client)
    return await extractor.extract_comprehensive_properties(
        analysis_text=analysis_text,
        document_context=document_context
    )


def convert_to_legacy_format(extraction_result: PropertyExtractionResult) -> Dict[str, Any]:
    """
    Convert enhanced extraction result to legacy material_properties format.
    
    This ensures backward compatibility with existing MIVAA components.
    """
    enhanced_props = extraction_result.enhanced_properties
    
    # Map enhanced properties to legacy format for backward compatibility
    legacy_format = {}
    
    # Determine material family from enhanced analysis
    if enhanced_props.mechanical_properties_extended:
        legacy_format['material_family'] = 'advanced_material'
    elif enhanced_props.dimensional_aesthetic:
        legacy_format['material_family'] = 'building_material'
    else:
        legacy_format['material_family'] = 'unknown'
        
    # Extract surface textures from enhanced gloss analysis
    surface_textures = []
    if enhanced_props.surface_gloss_reflectivity:
        gloss_levels = enhanced_props.surface_gloss_reflectivity.get('glossLevel', [])
        surface_textures.extend([level.replace('-', '_') for level in gloss_levels])
    if enhanced_props.dimensional_aesthetic:
        if enhanced_props.dimensional_aesthetic.get('textureRatingRange'):
            surface_textures.append('rated_texture')
    if surface_textures:
        legacy_format['surface_textures'] = surface_textures
        
    # Add comprehensive property summary for monitoring
    legacy_format['enhanced_extraction'] = {
        'coverage_percentage': extraction_result.property_coverage_percentage,
        'extraction_confidence': extraction_result.extraction_confidence,
        'categories_extracted': len([k for k, v in enhanced_props.to_dict().items() if v]),
        'extraction_method': extraction_result.extraction_method
    }
    
    return legacy_format
            
    async def _extract_category_properties(
        self, 
        category: PropertyExtractionCategory,
        analysis_text: str, 
        context: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Extract properties for a specific category using targeted LLM analysis."""
        try:
            # Prepare category-specific extraction prompt
            system_prompt = self.extraction_prompts.get(category)
            if not system_prompt:
                logger.warning(f"No extraction prompt found for category: {category.value}")
                return None
                
            # Combine document text with context for comprehensive analysis
            full_context = f"Document Content:\n{analysis_text}"
            if context:
                full_context += f"\n\nAdditional Context:\n{context}"
                
            # Use TogetherAI for sophisticated semantic analysis
            if self.together_ai_client:
                extraction_result = await self._llm_property_extraction(
                    system_prompt, full_context, category
                )
            else:
                # Fallback to enhanced rule-based extraction
                extraction_result = self._enhanced_rule_based_extraction(
                    category, analysis_text, context
                )
                
            return extraction_result
            
        except Exception as e:
            logger.error(f"Category extraction failed for {category.value}: {e}")
            return None
            
    async def _llm_property_extraction(
        self,
        system_prompt: str,
        document_content: str,
        category: PropertyExtractionCategory
    ) -> Optional[Dict[str, Any]]:
        """Perform LLM-based property extraction using TogetherAI."""
        try:
            # Construct focused analysis prompt
            analysis_prompt = f"""
            {system_prompt}
            
            DOCUMENT TO ANALYZE:
            {document_content}
            
            IMPORTANT: 
            - Return ONLY valid JSON as specified above
            - If a property is not clearly mentioned, omit it from the JSON
            - Set confidence based on clarity and specificity of found information
            - Higher confidence (0.8-1.0) for explicit technical specifications
            - Lower confidence (0.4-0.7) for implied or general mentions
            - Very low confidence (0.1-0.3) for uncertain or ambiguous references
            """
            
            # Call TogetherAI for semantic analysis
            if hasattr(self.together_ai_client, 'analyze_semantic_content'):
                response = await self.together_ai_client.analyze_semantic_content({
                    "content": analysis_prompt,
                    "analysis_type": "property_extraction",
                    "category": category.value
                })
                
                # Parse LLM response into structured format
                return self._parse_llm_response(response, category)
            else:
                logger.warning("TogetherAI client not properly configured for property extraction")
                return None
                
        except Exception as e:
            logger.error(f"LLM property extraction failed for {category.value}: {e}")
            return None
            
    def _parse_llm_response(
        self, 
        llm_response: Any, 
        category: PropertyExtractionCategory
    ) -> Optional[Dict[str, Any]]:
        """Parse LLM response into structured property data."""
        try:
            # Handle different response formats from TogetherAI
            response_text = ""
            if hasattr(llm_response, 'description'):
                response_text = llm_response.description
            elif isinstance(llm_response, dict) and 'content' in llm_response:
                response_text = llm_response['content']
            elif isinstance(llm_response, str):
                response_text = llm_response
            else:
                logger.warning(f"Unexpected LLM response format for {category.value}")
                return None
                
            # Extract JSON from LLM response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    parsed_data = json.loads(json_str)
                    
                    # Validate and sanitize the parsed data
                    validated_data = self._validate_category_data(parsed_data, category)
                    return validated_data
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON for {category.value}: {e}")
                    # Fallback to text-based extraction
                    return self._extract_from_text(response_text, category)
            else:
                logger.warning(f"No JSON found in LLM response for {category.value}")
                return self._extract_from_text(response_text, category)
                
        except Exception as e:
            logger.error(f"LLM response parsing failed for {category.value}: {e}")
            return None
            
    def _validate_category_data(
        self, 
        data: Dict[str, Any], 
        category: PropertyExtractionCategory
    ) -> Dict[str, Any]:
        """Validate and sanitize extracted property data for a specific category."""
        validated = {}
        confidence = data.get("confidence", 0.5)
        
        if category == PropertyExtractionCategory.SLIP_SAFETY_RATINGS:
            # Validate R-Values
            if "rValue" in data and isinstance(data["rValue"], list):
                valid_r_values = [r for r in data["rValue"] if r in ["R9", "R10", "R11", "R12", "R13"]]
                if valid_r_values:
                    validated["rValue"] = valid_r_values
                    
            # Validate DCOF range
            if "dcofRange" in data and isinstance(data["dcofRange"], list) and len(data["dcofRange"]) == 2:
                dcof_min, dcof_max = data["dcofRange"]
                if 0 <= dcof_min <= dcof_max <= 1:
                    validated["dcofRange"] = [float(dcof_min), float(dcof_max)]
                    
        elif category == PropertyExtractionCategory.MECHANICAL_PROPERTIES_EXTENDED:
            # Validate Mohs hardness
            if "mohsHardnessRange" in data and isinstance(data["mohsHardnessRange"], list):
                mohs_values = data["mohsHardnessRange"]
                if len(mohs_values) == 2 and 1 <= min(mohs_values) <= max(mohs_values) <= 10:
                    validated["mohsHardnessRange"] = [float(v) for v in mohs_values]
                    
            # Validate PEI ratings
            if "peiRating" in data and isinstance(data["peiRating"], list):
                valid_pei = [p for p in data["peiRating"] if isinstance(p, int) and 0 <= p <= 5]
                if valid_pei:
                    validated["peiRating"] = valid_pei
                    
        # Add validation for other categories as needed...
        
        validated["confidence"] = max(0.0, min(1.0, float(confidence)))
        return validated
        
    def _enhanced_rule_based_extraction(
        self, 
        category: PropertyExtractionCategory,
        analysis_text: str, 
        context: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Enhanced rule-based extraction as fallback when LLM is unavailable."""
        text_lower = analysis_text.lower()
        result = {"confidence": 0.0}
        
        if category == PropertyExtractionCategory.SLIP_SAFETY_RATINGS:
            # Enhanced R-value detection
            r_pattern = r'r[-\s]?(?:value|rating)?\s*[:\-]?\s*(r?(?:9|10|11|12|13))'
            r_matches = re.findall(r_pattern, text_lower, re.IGNORECASE)
            if r_matches:
                r_values = [f"R{m.replace('r', '')}" for m in r_matches]
                result["rValue"] = list(set(r_values))  # Remove duplicates
                result["confidence"] = 0.75
                
            # DCOF pattern detection
            dcof_pattern = r'dcof[:\s]*([0-9]+\.?[0-9]*)'
            dcof_matches = re.findall(dcof_pattern, text_lower)
            if dcof_matches:
                dcof_values = [float(m) for m in dcof_matches if 0 <= float(m) <= 1]
                if dcof_values:
                    result["dcofRange"] = [min(dcof_values), max(dcof_values)]
                    result["confidence"] = max(result["confidence"], 0.7)
                    
        elif category == PropertyExtractionCategory.MECHANICAL_PROPERTIES_EXTENDED:
            # Mohs hardness pattern
            mohs_pattern = r'mohs[:\s]+(?:hardness[:\s]+)?([0-9]+\.?[0-9]*)'
            mohs_matches = re.findall(mohs_pattern, text_lower)
            if mohs_matches:
                mohs_values = [float(m) for m in mohs_matches if 1 <= float(m) <= 10]
                if mohs_values:
                    result["mohsHardnessRange"] = [min(mohs_values), max(mohs_values)]
                    result["confidence"] = 0.8
                    
            # PEI rating pattern
            pei_pattern = r'pei[:\s]+(?:rating[:\s]+)?(?:class[:\s]+)?([0-5])'
            pei_matches = re.findall(pei_pattern, text_lower)
            if pei_matches:
                pei_values = [int(m) for m in pei_matches if m.isdigit()]
                if pei_values:
                    result["peiRating"] = list(set(pei_values))
                    result["confidence"] = max(result["confidence"], 0.75)
                    
        # Add enhanced patterns for other categories as needed...
        
        return result if result["confidence"] > 0 else None
        
    def _apply_category_result(
        self, 
        enhanced_properties: EnhancedMaterialProperties,
        category: PropertyExtractionCategory,
        category_result: Dict[str, Any]
    ) -> None:
        """Apply extracted category results to the enhanced properties structure."""
        # Remove confidence from result before applying
        result_data = {k: v for k, v in category_result.items() if k != "confidence"}
        
        if category == PropertyExtractionCategory.SLIP_SAFETY_RATINGS:
            enhanced_properties.slip_safety_ratings = result_data
        elif category == PropertyExtractionCategory.SURFACE_GLOSS_REFLECTIVITY:
            enhanced_properties.surface_gloss_reflectivity = result_data
        elif category == PropertyExtractionCategory.MECHANICAL_PROPERTIES_EXTENDED:
            enhanced_properties.mechanical_properties_extended = result_data
        elif category == PropertyExtractionCategory.THERMAL_PROPERTIES:
            enhanced_properties.thermal_properties = result_data
        elif category == PropertyExtractionCategory.WATER_MOISTURE_RESISTANCE:
            enhanced_properties.water_moisture_resistance = result_data
        elif category == PropertyExtractionCategory.CHEMICAL_HYGIENE_RESISTANCE:
            enhanced_properties.chemical_hygiene_resistance = result_data
        elif category == PropertyExtractionCategory.ACOUSTIC_ELECTRICAL_PROPERTIES:
            enhanced_properties.acoustic_electrical_properties = result_data
        elif category == PropertyExtractionCategory.ENVIRONMENTAL_SUSTAINABILITY:
            enhanced_properties.environmental_sustainability = result_data
        elif category == PropertyExtractionCategory.DIMENSIONAL_AESTHETIC:
            enhanced_properties.dimensional_aesthetic = result_data
            
        logger.debug(f"Applied {category.value} results: {len(result_data)} properties")
        
    def _extract_from_text(
        self, 
        text: str, 
        category: PropertyExtractionCategory
    ) -> Optional[Dict[str, Any]]:
        """Fallback text-based extraction when JSON parsing fails."""
        # Use enhanced rule-based extraction as fallback
        return self._enhanced_rule_based_extraction(category, text, None)
        
    def get_expected_properties_count(self) -> int:
        """Return the expected number of extractable properties across all categories."""
        return 60  # Based on frontend capability analysis
        
    def get_supported_categories(self) -> List[str]:
        """Return list of supported property extraction categories."""
        return [cat.value for cat in PropertyExtractionCategory]
        
    def calculate_coverage_score(
        self, 
        extraction_result: PropertyExtractionResult
    ) -> Dict[str, Any]:
        """Calculate detailed coverage score for the extraction result."""
        coverage_details = {
            "total_categories": len(PropertyExtractionCategory),
            "extracted_categories": 0,
            "category_breakdown": {},
            "overall_coverage_percentage": extraction_result.property_coverage_percentage,
            "confidence_score": extraction_result.extraction_confidence
        }
        
        properties_dict = extraction_result.enhanced_properties.to_dict()
        
        for category in PropertyExtractionCategory:
            category_data = properties_dict.get(category.value)
            if category_data and any(v is not None for v in category_data.values()):
                coverage_details["extracted_categories"] += 1
                coverage_details["category_breakdown"][category.value] = {
                    "extracted": True,
                    "property_count": len([k for k, v in category_data.items() if v is not None])
                }
            else:
                coverage_details["category_breakdown"][category.value] = {
                    "extracted": False,
                    "property_count": 0
                }
                
        return coverage_details