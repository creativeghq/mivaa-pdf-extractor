"""
Canonical Metadata Schema Service

Implements comprehensive metadata extraction for products including:
- Core Identity (name, collection, designer, brand)
- Physical Properties (dimensions, material, weight)
- Visual Properties (colors, finishes, patterns, textures)
- Technical Specifications (performance, ratings, certifications)
- Commercial Information (pricing, availability, warranty)
- Sustainability & Compliance (certifications, environmental impact)

Organizes existing 120+ metafields into logical categories for intelligent extraction.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from app.config import get_supabase_client
from app.services.ai_service import AIService


@dataclass
class CoreIdentityMetadata:
    """Primary identification, brand & manufacturer, collection & design, origin"""
    name: Optional[str] = None
    product_code: Optional[str] = None
    sku: Optional[str] = None
    model: Optional[str] = None
    manufacturer: Optional[str] = None
    brand: Optional[str] = None
    factory: Optional[str] = None
    group_of_companies: Optional[str] = None
    collection: Optional[str] = None
    designer: Optional[str] = None
    year: Optional[int] = None
    country_of_origin: Optional[str] = None
    quarry_name: Optional[str] = None


@dataclass
class PhysicalPropertiesMetadata:
    """Dimensions, weight, shape & form, material composition, physical characteristics"""
    length: Optional[float] = None
    width: Optional[float] = None
    thickness: Optional[float] = None
    dimension_unit: Optional[str] = None
    weight_value: Optional[float] = None
    weight_unit: Optional[str] = None
    shape: Optional[str] = None
    edge_type: Optional[str] = None
    rectified: Optional[bool] = None
    material_category: Optional[str] = None
    material_type: Optional[str] = None
    wood_species: Optional[str] = None
    stone_type: Optional[str] = None
    density: Optional[float] = None
    porosity: Optional[float] = None
    moisture_content: Optional[float] = None


@dataclass
class VisualPropertiesMetadata:
    """Colors, surface characteristics, visual patterns, shade variation"""
    primary_color: Optional[str] = None
    secondary_color: Optional[str] = None
    color_family: Optional[str] = None
    color_variation: Optional[str] = None
    surface_finish: Optional[str] = None
    surface_pattern: Optional[str] = None
    surface_texture: Optional[str] = None
    surface_treatment: Optional[str] = None
    grain_pattern: Optional[str] = None
    veining_pattern: Optional[str] = None
    movement_pattern: Optional[str] = None
    v_rating: Optional[str] = None


@dataclass
class TechnicalSpecificationsMetadata:
    """Strength & durability, hardness, resistance properties, performance ratings, thermal properties"""
    breaking_strength: Optional[float] = None
    modulus_of_rupture: Optional[float] = None
    compressive_strength: Optional[float] = None
    flexural_strength: Optional[float] = None
    mohs_hardness: Optional[str] = None
    janka_hardness: Optional[float] = None
    stone_hardness: Optional[float] = None
    water_absorption: Optional[str] = None
    slip_resistance: Optional[str] = None
    frost_resistance: Optional[bool] = None
    heat_resistance: Optional[bool] = None
    chemical_resistance: Optional[str] = None
    stain_resistance: Optional[str] = None
    fade_resistance: Optional[str] = None
    abrasion_resistance: Optional[float] = None
    wear_resistance: Optional[str] = None
    pei_rating: Optional[str] = None
    traffic_rating: Optional[str] = None
    fire_rating: Optional[str] = None
    thermal_expansion: Optional[float] = None
    thermal_conductivity: Optional[float] = None
    thermal_shock: Optional[bool] = None
    antimicrobial: Optional[bool] = None
    sound_insulation: Optional[str] = None
    dimensional_stability: Optional[float] = None


@dataclass
class CommercialInformationMetadata:
    """Pricing, availability, warranty & support, application areas"""
    price_range: Optional[str] = None
    price_currency: Optional[str] = None
    price_unit: Optional[str] = None
    stock_status: Optional[str] = None
    lead_time_days: Optional[int] = None
    minimum_order: Optional[int] = None
    warranty: Optional[str] = None
    certifications: Optional[List[str]] = None
    application_area: Optional[List[str]] = None
    usage_type: Optional[List[str]] = None
    environments: Optional[List[str]] = None


@dataclass
class SustainabilityComplianceMetadata:
    """Environmental impact, certifications, health & safety"""
    sustainability: Optional[str] = None
    recycled_content_percent: Optional[float] = None
    recyclable: Optional[bool] = None
    voc_level: Optional[str] = None
    energy_efficiency: Optional[str] = None
    carbon_footprint: Optional[str] = None
    certifications: Optional[List[str]] = None
    antimicrobial: Optional[bool] = None


@dataclass
class InstallationMaintenanceMetadata:
    """Installation, subfloor & underlayment, maintenance"""
    installation_method: Optional[str] = None
    installation_format: Optional[str] = None
    installation_pattern: Optional[str] = None
    difficulty_level: Optional[str] = None
    tools_required: Optional[List[str]] = None
    preparation_needed: Optional[str] = None
    installation_time: Optional[str] = None
    subfloor_requirements: Optional[str] = None
    underlayment_required: Optional[bool] = None
    joint_width: Optional[float] = None
    cleaning_method: Optional[List[str]] = None
    sealing_required: Optional[bool] = None
    maintenance_frequency: Optional[str] = None
    repairability: Optional[str] = None


@dataclass
class CanonicalMetadataSchema:
    """Complete canonical metadata schema with all categories"""
    core_identity: CoreIdentityMetadata
    physical_properties: PhysicalPropertiesMetadata
    visual_properties: VisualPropertiesMetadata
    technical_specifications: TechnicalSpecificationsMetadata
    commercial_information: CommercialInformationMetadata
    sustainability_compliance: SustainabilityComplianceMetadata
    installation_maintenance: InstallationMaintenanceMetadata


@dataclass
class MetadataExtractionOptions:
    """Options for metadata extraction"""
    include_categories: Optional[List[str]] = None
    confidence_threshold: float = 0.6
    extraction_method: str = 'ai_extraction'
    validate_required: bool = True


@dataclass
class MetadataExtractionResult:
    """Result of metadata extraction"""
    metadata: Dict[str, Any]
    confidence: float
    extracted_fields: int
    total_fields: int
    extraction_method: str
    processing_time: float
    issues: Optional[List[Dict[str, str]]] = None


class CanonicalMetadataSchemaService:
    """Service for extracting and managing canonical metadata schema"""
    
    # Field mappings to categories
    FIELD_MAPPINGS = {
        # Core Identity mappings
        'manufacturer': 'core_identity',
        'brand': 'core_identity',
        'collection': 'core_identity',
        'productCode': 'core_identity',
        'sku': 'core_identity',
        'model': 'core_identity',
        'year': 'core_identity',
        'countryOfOrigin': 'core_identity',
        'factory': 'core_identity',
        'groupOfCompanies': 'core_identity',
        'quarryName': 'core_identity',
        
        # Physical Properties mappings
        'length': 'physical_properties',
        'width': 'physical_properties',
        'thickness': 'physical_properties',
        'dimensionUnit': 'physical_properties',
        'weightValue': 'physical_properties',
        'weightUnit': 'physical_properties',
        'tileShape': 'physical_properties',
        'edgeType': 'physical_properties',
        'rectified': 'physical_properties',
        'materialCategory': 'physical_properties',
        'tileType': 'physical_properties',
        'woodSpecies': 'physical_properties',
        'stoneType': 'physical_properties',
        'stoneDensity': 'physical_properties',
        'porosity': 'physical_properties',
        'moistureContent': 'physical_properties',
        
        # Visual Properties mappings
        'primaryColor': 'visual_properties',
        'secondaryColor': 'visual_properties',
        'colorFamily': 'visual_properties',
        'colorVariation': 'visual_properties',
        'surfaceFinish': 'visual_properties',
        'surfacePattern': 'visual_properties',
        'surfaceTexture': 'visual_properties',
        'surfaceTreatment': 'visual_properties',
        'grainPattern': 'visual_properties',
        'veiningPattern': 'visual_properties',
        'movementPattern': 'visual_properties',
        'vRating': 'visual_properties',
        
        # Technical Specifications mappings
        'breakingStrength': 'technical_specifications',
        'modulusOfRupture': 'technical_specifications',
        'compressiveStrength': 'technical_specifications',
        'flexuralStrength': 'technical_specifications',
        'mohsHardness': 'technical_specifications',
        'jankaHardness': 'technical_specifications',
        'stoneHardness': 'technical_specifications',
        'waterAbsorption': 'technical_specifications',
        'slipResistance': 'technical_specifications',
        'frostResistance': 'technical_specifications',
        'heatResistance': 'technical_specifications',
        'chemicalResistance': 'technical_specifications',
        'stainResistance': 'technical_specifications',
        'fadeResistance': 'technical_specifications',
        'abrasionResistance': 'technical_specifications',
        'wearResistance': 'technical_specifications',
        'peiRating': 'technical_specifications',
        'trafficRating': 'technical_specifications',
        'fireRating': 'technical_specifications',
        'thermalExpansion': 'technical_specifications',
        'thermalConductivity': 'technical_specifications',
        'thermalShock': 'technical_specifications',
        'antimicrobial': 'technical_specifications',
        'soundInsulation': 'technical_specifications',
        'dimensionalStability': 'technical_specifications',
        
        # Commercial Information mappings
        'priceRange': 'commercial_information',
        'priceCurrency': 'commercial_information',
        'priceUnit': 'commercial_information',
        'stockStatus': 'commercial_information',
        'leadTimeDays': 'commercial_information',
        'minimumOrder': 'commercial_information',
        'warranty': 'commercial_information',
        'certifications': 'commercial_information',
        'applicationArea': 'commercial_information',
        'usageType': 'commercial_information',
        'environments': 'commercial_information',
        
        # Sustainability & Compliance mappings
        'sustainability': 'sustainability_compliance',
        'recycledContentPercent': 'sustainability_compliance',
        'recyclable': 'sustainability_compliance',
        'vocLevel': 'sustainability_compliance',
        'energyEfficiency': 'sustainability_compliance',
        'carbonFootprint': 'sustainability_compliance',
        
        # Installation & Maintenance mappings
        'installationMethod': 'installation_maintenance',
        'installationFormat': 'installation_maintenance',
        'installationPattern': 'installation_maintenance',
        'difficultyLevel': 'installation_maintenance',
        'toolsRequired': 'installation_maintenance',
        'preparationNeeded': 'installation_maintenance',
        'installationTime': 'installation_maintenance',
        'subfloorRequirements': 'installation_maintenance',
        'underlaymentRequired': 'installation_maintenance',
        'jointWidth': 'installation_maintenance',
        'cleaningMethod': 'installation_maintenance',
        'sealingRequired': 'installation_maintenance',
        'maintenanceFrequency': 'installation_maintenance',
        'repairability': 'installation_maintenance',
    }

    def __init__(self):
        self.supabase = get_supabase_client()
        self.ai_service = AIService()

    async def extract_canonical_metadata(
        self,
        content: str,
        product_id: Optional[str] = None,
        options: Optional[MetadataExtractionOptions] = None
    ) -> MetadataExtractionResult:
        """Extract comprehensive metadata from product content using canonical schema"""
        start_time = time.time()
        
        if options is None:
            options = MetadataExtractionOptions()
        
        print("🔍 Extracting canonical metadata using comprehensive schema...")
        
        try:
            # Get all metafield definitions
            response = self.supabase.table('material_metadata_fields').select('*').eq('is_global', True).execute()
            metafield_defs = response.data or []
            
            # Extract metadata using AI
            extracted_metadata = await self._extract_metadata_with_ai(content, metafield_defs)
            
            # Organize into canonical schema
            canonical_metadata = self._organize_into_canonical_schema(
                extracted_metadata,
                options.include_categories
            )
            
            # Calculate metrics
            extracted_fields = len(extracted_metadata)
            total_fields = len(metafield_defs)
            confidence = self._calculate_overall_confidence(extracted_metadata, extracted_fields, total_fields)
            processing_time = time.time() - start_time
            
            # Save to database if product_id provided
            if product_id:
                await self._save_canonical_metadata(product_id, canonical_metadata, extracted_metadata)
            
            print(f"✅ Extracted {extracted_fields}/{total_fields} metadata fields")
            print(f"📊 Overall confidence: {confidence * 100:.1f}%")
            
            return MetadataExtractionResult(
                metadata=canonical_metadata,
                confidence=confidence,
                extracted_fields=extracted_fields,
                total_fields=total_fields,
                extraction_method=options.extraction_method,
                processing_time=processing_time
            )
            
        except Exception as error:
            print(f"❌ Error extracting canonical metadata: {error}")
            raise Exception(f"Failed to extract metadata: {str(error)}")

    async def _extract_metadata_with_ai(self, content: str, metafield_defs: List[Dict]) -> Dict[str, Any]:
        """Extract metadata using AI service"""
        try:
            # Create extraction prompt
            field_descriptions = []
            for field_def in metafield_defs[:50]:  # Limit to avoid token limits
                field_descriptions.append(
                    f"- {field_def['field_name']}: {field_def['description']} "
                    f"(Type: {field_def['field_type']})"
                )
            
            prompt = f"""
            Extract metadata from the following product content. Return a JSON object with field names as keys and extracted values.
            
            Available fields:
            {chr(10).join(field_descriptions)}
            
            Content:
            {content[:2000]}  # Limit content to avoid token limits
            
            Return only valid JSON with extracted values. Use null for missing values.
            """
            
            # Call AI service
            response = await self.ai_service.generate_text(
                prompt=prompt,
                model="claude-4-5-haiku-20250514",
                max_tokens=2000
            )
            
            # Parse JSON response
            try:
                extracted_metadata = json.loads(response)
                return extracted_metadata
            except json.JSONDecodeError:
                print("⚠️ Failed to parse AI response as JSON, using empty metadata")
                return {}
                
        except Exception as error:
            print(f"❌ Error in AI metadata extraction: {error}")
            return {}

    def _organize_into_canonical_schema(
        self,
        extracted_metadata: Dict[str, Any],
        include_categories: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Organize extracted metadata into canonical schema categories"""
        canonical_metadata = {
            'core_identity': {},
            'physical_properties': {},
            'visual_properties': {},
            'technical_specifications': {},
            'commercial_information': {},
            'sustainability_compliance': {},
            'installation_maintenance': {},
        }
        
        # Organize fields by category
        for field_name, value in extracted_metadata.items():
            category = self.FIELD_MAPPINGS.get(field_name)
            
            if category and (not include_categories or category in include_categories):
                if category not in canonical_metadata:
                    canonical_metadata[category] = {}
                canonical_metadata[category][field_name] = value
        
        # Remove empty categories
        canonical_metadata = {
            k: v for k, v in canonical_metadata.items() 
            if v and len(v) > 0
        }
        
        return canonical_metadata

    def _calculate_overall_confidence(
        self,
        extracted_metadata: Dict[str, Any],
        extracted_fields: int,
        total_fields: int
    ) -> float:
        """Calculate overall confidence score based on extraction results"""
        # Base confidence from extraction coverage
        coverage_score = extracted_fields / total_fields if total_fields > 0 else 0
        
        # Bonus for critical fields
        critical_fields = ['manufacturer', 'brand', 'collection', 'materialCategory', 'primaryColor']
        critical_fields_found = sum(1 for field in critical_fields if extracted_metadata.get(field))
        critical_bonus = critical_fields_found / len(critical_fields) * 0.2
        
        # Quality bonus for non-empty values
        non_empty_values = sum(
            1 for value in extracted_metadata.values() 
            if value is not None and value != ''
        )
        quality_bonus = (non_empty_values / extracted_fields * 0.1) if extracted_fields > 0 else 0
        
        return min(1.0, coverage_score + critical_bonus + quality_bonus)

    async def _save_canonical_metadata(
        self,
        product_id: str,
        canonical_metadata: Dict[str, Any],
        raw_metadata: Dict[str, Any]
    ) -> None:
        """Save canonical metadata to database"""
        try:
            # Update product with canonical metadata in properties field
            update_data = {
                'properties': {
                    **canonical_metadata,
                    'extraction_timestamp': datetime.utcnow().isoformat(),
                    'extraction_method': 'canonical_schema',
                },
                'updated_at': datetime.utcnow().isoformat(),
            }
            
            response = self.supabase.table('products').update(update_data).eq('id', product_id).execute()
            
            if response.data:
                print(f"✅ Saved canonical metadata for product {product_id}")
            else:
                print(f"⚠️ No product found with ID {product_id}")
                
        except Exception as error:
            print(f"❌ Error saving canonical metadata: {error}")
            raise error

    async def get_product_canonical_metadata(self, product_id: str) -> Optional[Dict[str, Any]]:
        """Get canonical metadata for a product"""
        try:
            response = self.supabase.table('products').select('properties, metadata').eq('id', product_id).single().execute()
            
            if response.data:
                properties = response.data.get('properties', {})
                # Remove extraction metadata
                canonical_metadata = {
                    k: v for k, v in properties.items() 
                    if k not in ['extraction_timestamp', 'extraction_method']
                }
                return canonical_metadata
            
            return None
            
        except Exception as error:
            print(f"❌ Error getting canonical metadata: {error}")
            return None

    def validate_metadata_completeness(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate canonical metadata completeness"""
        critical_fields = {
            'core_identity': ['name', 'manufacturer', 'brand'],
            'physical_properties': ['materialCategory', 'length', 'width'],
            'visual_properties': ['primaryColor', 'surfaceFinish'],
            'technical_specifications': ['waterAbsorption', 'slipResistance'],
            'commercial_information': ['applicationArea', 'priceRange'],
        }
        
        missing_critical_fields = []
        recommendations = []
        total_critical_fields = 0
        found_critical_fields = 0
        
        for category, fields in critical_fields.items():
            category_data = metadata.get(category, {})
            
            for field in fields:
                total_critical_fields += 1
                if category_data.get(field):
                    found_critical_fields += 1
                else:
                    missing_critical_fields.append(f"{category}.{field}")
        
        completion_score = found_critical_fields / total_critical_fields if total_critical_fields > 0 else 0
        is_complete = completion_score >= 0.8  # 80% threshold
        
        # Generate recommendations
        if 'core_identity.manufacturer' in missing_critical_fields:
            recommendations.append('Add manufacturer information for better product identification')
        if 'physical_properties.materialCategory' in missing_critical_fields:
            recommendations.append('Specify material category for proper classification')
        if 'visual_properties.primaryColor' in missing_critical_fields:
            recommendations.append('Include primary color for visual search capabilities')
        if 'commercial_information.applicationArea' in missing_critical_fields:
            recommendations.append('Define application areas for better product matching')
        
        return {
            'is_complete': is_complete,
            'completion_score': completion_score,
            'missing_critical_fields': missing_critical_fields,
            'recommendations': recommendations,
        }

    @classmethod
    def get_schema_statistics(cls) -> Dict[str, Any]:
        """Get metadata schema statistics"""
        fields_by_category = {}
        
        # Count fields by category
        for field, category in cls.FIELD_MAPPINGS.items():
            fields_by_category[category] = fields_by_category.get(category, 0) + 1
        
        return {
            'total_categories': len(fields_by_category),
            'total_fields': len(cls.FIELD_MAPPINGS),
            'fields_by_category': fields_by_category,
            'critical_fields': ['manufacturer', 'brand', 'collection', 'materialCategory', 'primaryColor', 'applicationArea'],
        }
