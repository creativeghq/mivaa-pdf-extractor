"""
Metafield Extraction Service

Extracts metafield values from product discovery results and maps them to
existing metafield definitions in the material_metadata_fields table.

Focus areas (per user requirements):
- slip_resistance (R9-R13)
- fire_rating (A1-F)
- water_absorption
- pei_rating (1-5)
- dimensions (length, width, thickness)
- colors (primary_color, secondary_color, color_family)
- surface_finish
- certifications
- brand, manufacturer, factory, group_of_companies
- material_category
"""

import logging
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

from app.services.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)


class MetafieldExtractionService:
    """
    Service for extracting metafield values from product discovery results.
    
    Workflow:
    1. Product discovery identifies metafield categories in PDF
    2. This service maps discovered values to metafield definitions
    3. Creates records in product_metafield_values table
    4. Links metafields to products during product creation
    """
    
    def __init__(self):
        self.logger = logger
        self.supabase = get_supabase_client()
        self.metafield_definitions = {}
        self._load_metafield_definitions()
        
    def _load_metafield_definitions(self):
        """Load all metafield definitions from database."""
        try:
            response = self.supabase.client.table('material_metadata_fields')\
                .select('id, field_name, display_name, field_type, dropdown_options, extraction_hints')\
                .execute()
            
            if response.data:
                for field in response.data:
                    self.metafield_definitions[field['field_name']] = field
                
                self.logger.info(f"✅ Loaded {len(self.metafield_definitions)} metafield definitions")
            else:
                self.logger.warning("⚠️ No metafield definitions found in database")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to load metafield definitions: {e}")
    
    async def extract_metafields_from_catalog(
        self,
        catalog: Any,  # ProductCatalog object
        document_id: str
    ) -> Dict[str, Any]:
        """
        Extract metafield values from ProductCatalog and save to database.

        Args:
            catalog: ProductCatalog from ProductDiscoveryService
            document_id: Parent document ID

        Returns:
            Statistics about metafield extraction
        """
        try:
            self.logger.info(f"🔍 Extracting metafields from product catalog")

            total_metafields = 0
            product_metafields = {}

            # Get metafield_categories from catalog
            metafield_categories = {}
            if hasattr(catalog, 'metafield_categories'):
                metafield_categories = catalog.metafield_categories

            # Get all products from database to get their IDs
            products_response = self.supabase.client.table('products')\
                .select('id, name')\
                .eq('source_document_id', document_id)\
                .execute()

            product_name_to_id = {p['name']: p['id'] for p in products_response.data}

            # Extract metafields for each product
            for product in catalog.products:
                product_name = product.name
                product_id = product_name_to_id.get(product_name)

                if not product_id:
                    self.logger.warning(f"⚠️ Product '{product_name}' not found in database")
                    continue

                product_metafields_list = []

                # Extract product-specific metafields
                product_meta = {}
                if hasattr(product, 'metafields'):
                    product_meta = product.metafields

                # Priority metafields (ALL metafields, not just specific ones)
                priority_fields = [
                    'slip_resistance', 'fire_rating', 'water_absorption', 'pei_rating',
                    'length', 'width', 'thickness', 'dimension_unit',
                    'primary_color', 'secondary_color', 'color_family',
                    'surface_finish', 'certifications',
                    'brand', 'manufacturer', 'factory', 'group_of_companies',
                    'material_category', 'tile_type', 'edge_type', 'rectified'
                ]

                # Extract priority fields first
                for field_name in priority_fields:
                    value = product_meta.get(field_name) or metafield_categories.get(field_name)

                    if value:
                        metafield_record = self._create_metafield_record(
                            field_name=field_name,
                            value=value,
                            document_id=document_id
                        )

                        if metafield_record:
                            # Save to database
                            await self._save_metafield_to_database(
                                product_id=product_id,
                                metafield_record=metafield_record
                            )
                            product_metafields_list.append(metafield_record)
                            total_metafields += 1

                # Extract all other metafields
                for field_name, value in product_meta.items():
                    if field_name not in priority_fields and value:
                        metafield_record = self._create_metafield_record(
                            field_name=field_name,
                            value=value,
                            document_id=document_id
                        )

                        if metafield_record:
                            # Save to database
                            await self._save_metafield_to_database(
                                product_id=product_id,
                                metafield_record=metafield_record
                            )
                            product_metafields_list.append(metafield_record)
                            total_metafields += 1

                product_metafields[product_name] = product_metafields_list

                self.logger.info(
                    f"✅ Extracted {len(product_metafields_list)} metafields for product: {product_name}"
                )

            return {
                'total_metafields': total_metafields,
                'products_processed': len(product_metafields),
                'product_metafields': product_metafields
            }

        except Exception as e:
            self.logger.error(f"❌ Metafield extraction failed: {e}")
            return {
                'total_metafields': 0,
                'products_processed': 0,
                'product_metafields': {}
            }

    async def _save_metafield_to_database(
        self,
        product_id: str,
        metafield_record: Dict[str, Any]
    ):
        """Save metafield value to database."""
        try:
            record = {
                'id': str(uuid.uuid4()),
                'product_id': product_id,
                'field_id': metafield_record['field_id'],
                'value': str(metafield_record['value']),
                'confidence': metafield_record.get('confidence', 0.9),
                'created_at': datetime.utcnow().isoformat(),
                'updated_at': datetime.utcnow().isoformat()
            }

            self.supabase.client.table('product_metafield_values').insert(record).execute()

        except Exception as e:
            self.logger.error(f"❌ Failed to save metafield to database: {e}")
    
    def _create_metafield_record(
        self,
        field_name: str,
        value: Any,
        document_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Create a metafield record from field name and value.
        
        Args:
            field_name: Metafield name (e.g., 'slip_resistance')
            value: Extracted value (e.g., 'R11')
            document_id: Parent document ID
            
        Returns:
            Metafield record dict or None if invalid
        """
        try:
            # Get metafield definition
            definition = self.metafield_definitions.get(field_name)
            
            if not definition:
                self.logger.warning(f"⚠️ No definition found for metafield: {field_name}")
                return None
            
            # Validate value against field type
            validated_value = self._validate_metafield_value(
                value=value,
                field_type=definition['field_type'],
                dropdown_options=definition.get('dropdown_options')
            )
            
            if validated_value is None:
                self.logger.warning(
                    f"⚠️ Invalid value '{value}' for metafield '{field_name}' "
                    f"(type: {definition['field_type']})"
                )
                return None
            
            return {
                'field_id': definition['id'],
                'field_name': field_name,
                'display_name': definition['display_name'],
                'field_type': definition['field_type'],
                'value': validated_value,
                'source': 'product_discovery',
                'confidence': 0.9,  # High confidence from AI discovery
                'extracted_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create metafield record for {field_name}: {e}")
            return None
    
    def _validate_metafield_value(
        self,
        value: Any,
        field_type: str,
        dropdown_options: Optional[List[str]] = None
    ) -> Optional[Any]:
        """
        Validate metafield value against field type and dropdown options.
        
        Args:
            value: Value to validate
            field_type: Field type (text, number, boolean, dropdown, etc.)
            dropdown_options: Allowed values for dropdown fields
            
        Returns:
            Validated value or None if invalid
        """
        try:
            if value is None or value == '':
                return None
            
            # Text fields
            if field_type == 'text':
                return str(value)
            
            # Number fields
            elif field_type == 'number':
                try:
                    # Handle strings like "10mm" -> 10
                    if isinstance(value, str):
                        # Extract first number from string
                        match = re.search(r'[\d.]+', value)
                        if match:
                            return float(match.group())
                    return float(value)
                except (ValueError, TypeError):
                    return None
            
            # Boolean fields
            elif field_type == 'boolean':
                if isinstance(value, bool):
                    return value
                if isinstance(value, str):
                    return value.lower() in ['true', 'yes', '1', 'y']
                return bool(value)
            
            # Dropdown fields
            elif field_type == 'dropdown':
                str_value = str(value)
                
                if dropdown_options:
                    # Case-insensitive match
                    for option in dropdown_options:
                        if str_value.lower() == option.lower():
                            return option
                    
                    # Partial match
                    for option in dropdown_options:
                        if str_value.lower() in option.lower() or option.lower() in str_value.lower():
                            return option
                    
                    self.logger.warning(
                        f"⚠️ Value '{value}' not in dropdown options: {dropdown_options}"
                    )
                    return None
                
                return str_value
            
            # Default: return as string
            else:
                return str(value)
                
        except Exception as e:
            self.logger.error(f"❌ Value validation failed: {e}")
            return None
    
    async def link_metafields_to_product(
        self,
        product_id: str,
        metafields: List[Dict[str, Any]]
    ) -> int:
        """
        Link metafield values to a product in the database.
        
        Args:
            product_id: Product ID
            metafields: List of metafield records
            
        Returns:
            Number of metafields linked
        """
        try:
            linked_count = 0
            
            for metafield in metafields:
                try:
                    record = {
                        'id': str(uuid.uuid4()),
                        'product_id': product_id,
                        'field_id': metafield['field_id'],
                        'value': metafield['value'],
                        'created_at': datetime.utcnow().isoformat(),
                        'updated_at': datetime.utcnow().isoformat()
                    }
                    
                    self.supabase.client.table('product_metafield_values').insert(record).execute()
                    linked_count += 1
                    
                except Exception as e:
                    self.logger.error(
                        f"❌ Failed to link metafield {metafield['field_name']} to product {product_id}: {e}"
                    )
            
            self.logger.info(f"✅ Linked {linked_count}/{len(metafields)} metafields to product {product_id}")
            return linked_count
            
        except Exception as e:
            self.logger.error(f"❌ Failed to link metafields to product: {e}")
            return 0

