import pymupdf4llm
import pathlib
from pathlib import Path
import json
import os
# import pandas as pd
import fitz
import csv

# Extract PDF content as Markdown
def extract_pdf_to_markdown(file_name, page_number):
    
    page_number_list=None
    if not (page_number is None):
        page_number_list= [page_number]   
    
    return pymupdf4llm.to_markdown(file_name, pages=page_number_list)

def extract_pdf_tables(file_name,page_number,output_dir):

    doc = fitz.open(file_name)
    csv_path = os.path.join(output_dir, 'csv')
    os.makedirs(csv_path)
    page_count=1
    
    if not (page_number is None):
        page = doc[page_number]
        extract_table_to_csv(page.find_tables(),page_number, csv_path)
    else:
        for page in doc:    
            tabs = page.find_tables()
            # print(tabs.tables)
            extract_table_to_csv(tabs,page_count, csv_path)            
            page_count += 1

def extract_table_to_csv(tabs, page_count,output_dir):
    table_count=1
    
    for tab in tabs.tables:
        
        csv_file = output_dir + '/'+str(page_count)+'_'+str(table_count)+'.csv'
        
        data_file = open(csv_file, 'w', encoding="utf-8")

        # create the csv writer object
        csv_writer = csv.writer(data_file)
        
        for table in tab.extract():
            csv_writer.writerow(table)
        
        data_file.close()
        table_count += 1    


def extract_json_and_images(file_path, output_dir, page_number):

    page_number_list=None
    if not (page_number is None):
        page_number_list= [page_number-1]

    image_path = os.path.join(output_dir, 'images')
    os.makedirs(image_path)

    md_text_images = pymupdf4llm.to_markdown(doc=file_path,
                                             pages=page_number_list,
                                            page_chunks=True,
                                            write_images=True,
                                            image_path=image_path,
                                            image_format="jpg",
                                            dpi=200)
    
    pathlib.Path(output_dir+str("/output.json")).write_text(json.dumps(str(md_text_images)))


def extract_functional_metadata(file_path, page_number=None, extract_mode="comprehensive"):
    """
    Extract comprehensive functional metadata from material tile specification PDFs.
    
    Args:
        file_path: Path to the PDF file
        page_number: Specific page to extract (optional, extracts all pages if None)
        extract_mode: Extraction mode - "comprehensive", "safety", "surface", etc.
    
    Returns:
        Dictionary containing structured functional metadata for the 9 categories
    """
    import re
    from typing import Dict, Any, Optional
    
    # Define the 9 functional metadata categories
    FUNCTIONAL_CATEGORIES = {
        "slip_safety": {
            "name": "Slip/Safety Ratings",
            "fields": ["slip_resistance", "pendulum_test_value", "ramp_test_angle", "barefoot_rating", "shod_rating", "safety_classification"]
        },
        "surface_gloss": {
            "name": "Surface Gloss/Reflectivity",
            "fields": ["gloss_level", "light_reflectance_value", "surface_finish", "sheen_measurement", "matt_satin_gloss"]
        },
        "mechanical": {
            "name": "Mechanical Properties",
            "fields": ["breaking_strength", "modulus_rupture", "impact_resistance", "abrasion_resistance", "hardness", "compressive_strength"]
        },
        "thermal": {
            "name": "Thermal Properties",
            "fields": ["thermal_conductivity", "thermal_expansion", "fire_resistance", "heat_resistance", "thermal_shock_resistance"]
        },
        "water_moisture": {
            "name": "Water/Moisture Resistance",
            "fields": ["water_absorption", "porosity", "permeability", "frost_resistance", "moisture_resistance_rating"]
        },
        "chemical_hygiene": {
            "name": "Chemical/Hygiene Resistance",
            "fields": ["chemical_resistance", "stain_resistance", "acid_resistance", "alkali_resistance", "antibacterial_properties"]
        },
        "acoustic_electrical": {
            "name": "Acoustic/Electrical Properties",
            "fields": ["sound_absorption", "impact_sound_insulation", "electrical_conductivity", "antistatic_properties"]
        },
        "environmental": {
            "name": "Environmental/Sustainability",
            "fields": ["recycled_content", "emission_levels", "environmental_certification", "carbon_footprint", "leed_points"]
        },
        "dimensional_aesthetic": {
            "name": "Dimensional/Aesthetic",
            "fields": ["dimensions", "thickness", "weight", "color_variation", "surface_texture", "edge_finish"]
        }
    }
    
    try:
        # Extract markdown content from PDF
        page_number_list = None
        if page_number is not None:
            page_number_list = [page_number]
            
        markdown_content = extract_pdf_to_markdown(file_path, page_number)
        
        # Initialize result structure
        result = {
            "extraction_metadata": {
                "file_path": str(file_path),
                "page_number": page_number,
                "extract_mode": extract_mode,
                "categories_extracted": len(FUNCTIONAL_CATEGORIES),
                "extraction_timestamp": None
            },
            "functional_categories": {}
        }
        
        # Add timestamp
        from datetime import datetime
        result["extraction_metadata"]["extraction_timestamp"] = datetime.now().isoformat()
        
        # Determine which categories to extract based on mode
        categories_to_extract = FUNCTIONAL_CATEGORIES.keys()
        if extract_mode == "safety":
            categories_to_extract = ["slip_safety"]
        elif extract_mode == "surface":
            categories_to_extract = ["surface_gloss"]
        elif extract_mode == "mechanical":
            categories_to_extract = ["mechanical"]
        # Add more specific modes as needed
        
        # Extract metadata for each category
        for category_key in categories_to_extract:
            category_info = FUNCTIONAL_CATEGORIES[category_key]
            category_data = _extract_category_metadata(
                markdown_content,
                category_info,
                category_key
            )
            result["functional_categories"][category_key] = category_data
            
        return result
        
    except Exception as e:
        return {
            "extraction_metadata": {
                "file_path": str(file_path),
                "page_number": page_number,
                "extract_mode": extract_mode,
                "error": str(e),
                "extraction_timestamp": datetime.now().isoformat()
            },
            "functional_categories": {}
        }


def _extract_category_metadata(markdown_content: str, category_info: dict, category_key: str) -> dict:
    """
    Extract metadata for a specific functional category using pattern matching and AI prompts.
    
    Args:
        markdown_content: Extracted markdown text from PDF
        category_info: Category definition with name and fields
        category_key: Category identifier key
        
    Returns:
        Dictionary containing extracted metadata for the category
    """
    
    # Initialize category result structure
    category_result = {
        "category_name": category_info["name"],
        "extraction_confidence": "low",
        "extracted_values": {},
        "raw_text_matches": [],
        "extraction_notes": []
    }
    
    # Enhanced extraction patterns for slip/safety ratings (most developed)
    if category_key == "slip_safety":
        patterns = {
            "slip_resistance": [
                r"slip\s+resistance[:\s]*([R\d]+|Class\s*[ABC\d]+|\d+\.?\d*)",
                r"pendulum\s+test[:\s]*(\d+\.?\d*)",
                r"PTV[:\s]*(\d+\.?\d*)",
                r"ramp\s+test[:\s]*(\d+\.?\d*°?)",
                r"barefoot[:\s]*([R\d]+|Class\s*[ABC\d]+)",
                r"shod[:\s]*([R\d]+|Class\s*[ABC\d]+)"
            ],
            "pendulum_test_value": [
                r"PTV[:\s]*(\d+\.?\d*)",
                r"pendulum\s+test\s+value[:\s]*(\d+\.?\d*)",
                r"pendulum[:\s]*(\d+\.?\d*)"
            ],
            "ramp_test_angle": [
                r"ramp\s+test[:\s]*(\d+\.?\d*°?)",
                r"ramp\s+angle[:\s]*(\d+\.?\d*°?)"
            ]
        }
        
        # Extract using patterns
        extracted_data = _apply_extraction_patterns(markdown_content, patterns)
        category_result["extracted_values"] = extracted_data["values"]
        category_result["raw_text_matches"] = extracted_data["matches"]
        category_result["extraction_confidence"] = "medium" if extracted_data["values"] else "low"
        
    # Enhanced extraction patterns for surface gloss/reflectivity
    elif category_key == "surface_gloss":
        patterns = {
            "gloss_level": [
                r"gloss\s+level[:\s]*(\d+\.?\d*%?|high|medium|low|matt?e?|satin|semi-gloss|gloss)",
                r"gloss[:\s]*(\d+\.?\d*%?|high|medium|low|matt?e?|satin|semi-gloss)",
                r"finish[:\s]*(matt?e?|satin|semi-gloss|gloss|high\s+gloss|low\s+gloss)"
            ],
            "light_reflectance_value": [
                r"light\s+reflectance[:\s]*(\d+\.?\d*%?)",
                r"reflectance[:\s]*(\d+\.?\d*%?)",
                r"LRV[:\s]*(\d+\.?\d*%?)",
                r"light\s+reflection[:\s]*(\d+\.?\d*%?)"
            ],
            "surface_finish": [
                r"surface\s+finish[:\s]*(matt?e?|satin|semi-gloss|gloss|polished|textured|smooth)",
                r"finish[:\s]*(matt?e?|satin|semi-gloss|gloss|polished|textured|smooth)",
                r"surface[:\s]*(matt?e?|satin|semi-gloss|gloss|polished|textured|smooth)"
            ],
            "sheen_measurement": [
                r"sheen[:\s]*(\d+\.?\d*%?|high|medium|low)",
                r"sheen\s+level[:\s]*(\d+\.?\d*%?|high|medium|low)",
                r"sheen\s+measurement[:\s]*(\d+\.?\d*%?)"
            ],
            "matt_satin_gloss": [
                r"(matt?e?|satin|semi-gloss|gloss|high\s+gloss)",
                r"finish\s+type[:\s]*(matt?e?|satin|semi-gloss|gloss)",
                r"surface\s+type[:\s]*(matt?e?|satin|semi-gloss|gloss)"
            ]
        }
        
        # Extract using patterns
        extracted_data = _apply_extraction_patterns(markdown_content, patterns)
        category_result["extracted_values"] = extracted_data["values"]
        category_result["raw_text_matches"] = extracted_data["matches"]
        category_result["extraction_confidence"] = "medium" if extracted_data["values"] else "low"
        
    # Enhanced extraction patterns for mechanical properties
    elif category_key == "mechanical":
        patterns = {
            "breaking_strength": [
                r"breaking\s+strength[:\s]*(\d+\.?\d*\s*(?:N/mm²|MPa|PSI|kN))",
                r"tensile\s+strength[:\s]*(\d+\.?\d*\s*(?:N/mm²|MPa|PSI|kN))",
                r"ultimate\s+strength[:\s]*(\d+\.?\d*\s*(?:N/mm²|MPa|PSI|kN))"
            ],
            "modulus_rupture": [
                r"modulus\s+of\s+rupture[:\s]*(\d+\.?\d*\s*(?:N/mm²|MPa|PSI))",
                r"flexural\s+strength[:\s]*(\d+\.?\d*\s*(?:N/mm²|MPa|PSI))",
                r"MOR[:\s]*(\d+\.?\d*\s*(?:N/mm²|MPa|PSI))"
            ],
            "impact_resistance": [
                r"impact\s+resistance[:\s]*(\d+\.?\d*\s*(?:J|ft-lb|Nm))",
                r"impact\s+strength[:\s]*(\d+\.?\d*\s*(?:J|ft-lb|Nm))",
                r"Charpy\s+impact[:\s]*(\d+\.?\d*\s*(?:J|ft-lb|Nm))"
            ],
            "abrasion_resistance": [
                r"abrasion\s+resistance[:\s]*(\d+\.?\d*\s*(?:mm³|cm³|cycles))",
                r"wear\s+resistance[:\s]*(\d+\.?\d*\s*(?:mm³|cm³|cycles))",
                r"PEI\s+rating[:\s]*(\d+)",
                r"abrasion\s+class[:\s]*(Class\s*[I-V]+|\d+)"
            ],
            "hardness": [
                r"hardness[:\s]*(\d+\.?\d*\s*(?:HRC|HRB|HV|Shore\s*[AD]?))",
                r"Mohs\s+hardness[:\s]*(\d+\.?\d*)",
                r"Shore\s+hardness[:\s]*(\d+\.?\d*\s*(?:Shore\s*[AD]?))",
                r"Rockwell[:\s]*(\d+\.?\d*\s*(?:HRC|HRB))"
            ],
            "compressive_strength": [
                r"compressive\s+strength[:\s]*(\d+\.?\d*\s*(?:N/mm²|MPa|PSI))",
                r"compression\s+strength[:\s]*(\d+\.?\d*\s*(?:N/mm²|MPa|PSI))",
                r"crushing\s+strength[:\s]*(\d+\.?\d*\s*(?:N/mm²|MPa|PSI))"
            ]
        }
        
        # Extract using patterns
        extracted_data = _apply_extraction_patterns(markdown_content, patterns)
        category_result["extracted_values"] = extracted_data["values"]
        category_result["raw_text_matches"] = extracted_data["matches"]
        category_result["extraction_confidence"] = "medium" if extracted_data["values"] else "low"
        
    # Enhanced extraction patterns for thermal properties
    elif category_key == "thermal":
        patterns = {
            "thermal_conductivity": [
                r"thermal\s+conductivity[:\s]*(\d+\.?\d*\s*(?:W/m·K|W/mK|BTU/hr·ft·°F))",
                r"conductivity[:\s]*(\d+\.?\d*\s*(?:W/m·K|W/mK|BTU/hr·ft·°F))",
                r"k-value[:\s]*(\d+\.?\d*\s*(?:W/m·K|W/mK))"
            ],
            "thermal_expansion": [
                r"thermal\s+expansion[:\s]*(\d+\.?\d*\s*(?:×10⁻⁶/K|μm/m·K|in/in/°F))",
                r"coefficient\s+of\s+thermal\s+expansion[:\s]*(\d+\.?\d*\s*(?:×10⁻⁶/K|μm/m·K))",
                r"linear\s+expansion[:\s]*(\d+\.?\d*\s*(?:×10⁻⁶/K|μm/m·K))"
            ],
            "fire_resistance": [
                r"fire\s+resistance[:\s]*(\d+\.?\d*\s*(?:minutes|min|hours|hrs))",
                r"fire\s+rating[:\s]*(Class\s*[ABC\d]+|\d+\s*(?:minutes|min|hours|hrs))",
                r"flame\s+spread[:\s]*(\d+)",
                r"smoke\s+developed[:\s]*(\d+)"
            ],
            "heat_resistance": [
                r"heat\s+resistance[:\s]*(\d+\.?\d*\s*°?[CF])",
                r"maximum\s+(?:operating\s+)?temperature[:\s]*(\d+\.?\d*\s*°?[CF])",
                r"service\s+temperature[:\s]*(\d+\.?\d*\s*°?[CF])"
            ],
            "r_value": [
                r"R-value[:\s]*(\d+\.?\d*)",
                r"thermal\s+resistance[:\s]*(\d+\.?\d*\s*(?:m²·K/W|ft²·°F·hr/BTU))",
                r"insulation\s+value[:\s]*(\d+\.?\d*)"
            ]
        }
        
        extracted_data = _apply_extraction_patterns(markdown_content, patterns)
        category_result["extracted_values"] = extracted_data["values"]
        category_result["raw_text_matches"] = extracted_data["matches"]
        category_result["extraction_confidence"] = "medium" if extracted_data["values"] else "low"
        
    # Enhanced extraction patterns for water/moisture resistance
    elif category_key == "water_moisture":
        patterns = {
            "water_absorption": [
                r"water\s+absorption[:\s]*(\d+\.?\d*\s*%?)",
                r"moisture\s+absorption[:\s]*(\d+\.?\d*\s*%?)",
                r"24-hour\s+water\s+absorption[:\s]*(\d+\.?\d*\s*%?)"
            ],
            "porosity": [
                r"porosity[:\s]*(\d+\.?\d*\s*%?)",
                r"open\s+porosity[:\s]*(\d+\.?\d*\s*%?)",
                r"closed\s+porosity[:\s]*(\d+\.?\d*\s*%?)"
            ],
            "freeze_thaw_resistance": [
                r"freeze[\/\-\s]*thaw\s+(?:resistance|cycles)[:\s]*(\d+\.?\d*)",
                r"frost\s+resistance[:\s]*(Class\s*[ABCDEF\d]+|resistant|non-resistant)",
                r"freeze[\/\-\s]*thaw\s+rating[:\s]*(Class\s*[ABCDEF\d]+|\d+\.?\d*)"
            ],
            "permeability": [
                r"water\s+permeability[:\s]*(\d+\.?\d*\s*(?:mm/s|m/s|cm/s))",
                r"permeability[:\s]*(\d+\.?\d*\s*(?:mm/s|m/s|cm/s))",
                r"moisture\s+permeability[:\s]*(\d+\.?\d*\s*(?:perm|ng/Pa·s·m²))"
            ],
            "waterproof_rating": [
                r"waterproof\s+rating[:\s]*(IP\d+|Class\s*[IV]+|\d+\.?\d*)",
                r"water\s+resistance[:\s]*(IP\d+|Class\s*[IV]+|excellent|good|poor)",
                r"moisture\s+resistance[:\s]*(IP\d+|Class\s*[IV]+|excellent|good|poor)"
            ]
        }
        
        extracted_data = _apply_extraction_patterns(markdown_content, patterns)
        category_result["extracted_values"] = extracted_data["values"]
        category_result["raw_text_matches"] = extracted_data["matches"]
        category_result["extraction_confidence"] = "medium" if extracted_data["values"] else "low"
        
    # Enhanced extraction patterns for chemical/hygiene resistance
    elif category_key == "chemical_hygiene":
        patterns = {
            "chemical_resistance": [
                r"chemical\s+resistance[:\s]*(Class\s*[ABCD\d]+|excellent|good|poor)",
                r"acid\s+resistance[:\s]*(Class\s*[ABCD\d]+|excellent|good|poor)",
                r"alkali\s+resistance[:\s]*(Class\s*[ABCD\d]+|excellent|good|poor)"
            ],
            "stain_resistance": [
                r"stain\s+resistance[:\s]*(Class\s*[1-5\d]+|excellent|good|poor)",
                r"soil\s+resistance[:\s]*(Class\s*[1-5\d]+|excellent|good|poor)",
                r"dirt\s+resistance[:\s]*(Class\s*[1-5\d]+|excellent|good|poor)"
            ],
            "cleaning_properties": [
                r"ease\s+of\s+cleaning[:\s]*(Class\s*[1-5\d]+|excellent|good|poor)",
                r"cleanability[:\s]*(Class\s*[1-5\d]+|excellent|good|poor)",
                r"maintenance[:\s]*(low|medium|high)"
            ],
            "antimicrobial_properties": [
                r"antimicrobial[:\s]*(yes|no|present|Class\s*[ABCD\d]+)",
                r"antibacterial[:\s]*(yes|no|present|Class\s*[ABCD\d]+)",
                r"antifungal[:\s]*(yes|no|present|Class\s*[ABCD\d]+)"
            ],
            "ph_resistance": [
                r"pH\s+resistance[:\s]*(\d+\.?\d*[-–]\d+\.?\d*)",
                r"acid\s+(?:ph|pH)\s+resistance[:\s]*(\d+\.?\d*)",
                r"alkaline\s+(?:ph|pH)\s+resistance[:\s]*(\d+\.?\d*)"
            ]
        }
        
        extracted_data = _apply_extraction_patterns(markdown_content, patterns)
        category_result["extracted_values"] = extracted_data["values"]
        category_result["raw_text_matches"] = extracted_data["matches"]
        category_result["extraction_confidence"] = "medium" if extracted_data["values"] else "low"
        
    # Enhanced extraction patterns for acoustic/electrical properties
    elif category_key == "acoustic_electrical":
        patterns = {
            "sound_absorption": [
                r"sound\s+absorption[:\s]*(\d+\.?\d*\s*(?:dB|NRC))",
                r"noise\s+reduction[:\s]*(\d+\.?\d*\s*(?:dB|NRC))",
                r"acoustic\s+absorption[:\s]*(\d+\.?\d*\s*(?:dB|NRC))"
            ],
            "sound_transmission": [
                r"sound\s+transmission[:\s]*(\d+\.?\d*\s*dB)",
                r"acoustic\s+transmission[:\s]*(\d+\.?\d*\s*dB)",
                r"STC\s+rating[:\s]*(\d+\.?\d*)"
            ],
            "electrical_resistance": [
                r"electrical\s+resistance[:\s]*(\d+\.?\d*\s*(?:Ω|ohm|MΩ))",
                r"resistivity[:\s]*(\d+\.?\d*\s*(?:Ω·m|ohm·m))",
                r"surface\s+resistance[:\s]*(\d+\.?\d*\s*(?:Ω|ohm))"
            ],
            "electrical_conductivity": [
                r"electrical\s+conductivity[:\s]*(\d+\.?\d*\s*(?:S/m|mS/m))",
                r"conductivity[:\s]*(\d+\.?\d*\s*(?:S/m|mS/m))",
                r"conductive[:\s]*(yes|no|Class\s*[ABCD\d]+)"
            ],
            "antistatic_properties": [
                r"antistatic[:\s]*(yes|no|present|Class\s*[ABCD\d]+)",
                r"static\s+dissipative[:\s]*(yes|no|present)",
                r"ESD\s+protection[:\s]*(yes|no|present|Class\s*[ABCD\d]+)"
            ]
        }
        
        extracted_data = _apply_extraction_patterns(markdown_content, patterns)
        category_result["extracted_values"] = extracted_data["values"]
        category_result["raw_text_matches"] = extracted_data["matches"]
        category_result["extraction_confidence"] = "medium" if extracted_data["values"] else "low"
        
    # Enhanced extraction patterns for environmental/sustainability
    elif category_key == "environmental":
        patterns = {
            "recycled_content": [
                r"recycled\s+content[:\s]*(\d+\.?\d*\s*%?)",
                r"post[\/\-\s]*consumer\s+recycled[:\s]*(\d+\.?\d*\s*%?)",
                r"pre[\/\-\s]*consumer\s+recycled[:\s]*(\d+\.?\d*\s*%?)"
            ],
            "voc_emissions": [
                r"VOC\s+emissions[:\s]*(\d+\.?\d*\s*(?:μg/m³|mg/m³|ppm))",
                r"volatile\s+organic\s+compounds[:\s]*(\d+\.?\d*\s*(?:μg/m³|mg/m³|ppm))",
                r"formaldehyde\s+emissions[:\s]*(\d+\.?\d*\s*(?:μg/m³|mg/m³|ppm))"
            ],
            "environmental_certifications": [
                r"GREENGUARD[:\s]*(certified|gold|yes|no)",
                r"LEED[:\s]*(certified|points|yes|no|\d+\.?\d*)",
                r"ENERGY\s*STAR[:\s]*(certified|yes|no)",
                r"Cradle\s*to\s*Cradle[:\s]*(certified|bronze|silver|gold|platinum)"
            ],
            "carbon_footprint": [
                r"carbon\s+footprint[:\s]*(\d+\.?\d*\s*(?:kg\s*CO₂|kgCO2|tCO2))",
                r"embodied\s+carbon[:\s]*(\d+\.?\d*\s*(?:kg\s*CO₂|kgCO2|tCO2))",
                r"GWP[:\s]*(\d+\.?\d*\s*(?:kg\s*CO₂|kgCO2))"
            ],
            "sustainability_rating": [
                r"sustainability\s+rating[:\s]*(A\+*|B\+*|C\+*|D\+*|E\+*|\d+\.?\d*/10)",
                r"environmental\s+rating[:\s]*(A\+*|B\+*|C\+*|D\+*|E\+*|\d+\.?\d*/10)",
                r"green\s+rating[:\s]*(A\+*|B\+*|C\+*|D\+*|E\+*|\d+\.?\d*/10)"
            ]
        }
        
        extracted_data = _apply_extraction_patterns(markdown_content, patterns)
        category_result["extracted_values"] = extracted_data["values"]
        category_result["raw_text_matches"] = extracted_data["matches"]
        category_result["extraction_confidence"] = "medium" if extracted_data["values"] else "low"
        
    # Enhanced extraction patterns for dimensional/aesthetic properties
    elif category_key == "dimensional_aesthetic":
        patterns = {
            "dimensions": [
                r"dimensions[:\s]*(\d+\.?\d*\s*x\s*\d+\.?\d*\s*(?:mm|cm|m|in|ft))",
                r"size[:\s]*(\d+\.?\d*\s*x\s*\d+\.?\d*\s*(?:mm|cm|m|in|ft))",
                r"nominal\s+size[:\s]*(\d+\.?\d*\s*x\s*\d+\.?\d*\s*(?:mm|cm|m|in|ft))"
            ],
            "thickness": [
                r"thickness[:\s]*(\d+\.?\d*\s*(?:mm|cm|m|in))",
                r"depth[:\s]*(\d+\.?\d*\s*(?:mm|cm|m|in))",
                r"height[:\s]*(\d+\.?\d*\s*(?:mm|cm|m|in))"
            ],
            "surface_finish": [
                r"surface\s+finish[:\s]*([a-zA-Z\s]+(?:glossy|matte?|polished|textured|smooth|rough))",
                r"finish[:\s]*([a-zA-Z\s]+(?:glossy|matte?|polished|textured|smooth|rough))",
                r"texture[:\s]*([a-zA-Z\s]+(?:glossy|matte?|polished|textured|smooth|rough))"
            ],
            "color_options": [
                r"color[s]?[:\s]*([a-zA-Z\s,]+)",
                r"colour[s]?[:\s]*([a-zA-Z\s,]+)",
                r"available\s+colors?[:\s]*([a-zA-Z\s,]+)"
            ],
            "pattern_design": [
                r"pattern[:\s]*([a-zA-Z\s]+(?:wood|stone|marble|geometric|floral))",
                r"design[:\s]*([a-zA-Z\s]+(?:wood|stone|marble|geometric|floral))",
                r"motif[:\s]*([a-zA-Z\s]+(?:wood|stone|marble|geometric|floral))"
            ],
            "edge_profile": [
                r"edge\s+(?:profile|treatment)[:\s]*([a-zA-Z\s]+(?:beveled|squared|rounded|chamfered))",
                r"edge\s+finish[:\s]*([a-zA-Z\s]+(?:beveled|squared|rounded|chamfered))",
                r"edge\s+type[:\s]*([a-zA-Z\s]+(?:beveled|squared|rounded|chamfered))"
            ]
        }
        
        extracted_data = _apply_extraction_patterns(markdown_content, patterns)
        category_result["extracted_values"] = extracted_data["values"]
        category_result["raw_text_matches"] = extracted_data["matches"]
        category_result["extraction_confidence"] = "medium" if extracted_data["values"] else "low"
        
    # Basic pattern extraction for other categories (can be enhanced later)
    else:
        # Generic patterns for other categories
        generic_patterns = _get_generic_patterns_for_category(category_key)
        extracted_data = _apply_extraction_patterns(markdown_content, generic_patterns)
        category_result["extracted_values"] = extracted_data["values"]
        category_result["raw_text_matches"] = extracted_data["matches"]
        category_result["extraction_confidence"] = "low" if extracted_data["values"] else "very_low"
        
    return category_result


def _apply_extraction_patterns(text: str, patterns: dict) -> dict:
    """Apply regex patterns to extract values from text."""
    import re
    
    result = {
        "values": {},
        "matches": []
    }
    
    for field, pattern_list in patterns.items():
        for pattern in pattern_list:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                value = match.group(1).strip()
                if value:
                    result["values"][field] = value
                    result["matches"].append({
                        "field": field,
                        "value": value,
                        "pattern": pattern,
                        "context": match.group(0)
                    })
                    break  # Take first match for each field
        
    return result


def _get_generic_patterns_for_category(category_key: str) -> dict:
    """Get basic extraction patterns for different categories."""
    
    base_patterns = {
        "surface_gloss": {
            "gloss_level": [r"gloss[:\s]*(\d+\.?\d*%?)", r"matt|satin|gloss|high\s+gloss"],
            "light_reflectance": [r"reflectance[:\s]*(\d+\.?\d*%?)"]
        },
        "mechanical": {
            "breaking_strength": [r"breaking\s+strength[:\s]*(\d+\.?\d*\s*(?:MPa|N/mm²|psi)?)"],
            "abrasion_resistance": [r"abrasion[:\s]*([A-Z\d]+|Class\s*[ABC\d]+)"]
        },
        "thermal": {
            "thermal_conductivity": [r"thermal\s+conductivity[:\s]*(\d+\.?\d*\s*(?:W/mK|W/m·K)?)"],
            "fire_resistance": [r"fire\s+resistance[:\s]*([A-Z\d]+|Class\s*[ABC\d]+)"]
        },
        "water_moisture": {
            "water_absorption": [r"water\s+absorption[:\s]*(\d+\.?\d*%?)"],
            "porosity": [r"porosity[:\s]*(\d+\.?\d*%?)"]
        },
        "chemical_hygiene": {
            "chemical_resistance": [r"chemical\s+resistance[:\s]*([A-Z\d]+|Class\s*[ABC\d]+)"],
            "stain_resistance": [r"stain\s+resistance[:\s]*([A-Z\d]+|Class\s*[ABC\d]+)"]
        },
        "acoustic_electrical": {
            "sound_absorption": [r"sound\s+absorption[:\s]*(\d+\.?\d*\s*(?:dB|%|NRC)?)"],
            "electrical_conductivity": [r"electrical[:\s]*(\d+\.?\d*\s*(?:S/m|Ω·m)?)", r"antistatic|conductive"]
        },
        "environmental": {
            "recycled_content": [r"recycled[:\s]*(\d+\.?\d*%?)"],
            "emission_levels": [r"emission[s]?[:\s]*(\d+\.?\d*\s*(?:μg/m³|ppm)?)"]
        },
        "dimensional_aesthetic": {
            "dimensions": [r"dimensions?[:\s]*(\d+\.?\d*\s*[×x]\s*\d+\.?\d*\s*(?:mm|cm|inch)?)"],
            "thickness": [r"thickness[:\s]*(\d+\.?\d*\s*(?:mm|cm|inch)?)"],
            "weight": [r"weight[:\s]*(\d+\.?\d*\s*(?:kg/m²|lb/ft²|g/m²)?)"]
        }
    }
    
    return base_patterns.get(category_key, {})