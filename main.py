from fastapi import FastAPI, UploadFile
from extractor import extract_pdf_to_markdown, extract_pdf_tables, extract_json_and_images
from pathlib import Path
from tempfile import NamedTemporaryFile
import shutil
import os, datetime, io, zipfile
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.post("/extract/markdown")
def extract_markdown(file: UploadFile, page_number:int=None):
    
    return extract_pdf_to_markdown(save_upload_file_tmp((file)),page_number)

@app.post("/extract/tables")
def extract_table(file: UploadFile, page_number:int=None):
    
    out_dir= create_output_dir()
    file_name, file_extension = os.path.splitext(file.filename)
    extract_pdf_tables(save_upload_file_tmp((file)),page_number,out_dir)
    zip_stream = create_zip_stream(out_dir)
    
    return StreamingResponse(zip_stream, media_type="application/octet-stream", headers={"Content-Disposition": "attachment;  filename="+file_name+"_csv"+".zip"})

@app.post("/extract/images")
def extract_images(file: UploadFile, page_number:int=None, include_functional_metadata:bool=False, extract_mode:str="comprehensive"):
    """
    Extract images and JSON metadata from PDFs, optionally including functional metadata.
    
    Args:
        file: PDF file upload
        page_number: Specific page to extract (optional, extracts all pages if None)
        include_functional_metadata: Whether to include functional metadata extraction
        extract_mode: Functional metadata extraction mode - "comprehensive", "safety", etc.
    """
    
    out_dir= create_output_dir()
    file_name, file_extension = os.path.splitext(file.filename)
    
    # Extract standard images and JSON
    extract_json_and_images(save_upload_file_tmp((file)),out_dir,page_number)
    
    # Optionally add functional metadata
    if include_functional_metadata:
        try:
            from extractor import extract_functional_metadata as extract_func
            file_path = save_upload_file_tmp(file)
            functional_metadata = extract_func(file_path, page_number, extract_mode)
            
            # Save functional metadata as additional JSON file
            import json
            metadata_file = os.path.join(out_dir, 'functional_metadata.json')
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "functional_metadata": functional_metadata,
                    "processing_info": {
                        "filename": file.filename,
                        "page_number": page_number,
                        "extract_mode": extract_mode
                    }
                }, f, indent=2)
            
            # Clean up temp file
            if file_path.exists():
                file_path.unlink()
                
        except Exception as e:
            # Create error file instead
            error_file = os.path.join(out_dir, 'functional_metadata_error.json')
            with open(error_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "error": str(e),
                    "processing_info": {
                        "filename": file.filename,
                        "page_number": page_number,
                        "extract_mode": extract_mode
                    }
                }, f, indent=2)
    
    zip_stream = create_zip_stream(out_dir)
    return StreamingResponse(zip_stream, media_type="application/octet-stream", headers={"Content-Disposition": "attachment; filename="+file_name+".zip"})

def save_upload_file_tmp(upload_file: UploadFile) -> Path:
    try:
        suffix = Path(upload_file.filename).suffix
        with NamedTemporaryFile(delete=False,prefix=upload_file.filename, suffix=suffix) as tmp:
            shutil.copyfileobj(upload_file.file, tmp)
            tmp_path = Path(tmp.name)
    finally:
        upload_file.file.close()
    return tmp_path

def create_output_dir():
    output_dir = os.path.join('output', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(output_dir)
    return output_dir

def _structure_metadata_for_frontend(functional_metadata, filename, page_number, extract_mode):
    """
    Structure functional metadata for optimal frontend Knowledge Base display.
    
    Args:
        functional_metadata: Raw extraction results from extract_functional_metadata()
        filename: PDF filename
        page_number: Extracted page number
        extract_mode: Extraction mode used
        
    Returns:
        dict: Structured metadata optimized for frontend consumption
    """
    
    # Category display mapping for frontend
    category_display_names = {
        "slip_safety": "🦶 Slip/Safety Ratings",
        "surface_gloss": "✨ Surface Gloss & Reflectivity",
        "mechanical": "🔧 Mechanical Properties",
        "thermal": "🌡️ Thermal Properties",
        "water_moisture": "💧 Water/Moisture Resistance",
        "chemical_hygiene": "🧪 Chemical/Hygiene Resistance",
        "acoustic_electrical": "⚡ Acoustic/Electrical Properties",
        "environmental": "🌱 Environmental/Sustainability",
        "dimensional_aesthetic": "📐 Dimensional/Aesthetic"
    }
    
    # Structure metadata for Knowledge Base display
    structured_output = {
        "document_info": {
            "filename": filename,
            "page_number": page_number,
            "extract_mode": extract_mode,
            "processing_timestamp": functional_metadata.get("processing_timestamp"),
            "total_categories_found": len([cat for cat in functional_metadata.get("categories", {}).values()
                                         if cat.get("properties")])
        },
        "functional_properties": {},
        "summary": {
            "extraction_confidence": functional_metadata.get("extraction_confidence", "low"),
            "categories_with_data": [],
            "key_properties_found": [],
            "suggested_applications": []
        }
    }
    
    # Process each category for frontend display
    for category_key, category_data in functional_metadata.get("categories", {}).items():
        if not category_data.get("properties"):
            continue
            
        display_name = category_display_names.get(category_key, category_key.replace("_", " ").title())
        
        # Structure category data for frontend cards/panels
        structured_category = {
            "display_name": display_name,
            "category_key": category_key,
            "extraction_confidence": category_data.get("extraction_confidence", "low"),
            "properties_count": len(category_data.get("properties", {})),
            "properties": {},
            "highlights": [],  # Key properties for quick display
            "technical_details": []  # Detailed properties for expanded view
        }
        
        # Process properties with frontend-friendly formatting
        for prop_name, prop_values in category_data.get("properties", {}).items():
            if not prop_values:
                continue
                
            # Format property for display
            formatted_prop = {
                "name": prop_name.replace("_", " ").title(),
                "key": prop_name,
                "values": prop_values,
                "primary_value": prop_values[0] if prop_values else None,
                "unit_detected": _extract_unit_from_value(prop_values[0] if prop_values else ""),
                "display_priority": _get_property_display_priority(category_key, prop_name)
            }
            
            structured_category["properties"][prop_name] = formatted_prop
            
            # Add to highlights if high priority
            if formatted_prop["display_priority"] == "high":
                structured_category["highlights"].append({
                    "name": formatted_prop["name"],
                    "value": formatted_prop["primary_value"],
                    "key": prop_name
                })
            else:
                structured_category["technical_details"].append({
                    "name": formatted_prop["name"],
                    "value": formatted_prop["primary_value"],
                    "key": prop_name
                })
        
        structured_output["functional_properties"][category_key] = structured_category
        structured_output["summary"]["categories_with_data"].append(display_name)
        
        # Add key properties to summary
        for highlight in structured_category["highlights"][:2]:  # Top 2 per category
            structured_output["summary"]["key_properties_found"].append(
                f"{highlight['name']}: {highlight['value']}"
            )
    
    # Generate suggested applications based on found properties
    structured_output["summary"]["suggested_applications"] = _generate_application_suggestions(
        structured_output["functional_properties"]
    )
    
    return structured_output

def _extract_unit_from_value(value_str):
    """Extract unit from a property value string."""
    if not value_str:
        return ""
    
    import re
    # Common unit patterns
    unit_patterns = [
        r'(\d+\.?\d*)\s*([A-Za-z/°²³µ·]+)',  # Numbers with units
        r'(Class\s*[A-Z\d]+)',  # Class ratings
        r'(IP\d+)',  # IP ratings
    ]
    
    for pattern in unit_patterns:
        match = re.search(pattern, str(value_str))
        if match:
            return match.group(2) if len(match.groups()) > 1 else ""
    return ""

def _get_property_display_priority(category_key, prop_name):
    """Determine display priority for properties in frontend."""
    
    # High priority properties for each category (shown as highlights)
    high_priority_props = {
        "slip_safety": ["slip_resistance", "pendulum_test_value"],
        "surface_gloss": ["gloss_level", "light_reflectance_value"],
        "mechanical": ["breaking_strength", "abrasion_resistance"],
        "thermal": ["thermal_conductivity", "fire_resistance"],
        "water_moisture": ["water_absorption", "freeze_thaw_resistance"],
        "chemical_hygiene": ["chemical_resistance", "stain_resistance"],
        "acoustic_electrical": ["sound_absorption", "electrical_resistance"],
        "environmental": ["recycled_content", "environmental_certifications"],
        "dimensional_aesthetic": ["dimensions", "surface_finish"]
    }
    
    if prop_name in high_priority_props.get(category_key, []):
        return "high"
    return "medium"

def _generate_application_suggestions(functional_properties):
    """Generate application suggestions based on extracted properties."""
    suggestions = []
    
    # Simple rule-based suggestions
    if "slip_safety" in functional_properties:
        if functional_properties["slip_safety"].get("highlights"):
            suggestions.append("High-traffic commercial flooring")
    
    if "water_moisture" in functional_properties:
        if functional_properties["water_moisture"].get("highlights"):
            suggestions.append("Bathroom/wet area applications")
    
    if "thermal" in functional_properties:
        if functional_properties["thermal"].get("highlights"):
            suggestions.append("High-temperature environments")
    
    if "environmental" in functional_properties:
        if functional_properties["environmental"].get("highlights"):
            suggestions.append("Green building projects")
    
    return suggestions[:3]  # Limit to top 3 suggestions

def create_zip_stream(output_dir):

    zip_stream = io.BytesIO()  
    with zipfile.ZipFile(zip_stream, "w") as zf:  
        for root, _, files in os.walk(output_dir):  
            for file in files:  
                file_path = os.path.join(root, file)  
                zf.write(file_path, os.path.relpath(file_path, output_dir))  
    zip_stream.seek(0) 
    return zip_stream