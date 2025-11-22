# PDF Processing Pipeline - Refactored Modular Architecture

## Overview

The PDF processing pipeline has been refactored from a monolithic 2900+ line function into modular services and API endpoints. Each stage can now be tested, debugged, and retried independently.

## Architecture

### Before (Monolithic)
```
POST /api/rag/documents/upload
  └─ Single massive function (2900+ lines)
     ├─ PDF extraction
     ├─ Image classification
     ├─ Image upload
     ├─ DB save
     ├─ CLIP generation
     ├─ Product detection
     ├─ Chunking
     └─ Text embeddings
     
❌ Problems:
- Hard to debug (which stage failed?)
- Can't retry individual stages
- 2900+ lines in one function
- Difficult to test
```

### After (Modular)
```
POST /api/rag/documents/upload (Orchestrator)
  ↓
  ├─ POST /api/internal/classify-images/{job_id}
  │  └─ ImageProcessingService.classify_images()
  │
  ├─ POST /api/internal/upload-images/{job_id}
  │  └─ ImageProcessingService.upload_images_to_storage()
  │
  ├─ POST /api/internal/save-images-db/{job_id}
  │  └─ ImageProcessingService.save_images_and_generate_clips()
  │
  ├─ POST /api/internal/create-chunks/{job_id}
  │  └─ ChunkingService.create_chunks_and_embeddings()
  │
  └─ POST /api/internal/create-relationships/{job_id}
     └─ RelevancyService.create_all_relationships()

✅ Benefits:
- Each stage is independent
- Failed stages can be retried
- Easy to debug (clear stage boundaries)
- Testable (unit tests per service)
- Maintainable (200 lines per service)
```

## Services

### 1. ImageProcessingService
**File:** `app/services/image_processing_service.py`

**Methods:**
- `classify_images()` - Llama Vision + Claude validation
- `upload_images_to_storage()` - Upload to Supabase Storage
- `save_images_and_generate_clips()` - DB save + CLIP embeddings

**Responsibilities:**
- Extract images from PDF
- Classify as material/non-material
- Upload material images to storage
- Save to database
- Generate CLIP embeddings (visual + specialized)

### 2. ChunkingService
**File:** `app/services/chunking_service.py`

**Methods:**
- `create_chunks_and_embeddings()` - Create chunks + text embeddings

**Responsibilities:**
- Create semantic chunks from text
- Generate text embeddings
- Create chunk-to-product relationships

### 3. RelevancyService
**File:** `app/services/relevancy_service.py`

**Methods:**
- `create_chunk_image_relationships()` - Based on embedding similarity
- `create_product_image_relationships()` - Based on page ranges
- `create_all_relationships()` - Orchestrate all relationships

**Responsibilities:**
- Calculate embedding similarity
- Create chunk-to-image relationships
- Create product-to-image relationships
- Manage relevancy scores

## API Endpoints

### Internal Endpoints (Modular Pipeline Stages)

#### 1. Classify Images
```http
POST /api/internal/classify-images/{job_id}
Content-Type: application/json

{
  "job_id": "uuid",
  "extracted_images": [...],
  "confidence_threshold": 0.7
}

Response:
{
  "success": true,
  "material_images": [...],
  "non_material_images": [...],
  "total_classified": 132,
  "material_count": 132,
  "non_material_count": 0
}
```

#### 2. Upload Images
```http
POST /api/internal/upload-images/{job_id}
Content-Type: application/json

{
  "job_id": "uuid",
  "material_images": [...],
  "document_id": "uuid"
}

Response:
{
  "success": true,
  "uploaded_images": [...],
  "uploaded_count": 132,
  "failed_count": 0
}
```

#### 3. Save Images & Generate CLIP
```http
POST /api/internal/save-images-db/{job_id}
Content-Type: application/json

{
  "job_id": "uuid",
  "material_images": [...],
  "document_id": "uuid",
  "workspace_id": "uuid"
}

Response:
{
  "success": true,
  "images_saved": 132,
  "clip_embeddings_generated": 132
}
```

#### 4. Create Chunks
```http
POST /api/internal/create-chunks/{job_id}
Content-Type: application/json

{
  "job_id": "uuid",
  "document_id": "uuid",
  "workspace_id": "uuid",
  "extracted_text": "...",
  "product_ids": ["uuid1", "uuid2"],
  "chunk_size": 512,
  "chunk_overlap": 50
}

Response:
{
  "success": true,
  "chunks_created": 107,
  "embeddings_generated": 107,
  "relationships_created": 163
}
```

