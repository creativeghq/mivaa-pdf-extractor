# Modular PDF Processing Pipeline - API Endpoints Documentation

## Overview

The PDF processing pipeline has been refactored into **5 modular internal API endpoints** that can be called independently or orchestrated together. Each endpoint handles a specific stage of the pipeline with proper error handling, progress tracking, and database sync.

---

## 🏗️ Architecture

### Main Orchestrator
- **Endpoint**: `POST /api/rag/documents/upload`
- **Purpose**: Main entry point that orchestrates the complete pipeline
- **Calls**: All 5 internal endpoints sequentially
- **Infrastructure**: Manages ProgressTracker, CheckpointRecoveryService, job_storage, heartbeat monitoring

### Internal Endpoints
All internal endpoints are prefixed with `/api/internal/` and tagged as "Internal Pipeline Stages" in OpenAPI docs.

---

## 📋 Endpoint Details

### 1. POST /api/internal/classify-images/{job_id}

**Purpose**: Classify images as material or non-material using two-stage AI classification

**AI Processing**:
- **Stage 1 - Llama Vision (Fast & Cheap)**:
  - Model: `meta-llama/Llama-4-Scout-17B-16E-Instruct` (TogetherAI)
  - Classifies images into 3 categories:
    - `material_closeup`: Close-up of material texture/surface/pattern
    - `material_in_situ`: Material shown in application/context
    - `non_material`: NOT material-related (faces, logos, charts, text)
  - Returns confidence score (0-1)
  
- **Stage 2 - Claude Validation (High Quality)**:
  - Model: `Claude Sonnet 4.5` (Anthropic)
  - Only validates images with confidence < threshold (default: 0.7)
  - Provides detailed reasoning for classification
  - Improves accuracy for edge cases

**Request**:
```json
{
  "job_id": "string",
  "extracted_images": [
    {
      "filename": "image1.jpg",
      "path": "/tmp/image1.jpg",
      "page_number": 5,
      "width": 800,
      "height": 600
    }
  ],
  "confidence_threshold": 0.7
}
```

**Response**:
```json
{
  "success": true,
  "material_images": [...],
  "non_material_images": [...],
  "total_classified": 100,
  "material_count": 65,
  "non_material_count": 35
}
```

**Defaults**:
- `confidence_threshold`: 0.7 (70% minimum confidence)
- Concurrency: 5 Llama calls, 2 Claude calls in parallel
- Timeout: 600s (10 minutes)

**Progress**: Updates job to 60%

---

### 2. POST /api/internal/upload-images/{job_id}

**Purpose**: Upload material images to Supabase Storage

**Processing**:
- Uploads ONLY material images (filtered by Stage 1)
- Storage bucket: `material-images`
- Path format: `{document_id}/{filename}`
- Parallel uploads with rate limiting
- Generates public URLs for each image

**Request**:
```json
{
  "job_id": "string",
  "material_images": [
    {
      "filename": "material1.jpg",
      "path": "/tmp/material1.jpg",
      "page_number": 5
    }
  ],
  "document_id": "uuid"
}
```

**Response**:
```json
{
  "success": true,
  "uploaded_images": [
    {
      "filename": "material1.jpg",
      "storage_url": "https://...supabase.co/storage/v1/object/public/material-images/...",
      "page_number": 5
    }
  ],
  "uploaded_count": 65,
  "failed_count": 0
}
```

**Defaults**:
- Concurrency: 5 parallel uploads
- Timeout: 600s (10 minutes)
- Retry: 3 attempts with exponential backoff

**Progress**: Updates job to 65%

---

### 3. POST /api/internal/save-images-db/{job_id}

**Purpose**: Save images to database and generate CLIP embeddings

**AI Processing**:
- **CLIP Embeddings (5 types per image)**:
  - Model: OpenAI CLIP (via RealEmbeddingsService)
  - Embedding types:
    1. **Visual CLIP** (512D): General visual features
    2. **Color CLIP** (512D): Color-focused embedding
    3. **Texture CLIP** (512D): Texture-focused embedding
    4. **Application CLIP** (512D): Application/context-focused embedding
    5. **Material CLIP** (512D): Material-specific embedding
  
- **Storage**:
  - Saves to `document_images` table (PostgreSQL)
  - Saves to `embeddings` table (PostgreSQL)
  - Upserts to VECS collections (vector search)

**Request**:
```json
{
  "job_id": "string",
  "material_images": [
    {
      "filename": "material1.jpg",
      "storage_url": "https://...",
      "page_number": 5
    }
  ],
  "document_id": "uuid",
  "workspace_id": "uuid"
}
```

**Response**:
```json
{
  "success": true,
  "images_saved": 65,
  "clip_embeddings_generated": 325
}
```

**Defaults**:
- Embeddings per image: 5 (visual, color, texture, application, material)
- Timeout: 600s (10 minutes)
- Batch processing: All embeddings generated in parallel

**Progress**: Updates job to 75%

---

### 4. POST /api/internal/create-chunks/{job_id}

**Purpose**: Create semantic chunks from extracted text and generate text embeddings

**AI Processing**:
- **Text Chunking**:
  - Method: Semantic chunking with overlap
  - Respects product boundaries (doesn't mix content from different products)
  - Excludes index/navigation pages

- **Text Embeddings**:
  - Model: OpenAI `text-embedding-3-small` (via RealEmbeddingsService)
  - Dimension: 1536D
  - One embedding per chunk

- **Relationships**:
  - Creates chunk-to-product relationships
  - Links chunks to products based on page ranges

**Request**:
```json
{
  "job_id": "string",
  "document_id": "uuid",
  "workspace_id": "uuid",
  "extracted_text": "Full PDF text content...",
  "product_ids": ["uuid1", "uuid2"],
  "chunk_size": 512,
  "chunk_overlap": 50
}
```

**Response**:
```json
{
  "success": true,
  "chunks_created": 107,
  "embeddings_generated": 107,
  "relationships_created": 163
}
```

**Defaults**:
- `chunk_size`: 512 characters
- `chunk_overlap`: 50 characters
- Timeout: 600s (10 minutes)
- Batch processing: All embeddings generated in parallel

**Progress**: Updates job to 85%

---

### 5. POST /api/internal/create-relationships/{job_id}

**Purpose**: Create chunk-image and product-image relationships

**Processing**:
- **Chunk-Image Relationships**:
  - Method: Cosine similarity between chunk text embeddings and image CLIP embeddings
  - Threshold: 0.5 (default)
  - Links semantically related chunks and images

- **Product-Image Relationships**:
  - Method: Page range matching
  - Links images to products based on page numbers
  - Uses product page ranges from discovery

**Request**:
```json
{
  "job_id": "string",
  "document_id": "uuid",
  "product_ids": ["uuid1", "uuid2"],
  "similarity_threshold": 0.5
}
```

**Response**:
```json
{
  "success": true,
  "chunk_image_relationships": 245,
  "product_image_relationships": 132
}
```

**Defaults**:
- `similarity_threshold`: 0.5 (50% minimum similarity)
- Timeout: 600s (10 minutes)

**Progress**: Updates job to 100%

---

## 🎯 Default Behavior

### Focused Extraction (Default: ENABLED)
- **Purpose**: Process ONLY material-related images, skip non-material content
- **Default**: `focused_extraction=True`
- **Categories**: `['products']` (only product-related content)
- **Behavior**:
  - Classifies ALL images with AI
  - Uploads ONLY material images to storage
  - Saves ONLY material images to database
  - Generates CLIP embeddings ONLY for material images
  - Skips non-material images (faces, logos, charts, text)

### Extract Categories
- **Options**: `['products', 'certificates', 'logos', 'specifications', 'all']`
- **Default**: `['products']`
- **Behavior**:
  - `products`: Only product pages and images
  - `certificates`: Only certificate pages and images
  - `logos`: Only logo pages and images
  - `specifications`: Only specification pages and images
  - `all`: Process entire PDF (disables focused extraction)

### AI Models Used
1. **Product Discovery**: Claude Sonnet 4.5 or GPT-5 (configurable)
2. **Image Classification**: Llama 4 Scout 17B Vision → Claude Sonnet 4.5 (validation)
3. **CLIP Embeddings**: OpenAI CLIP (5 types per image)
4. **Text Embeddings**: OpenAI text-embedding-3-small (1536D)

### Thresholds
- **Image Classification Confidence**: 0.7 (70% minimum)
- **Relationship Similarity**: 0.5 (50% minimum)
- **Chunk Size**: 512 characters
- **Chunk Overlap**: 50 characters

### Retry & Timeout
- **Max Retries**: 3 attempts per endpoint
- **Timeout**: 600s (10 minutes) per endpoint
- **Backoff**: Exponential (2^attempt seconds)

---

## 📊 Progress Tracking

### Pipeline Stages (50-100%)
- **50-60%**: Image Classification (Llama + Claude)
- **60-65%**: Image Upload (Supabase Storage)
- **65-75%**: Save Images & CLIP Embeddings (5 per image)
- **75-85%**: Chunking & Text Embeddings
- **85-100%**: Relationships (chunk-image, product-image)

### Infrastructure Integration
- **ProgressTracker**: Real-time progress updates, heartbeat monitoring (30s interval)
- **CheckpointRecoveryService**: Checkpoint creation at each stage for recovery
- **JobTracker**: Database sync after each stage
- **job_storage**: In-memory tracking for fast access
- **Sentry**: Error tracking and monitoring

---

## ✅ Summary

### What Each Endpoint Does

1. **classify-images**: AI classification (Llama → Claude) to filter material vs non-material images
2. **upload-images**: Upload material images to Supabase Storage
3. **save-images-db**: Save to database + generate 5 CLIP embeddings per image
4. **create-chunks**: Semantic chunking + text embeddings + chunk-product relationships
5. **create-relationships**: Chunk-image and product-image relationships via similarity

### Default Processing Flow

1. Extract ALL images from PDF
2. Classify ALL images with AI (Llama + Claude)
3. Upload ONLY material images to storage
4. Save ONLY material images to database
5. Generate 5 CLIP embeddings per material image
6. Create semantic chunks from text
7. Generate text embeddings for chunks
8. Create relationships between chunks, images, and products

### Key Features

- ✅ **Focused extraction by default** (only material images)
- ✅ **Two-stage AI classification** (Llama → Claude validation)
- ✅ **5 CLIP embeddings per image** (visual, color, texture, application, material)
- ✅ **Semantic chunking** with product boundary respect
- ✅ **Comprehensive progress tracking** (5% increments)
- ✅ **Checkpoint creation** at each stage for recovery
- ✅ **Retry logic** with exponential backoff
- ✅ **Error handling** with Sentry integration
- ✅ **Real-time database sync** after each stage

