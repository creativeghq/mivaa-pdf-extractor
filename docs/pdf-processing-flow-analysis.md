# PDF Processing Flow - Complete Analysis

**Date**: 2025-11-23  
**Purpose**: Comprehensive step-by-step analysis of PDF processing pipeline with all models, API calls, and data transformations

---

## 🎯 EXECUTIVE SUMMARY

### Current Status
- ✅ **Metadata Extraction**: FIXED (`_ensure_properties_exist` error resolved)
- ❌ **SigLIP Embeddings**: BROKEN (compatibility issue with sentence-transformers 3.0.0)
- ✅ **CLIP Fallback**: WORKING (all images falling back to CLIP)
- ❌ **Image Count Mismatch**: Expected 388, Got 256 (132 missing)

### Critical Issues Found
1. **SigLIP Model Incompatibility**: `sentence-transformers` 3.0.0 + `transformers` 4.41.0 cannot load `google/siglip-so400m-patch14-384` due to `hidden_size` attribute error
2. **Missing Images**: 132 images not saved to database (likely classification or upload failures)
3. **No Relationships**: 0 chunk-image and product-image relationships created

---

## 📊 COMPLETE PIPELINE FLOW

### Stage 0: Upload & Initialization (0-10%)
**Entry Point**: `POST /api/rag/documents/upload`  
**File**: `app/api/rag_routes.py:353`

**Input**:
- PDF file or URL
- Processing mode: `quick` | `standard` | `deep`
- Extract categories: `["products"]` (default)
- Discovery model: `claude-sonnet-4-20250514` | `gpt-5` | `gpt-4o`

**Process**:
1. Upload PDF to Supabase Storage (`pdf-documents` bucket)
2. Create `documents` table entry
3. Create `background_jobs` table entry with job_id
4. Extract PDF text and images using PyMuPDF

**Output**:
- `job_id`: UUID for tracking
- `document_id`: UUID for document
- `extracted_text`: Full PDF text
- `extracted_images`: List of image dicts with base64 data

**Models Used**: None (file processing only)

---

### Stage 0B: Product Discovery (10-50%)
**Service**: `ProductDiscoveryService`  
**File**: `app/services/product_discovery_service.py`

**Input**:
- PDF text (first 10,000 chars)
- Extracted images (all pages)
- Discovery model (Claude Sonnet 4.5 or GPT-5)

**Process**:
1. **AI Product Identification** (Claude/GPT-5):
   - Analyze PDF text for product names, page ranges, metadata
   - Identify product boundaries and variants
   - Extract initial metadata (dimensions, colors, materials)
   
2. **Product Creation**:
   - Insert into `products` table
   - Store metadata in `products.metadata` JSONB field
   - Link to workspace and document

**API Calls**:
- **Claude Sonnet 4.5**: `https://api.anthropic.com/v1/messages`
  - Model: `claude-sonnet-4-20250514`
  - Max tokens: 4096
  - Temperature: 0.1
- **GPT-5**: `https://api.openai.com/v1/chat/completions`
  - Model: `gpt-5`
  - Max tokens: 4096
  - Temperature: 0.1

**Output**:
- `product_ids`: List of created product UUIDs
- `products_count`: Number of products discovered
- Checkpoint: `PRODUCTS_DETECTED`

**Expected**: 11 products for NOVA PDF  
**Actual**: ✅ 11 products created

---

### Stage 1: Image Classification (50-60%)
**Service**: `PipelineOrchestrator` → Internal API  
**Endpoint**: `POST /api/internal/classify-images/{job_id}`  
**File**: `app/api/internal_routes.py`

**Input**:
- `extracted_images`: List of 388 image dicts
- `confidence_threshold`: 0.7
- `focused_extraction`: True (only material images)

**Process**:
1. **Primary Classification** (Llama Vision):
   - Model: `meta-llama/Llama-4-Scout-17B-16E-Instruct`
   - API: TogetherAI `https://api.together.xyz/v1/chat/completions`
   - Classifies each image as: `material` | `non_material`
   - Returns confidence score (0.0-1.0)

2. **Validation** (Claude - only for low confidence < 0.7):
   - Model: `claude-sonnet-4-20250514`
   - API: Anthropic `https://api.anthropic.com/v1/messages`
   - Re-classifies low-confidence images
   - Final decision based on consensus

**API Calls**:
- **Llama Vision** (TogetherAI):
  ```json
  {
    "model": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Is this a material product image?"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
      ]
    }],
    "max_tokens": 512,
    "temperature": 0.1
  }
  ```

**Output**:
- `material_images`: List of classified material images
- `non_material_images`: List of non-material images
- Checkpoint: `IMAGES_EXTRACTED`

**Expected**: ~260 material images  
**Actual**: ⚠️ Unknown (need to check classification logs)

---

### Stage 2: Image Upload (60-65%)
**Endpoint**: `POST /api/internal/upload-images/{job_id}`  
**File**: `app/api/internal_routes.py:258`

**Input**:
- `material_images`: Classified material images with base64 data
- `document_id`: Document UUID

**Process**:
1. Upload each image to Supabase Storage
2. Bucket: `material-images`
3. Path: `{workspace_id}/{document_id}/{image_index}.jpg`
4. Get public URL for each uploaded image

**API Calls**:
- **Supabase Storage API**: `POST /storage/v1/object/{bucket}/{path}`

**Output**:
- `uploaded_images`: List with storage URLs added
- `failed_uploads`: Count of failed uploads

**Expected**: ~260 uploaded images
**Actual**: ⚠️ Need to verify upload logs

---

### Stage 3: Save Images & Generate CLIP Embeddings (65-75%)
**Endpoint**: `POST /api/internal/save-images-db/{job_id}`
**File**: `app/api/internal_routes.py:317`
**Service**: `RealEmbeddingsService`

**Input**:
- `material_images`: Uploaded images with storage URLs
- `document_id`: Document UUID
- `workspace_id`: Workspace UUID

**Process**:
1. **Save to Database**:
   - Insert into `document_images` table
   - Fields: `id`, `document_id`, `workspace_id`, `image_url`, `page_number`, `metadata`

2. **Generate Embeddings** (per image):
   - Call `RealEmbeddingsService.generate_all_embeddings()`
   - Entity type: `"image"`
   - Text content: `""` (empty for images)
   - Image data: base64 encoded

**Embedding Types Generated (8 total per image)**:

#### 1. Text Embedding (1536D) - SKIPPED for images
- Model: `text-embedding-3-small` (OpenAI)
- API: `https://api.openai.com/v1/embeddings`
- Input: Empty string for images
- Output: Not generated for images

#### 2. Visual Embedding (512D) - PRIMARY
- **Primary Model**: `google/siglip-so400m-patch14-384` (SigLIP)
- **Library**: `sentence-transformers` 3.0.0
- **Status**: ❌ BROKEN - `'SiglipConfig' object has no attribute 'hidden_size'`
- **Error Location**: `sentence_transformers/models/Transformer.py:133`
- **Root Cause**: SigLIP model config has `vision_config.hidden_size` not `config.hidden_size`

#### 3. Visual Embedding (512D) - FALLBACK
- **Fallback Model**: `openai/clip-vit-base-patch32` (CLIP)
- **Library**: `transformers` 4.41.0 + `torch`
- **Status**: ✅ WORKING
- **Process**:
  1. Load CLIP model and processor
  2. Convert base64 → PIL Image → RGB
  3. Process image through CLIP vision encoder
  4. Extract image features (512D)
  5. L2 normalize to unit vector
- **Output**: 512D float array

#### 4-7. Specialized Visual Embeddings (4 × 512D)
- **Types**: `color_clip_512`, `texture_clip_512`, `style_clip_512`, `material_clip_512`
- **Current Implementation**: All use same base visual embedding (duplicates)
- **Status**: ✅ WORKING (but redundant)
- **Note**: Should use CLIP text encoder with specialized prompts for true specialization

#### 8. Multimodal Fusion (2048D)
- **Input**: text_1536 + visual_512
- **Process**: Concatenate and pad to 2048D
- **Status**: ⚠️ PARTIAL (only works if text embedding exists)
- **For Images**: Not generated (no text content)

**API Calls**:
- **SigLIP** (sentence-transformers): Local model loading - FAILS
- **CLIP** (transformers): Local model loading - WORKS
- **OpenAI Embeddings**: `https://api.openai.com/v1/embeddings` - SKIPPED for images

**Database Operations**:
1. Insert into `document_images` table
2. Insert into `embeddings` table (8 rows per image):
   - `entity_id`: image_id
   - `entity_type`: "image"
   - `embedding_type`: "visual_512", "color_clip_512", etc.
   - `embedding`: float array
   - `dimension`: 512 or 1536 or 2048
   - `model`: "clip-vit-base-patch32" (fallback)

**Output**:
- `images_saved`: Count of images saved to DB
- `clip_embeddings`: Total embeddings generated (should be images_saved × 5)
- Checkpoint: `IMAGE_EMBEDDINGS_GENERATED`

**Expected**:
- Images saved: 256
- CLIP embeddings: 256 × 5 = 1,280

**Actual**:
- Images saved: 256 ✅
- Embeddings in DB: 107 ❌ (WRONG - should be 1,280)
- **CRITICAL BUG**: Only text embeddings (107 chunks) saved, no image embeddings!

---

### Stage 4: Create Chunks (75-85%)
**Endpoint**: `POST /api/internal/create-chunks/{job_id}`
**File**: `app/api/internal_routes.py`
**Service**: `UnifiedChunkingService`

**Input**:
- `extracted_text`: Full PDF text
- `product_ids`: List of product UUIDs
- `chunk_size`: 512 tokens
- `chunk_overlap`: 50 tokens

**Process**:
1. **Semantic Chunking**:
   - Split text by product boundaries
   - Create chunks respecting sentence boundaries
   - Enrich with context (product name, page number)

2. **Generate Text Embeddings**:
   - Call `RealEmbeddingsService.generate_all_embeddings()`
   - Entity type: `"chunk"`
   - Text content: chunk text
   - No image data

3. **Link to Products**:
   - Create `chunk_product_relationships` entries
   - Link each chunk to relevant product(s)

**Embedding Types Generated (3 total per chunk)**:

#### 1. Text Embedding (1536D)
- Model: `text-embedding-3-small` (OpenAI)
- API: `https://api.openai.com/v1/embeddings`
- Input: Chunk text (max 8,191 chars)
- Output: 1536D float array
- Status: ✅ WORKING

#### 2. Visual Embedding (512D)
- Status: ⏭️ SKIPPED (no image data for chunks)

#### 3. Multimodal Fusion (2048D)
- Status: ⏭️ SKIPPED (no visual embedding)

**Database Operations**:
1. Insert into `document_chunks` table
2. Insert into `embeddings` table (1 row per chunk):
   - `entity_id`: chunk_id
   - `entity_type`: "chunk"
   - `embedding_type`: "text_1536"
   - `embedding`: float array
   - `dimension`: 1536
3. Insert into `chunk_product_relationships` table

**Output**:
- `chunks_created`: 107
- `text_embeddings`: 107
- `chunk_product_relationships`: 0 ❌ (SHOULD BE > 0)
- Checkpoint: `CHUNKS_CREATED`

**Expected**: 107 chunks with 107 text embeddings
**Actual**: ✅ 107 chunks, ✅ 107 embeddings, ❌ 0 relationships

---

### Stage 5: Create Relationships (85-100%)
**Endpoint**: `POST /api/internal/create-relationships/{job_id}`
**File**: `app/api/internal_routes.py`
**Service**: `RelevancyService`

**Input**:
- `document_id`: Document UUID
- `product_ids`: List of product UUIDs

**Process**:
1. **Chunk-Image Relationships**:
   - Find semantic similarity between chunks and images
   - Use cosine similarity on embeddings
   - Threshold: 0.5
   - Create `chunk_image_relationships` entries

2. **Product-Image Relationships**:
   - Link images to products based on page ranges
   - Use product metadata (page_start, page_end)
   - Create `product_image_relationships` entries

**Database Operations**:
1. Query `document_chunks` and `document_images`
2. Query `embeddings` for both
3. Calculate cosine similarity
4. Insert into `chunk_image_relationships`
5. Insert into `product_image_relationships`

**Output**:
- `chunk_image_relationships`: 0 ❌ (SHOULD BE > 0)
- `product_image_relationships`: 0 ❌ (SHOULD BE > 0)
- Checkpoint: `COMPLETED`

**Expected**:
- Chunk-Image: ~500 relationships
- Product-Image: ~260 relationships

**Actual**:
- Chunk-Image: 0 ❌
- Product-Image: 0 ❌

**Root Cause**: No image embeddings in database → cannot calculate similarity

---

## 🔍 ROOT CAUSE ANALYSIS

### Issue #1: SigLIP Model Loading Failure

**Error**:
```
AttributeError: 'SiglipConfig' object has no attribute 'hidden_size'
```

**Location**: `sentence_transformers/models/Transformer.py:133`

**Code**:
```python
def get_word_embedding_dimension(self):
    return self.auto_model.config.hidden_size  # ❌ FAILS for SigLIP
```

**Why It Fails**:
- SigLIP has composite config: `SiglipConfig(text_config, vision_config)`
- `hidden_size` is in `config.vision_config.hidden_size` NOT `config.hidden_size`
- `sentence-transformers` 3.0.0 expects flat config structure
- This is a known incompatibility between sentence-transformers and multi-modal models

**Fix Options**:
1. **Downgrade sentence-transformers** to 2.x (may break other features)
2. **Use transformers directly** instead of sentence-transformers
3. **Patch SiglipConfig** to add `hidden_size` property
4. **Switch to different SigLIP wrapper** (e.g., HuggingFace Transformers only)

---

### Issue #2: Missing Image Embeddings in Database

**Expected**: 256 images × 5 embeddings = 1,280 image embeddings
**Actual**: 0 image embeddings in `embeddings` table

**Possible Causes**:
1. ❌ Embedding generation fails silently
2. ❌ Database insert fails without error logging
3. ❌ Transaction rollback due to error
4. ❌ Wrong table being queried

**Investigation Needed**:
- Check `save_images_and_generate_clips()` implementation
- Verify database insert logic
- Check for silent exceptions

---

### Issue #3: Missing 132 Images

**Expected**: 388 images extracted
**Actual**: 256 images in database

**Possible Causes**:
1. ❌ Classification filtered out 132 images as non-material
2. ❌ Upload failures (network/storage errors)
3. ❌ Duplicate detection removed images
4. ❌ Page range filtering excluded images

**Investigation Needed**:
- Check classification logs for material vs non-material counts
- Check upload logs for failures
- Verify image extraction count

---

## 📋 COMPLETE MODEL INVENTORY

### AI Models Used

| Model | Provider | Purpose | API Endpoint | Dimensions | Status |
|-------|----------|---------|--------------|------------|--------|
| `claude-sonnet-4-20250514` | Anthropic | Product Discovery | `api.anthropic.com/v1/messages` | N/A | ✅ Working |
| `gpt-5` | OpenAI | Product Discovery (alt) | `api.openai.com/v1/chat/completions` | N/A | ✅ Working |
| `meta-llama/Llama-4-Scout-17B-16E-Instruct` | TogetherAI | Image Classification | `api.together.xyz/v1/chat/completions` | N/A | ✅ Working |
| `text-embedding-3-small` | OpenAI | Text Embeddings | `api.openai.com/v1/embeddings` | 1536D | ✅ Working |
| `google/siglip-so400m-patch14-384` | Google (local) | Visual Embeddings | sentence-transformers | 512D | ❌ BROKEN |
| `openai/clip-vit-base-patch32` | OpenAI (local) | Visual Embeddings (fallback) | transformers | 512D | ✅ Working |

### Library Versions

| Library | Version | Status |
|---------|---------|--------|
| `sentence-transformers` | 3.0.0 | ⚠️ Incompatible with SigLIP |
| `transformers` | 4.41.0 | ✅ Working |
| `torch` | Latest | ✅ Working |
| `httpx` | Latest | ✅ Working |

---

## ✅ RECOMMENDED FIXES

### Priority 1: Fix SigLIP Loading (CRITICAL)

**Option A: Use Transformers Directly** (RECOMMENDED)
```python
from transformers import AutoModel, AutoProcessor
import torch

# Load SigLIP using transformers (not sentence-transformers)
model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384")
processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")

# Generate embedding
inputs = processor(images=pil_image, return_tensors="pt")
with torch.no_grad():
    outputs = model.get_image_features(**inputs)
    embedding = outputs / outputs.norm(dim=-1, keepdim=True)
```

**Option B: Patch SiglipConfig**
```python
# Add hidden_size property to SiglipConfig
from transformers import SiglipConfig

original_getattribute = SiglipConfig.__getattribute__

def patched_getattribute(self, name):
    if name == "hidden_size":
        return self.vision_config.hidden_size
    return original_getattribute(self, name)

SiglipConfig.__getattribute__ = patched_getattribute
```

---

### Priority 2: Fix Missing Image Embeddings

**Investigation Steps**:
1. Add detailed logging to `save_images_and_generate_clips()`
2. Verify database insert success
3. Check for transaction rollbacks
4. Add error handling for embedding generation failures

---

### Priority 3: Investigate Missing Images

**Investigation Steps**:
1. Log classification results (material vs non-material counts)
2. Log upload success/failure counts
3. Verify image extraction count matches PDF page count
4. Check for duplicate detection logic

---

## 📊 EXPECTED VS ACTUAL RESULTS

| Metric | Expected | Actual | Status |
|--------|----------|--------|--------|
| Products Created | 11 | 11 | ✅ |
| Images Extracted | 388 | 388 | ✅ |
| Images Classified (Material) | ~260 | ? | ⚠️ |
| Images Uploaded | ~260 | ? | ⚠️ |
| Images in DB | ~260 | 256 | ⚠️ |
| Image Embeddings | 1,280 (256×5) | 0 | ❌ |
| Chunks Created | 107 | 107 | ✅ |
| Text Embeddings | 107 | 107 | ✅ |
| Chunk-Image Relationships | ~500 | 0 | ❌ |
| Product-Image Relationships | ~260 | 0 | ❌ |
| Chunk-Product Relationships | ~107 | 0 | ❌ |

---

## 🎯 NEXT STEPS

1. ✅ **Fix `_ensure_properties_exist` error** - COMPLETED
2. ⏳ **Fix SigLIP loading** - Use transformers directly
3. ⏳ **Debug missing image embeddings** - Add logging and verify DB inserts
4. ⏳ **Investigate missing 132 images** - Check classification and upload logs
5. ⏳ **Fix relationship creation** - Depends on image embeddings being saved
6. ⏳ **Re-run NOVA test** - Validate all fixes

---

**Document Version**: 1.0
**Last Updated**: 2025-11-23
**Author**: AI Assistant (Augment Agent)

