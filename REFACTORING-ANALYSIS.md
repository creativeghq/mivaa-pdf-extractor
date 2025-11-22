# MIVAA Codebase Refactoring Analysis

**Date:** 2025-11-22  
**Status:** Analysis Complete - Ready for Cleanup

---

## 📋 TODO COMMENTS ANALYSIS

### ✅ KEEP (Functionality Not Yet Built)

1. **Metadata Scope Tracking** - `app/api/metadata.py` (Lines 319, 321, 406, 407)
   - **TODO:** Track scope (catalog_general vs product_specific) and source_chunk_id
   - **Status:** NOT IMPLEMENTED - needs to be built
   - **Action:** Keep TODOs, implement in future

2. **Full Metadata Extraction** - `app/core/extractor.py` (Line 464)
   - **TODO:** Implement full metadata extraction logic with pattern matching
   - **Status:** ✅ ALREADY IMPLEMENTED in `/extractor.py` (root level, lines 75-614)
   - **Action:** REMOVE TODO - full implementation exists with comprehensive pattern matching for all 9 categories
   - **Note:** The stub in `app/core/extractor.py` should either call the real implementation or be removed

3. **AI Model Testing** - `app/services/admin_prompt_service.py` (Line 200)
   - **TODO:** Implement actual AI model testing (currently returns mock results)
   - **Status:** NOT IMPLEMENTED - mock implementation
   - **Action:** Keep TODO, implement in future

4. **Semantic Embeddings for Duplicates** - `app/services/duplicate_detection_service.py` (Line 361)
   - **TODO:** Use semantic embeddings for better accuracy (currently uses simple text matching)
   - **Status:** NOT IMPLEMENTED - using SequenceMatcher
   - **Action:** Keep TODO, implement in future

5. **Metadata Filters Conversion** - `app/services/llamaindex_service.py` (Line 279)
   - **TODO:** Convert filters format if needed
   - **Status:** NOT IMPLEMENTED - currently passes None
   - **Action:** Keep TODO, implement in future

6. **Spatial Proximity Filtering** - `app/services/product_creation_service.py` (Line 765)
   - **TODO:** Implement spatial proximity filtering for chunks
   - **Status:** NOT IMPLEMENTED - returns first chunk from same page
   - **Action:** Keep TODO, implement in future

7. **Page-Specific Text Extraction** - `app/services/product_discovery_service.py` (Line 1402)
   - **TODO:** Implement page-specific text extraction (currently returns first 10k chars)
   - **Status:** NOT IMPLEMENTED - needs page boundary storage
   - **Action:** Keep TODO, implement in future

8. **Multiple Products Per Page** - `app/services/product_vision_extractor.py` (Line 227)
   - **TODO:** Handle multiple products per page (currently returns first product)
   - **Status:** NOT IMPLEMENTED - returns products_data[0]
   - **Action:** Keep TODO, implement in future

9. **Claude/GPT Semantic Expansion** - `app/services/search_suggestions_service.py` (Line 499)
   - **TODO:** Integrate with Claude/GPT for semantic expansion
   - **Status:** NOT IMPLEMENTED - placeholder pass statement
   - **Action:** Keep TODO, implement in future

### ❌ REMOVE (User Decision)

10. **Redis Blacklist Checking** - `app/middleware/jwt_auth.py` (Line 528)
    - **TODO:** Implement Redis blacklist checking if Redis is configured
    - **Status:** NOT IMPLEMENTED
    - **Action:** REMOVE - user decided not to implement Redis

### ⚠️ INVESTIGATE (Potentially Already Implemented)

11. **CLIP Integration** - `app/services/llamaindex_service.py` (Line 5831)
    - **TODO:** Integrate with actual CLIP model
    - **Status:** NEEDS INVESTIGATION - we already have CLIP embeddings working
    - **Action:** Check if this is outdated TODO from before CLIP was implemented

---

## 🗄️ DATABASE TABLES ANALYSIS

### Tables Comparison

| Table Name | Rows | Used in Code | Purpose | Decision |
|------------|------|--------------|---------|----------|
| **processing_jobs** | 0 | ❌ NO | Unknown - no code references | ❌ DELETE |
| **background_jobs** | 1 | ❌ NO | Unknown - no code references | ❌ DELETE |
| **processing_queue** | 0 | ❌ NO | Unknown - no code references | ❌ DELETE |
| **batch_jobs** | 0 | ❌ NO | Unknown - no code references | ❌ DELETE |
| **pdf_processing_results** | 0 | ❌ NO | Old PDF processing results | ❌ DELETE |
| **processed_documents** | 12 | ❌ NO | Has data but no code uses it | ⚠️ INVESTIGATE |
| **document_processing_status** | 0 | ❌ NO | Old processing status tracking | ❌ DELETE |

### ✅ RECOMMENDATION: DELETE ALL UNUSED TABLES

**Rationale:**
- None of these tables are referenced in the Python codebase
- Most tables are empty (0 rows)
- `processed_documents` has 12 rows but no code uses it
- These appear to be legacy tables from old implementations

**Safe to Delete:**
1. `processing_jobs` - 0 rows, no code
2. `background_jobs` - 1 row, no code
3. `processing_queue` - 0 rows, no code
4. `batch_jobs` - 0 rows, no code
5. `pdf_processing_results` - 0 rows, no code
6. `document_processing_status` - 0 rows, no code

**Needs Investigation:**
- `processed_documents` - 12 rows, but no code references it. Check if this is old test data.

---

## 🗑️ FILES TO DELETE

1. **`admin_modules_old/` directory** - Empty directory with only `__pycache__` files

---

## ⚠️ CODE TO REMOVE

1. **Deprecated Search Strategy** - `app/api/rag_routes.py`
   - Lines 4406, 4668-4673
   - Remove `strategy='all'` code path
   - Force users to use `strategy='multi_vector'`

2. **Noise Comment** - `app/api/search.py`
   - Line 1644: `# REMOVED DEPRECATED ENDPOINTS`
   - Just remove this comment

---

## 📊 SUMMARY

- **TODOs to Keep:** 9 (functionality not yet built)
- **TODOs to Remove:** 1 (Redis - user decision)
- **TODOs to Investigate:** 1 (CLIP integration - may be outdated)
- **Database Tables to Delete:** 6-7 tables
- **Directories to Delete:** 1 (`admin_modules_old/`)
- **Code Paths to Remove:** 1 (deprecated `strategy='all'`)

