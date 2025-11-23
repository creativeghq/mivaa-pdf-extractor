-- Migration: Drop redundant embedding columns from document_images table
-- Date: 2025-11-23
-- Reason: Embeddings now stored in embeddings table + VECS collections only
--
-- Architecture Change:
-- BEFORE: Embeddings stored in 3 places (document_images + embeddings + VECS)
-- AFTER: Embeddings stored in 2 places (embeddings + VECS)
--
-- Benefits:
-- - Eliminates data duplication
-- - Reduces storage waste
-- - Simplifies queries (single source of truth)
-- - Maintains fast similarity search (VECS) + tracking/analytics (embeddings table)

-- Drop the 5 redundant embedding columns from document_images table
ALTER TABLE document_images 
DROP COLUMN IF EXISTS visual_clip_embedding_512,
DROP COLUMN IF EXISTS color_clip_embedding_512,
DROP COLUMN IF EXISTS texture_clip_embedding_512,
DROP COLUMN IF EXISTS application_clip_embedding_512,
DROP COLUMN IF EXISTS material_clip_embedding_512;

-- Verify columns are dropped
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'document_images' 
AND column_name LIKE '%embedding%';

-- Expected result: No rows (all embedding columns removed)

-- Note: Embeddings are still accessible via:
-- 1. embeddings table: SELECT * FROM embeddings WHERE entity_type = 'image' AND entity_id = '<image_id>';
-- 2. VECS collections: Use VecsService.search_similar_images() for similarity search

