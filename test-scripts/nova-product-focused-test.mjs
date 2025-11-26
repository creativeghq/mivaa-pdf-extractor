/**
 * NOVA Product Focused End-to-End Test
 * Tests the complete PDF processing pipeline from Harmony PDF catalog
 */

const MIVAA_API = 'https://v1api.materialshub.gr';
const HARMONY_PDF_URL = 'https://bgbavxtjlbvgplozizxu.supabase.co/storage/v1/object/public/pdf-documents/harmony-signature-book-24-25.pdf';
const WORKSPACE_ID = 'ffafc28b-1b8b-4b0d-b226-9f9a6154004e';

const NOVA_PRODUCT = {
  name: 'NOVA',
  designer: 'SG NY',
  searchTerms: ['NOVA', 'SG NY', 'SGNY']
};

function log(category, message, level = 'info') {
  const emoji = {
    'step': '📋',
    'info': '📝',
    'success': '✅',
    'error': '❌',
    'warning': '⚠️',
    'data': '📊'
  }[level] || '📝';
  
  console.log(`${emoji} [${category}] ${message}`);
}

function logSection(title) {
  console.log('\n' + '='.repeat(100));
  console.log(`🎯 ${title}`);
  console.log('='.repeat(100));
}

async function cleanupOldTestData() {
  // DISABLED: Do not delete jobs - we need them to see test results!
  // Only cleanup should be done manually before running tests
  log('CLEANUP', 'Cleanup disabled - preserving job records for analysis', 'info');
  log('CLEANUP', 'Please manually cleanup old test data if needed before running tests', 'info');
  return;

  /* ORIGINAL CLEANUP CODE - DISABLED
  log('CLEANUP', 'Deleting all old test data from database...', 'step');

  try {
    const response = await fetch(`${MIVAA_API}/api/rag/documents/jobs?limit=100&sort=created_at:desc`);
    if (!response.ok) {
      log('CLEANUP', 'Failed to fetch jobs list', 'warning');
      return;
    }

    const data = await response.json();
    const jobs = data.jobs || [];

    const harmonyJobs = jobs.filter(job => {
      const filename = job.metadata?.filename || job.filename || '';
      return filename.includes('harmony-signature-book-24-25');
    });

    if (harmonyJobs.length === 0) {
      log('CLEANUP', 'No old Harmony PDF jobs found', 'info');
      return;
    }

    log('CLEANUP', `Found ${harmonyJobs.length} old Harmony PDF jobs to delete`, 'info');

    for (const job of harmonyJobs) {
      const jobId = job.id;
      const documentId = job.document_id;

      log('CLEANUP', `Deleting job ${jobId} and document ${documentId}...`, 'info');

      try {
        const deleteResponse = await fetch(`${MIVAA_API}/api/rag/documents/jobs/${jobId}`, {
          method: 'DELETE'
        });

        if (deleteResponse.ok) {
          log('CLEANUP', `✅ Deleted job ${jobId}`, 'success');
        } else {
          log('CLEANUP', `⚠️ Failed to delete job ${jobId}: ${deleteResponse.status}`, 'warning');
        }
      } catch (error) {
        log('CLEANUP', `❌ Error deleting job ${jobId}: ${error.message}`, 'error');
      }
    }

    log('CLEANUP', '✅ Cleanup complete!', 'success');
    await new Promise(resolve => setTimeout(resolve, 2000));

  } catch (error) {
    log('CLEANUP', `Error during cleanup: ${error.message}`, 'error');
  }
  */
}

async function uploadPDFForNovaExtraction() {
  log('UPLOAD', `Using URL-based upload: ${HARMONY_PDF_URL}`, 'info');

  const formData = new FormData();
  formData.append('file_url', HARMONY_PDF_URL);
  formData.append('title', 'NOVA Product Extraction - Focused Test');
  formData.append('description', 'Extract all products from Harmony catalog');
  formData.append('tags', 'nova,harmony,test');
  formData.append('categories', 'products');
  formData.append('processing_mode', 'deep');
  formData.append('discovery_model', 'claude');
  formData.append('chunk_size', '1024');
  formData.append('chunk_overlap', '128');
  formData.append('enable_prompt_enhancement', 'true');
  formData.append('workspace_id', WORKSPACE_ID);

  log('UPLOAD', `Triggering Consolidated Upload via MIVAA API: ${MIVAA_API}/api/rag/documents/upload`, 'info');
  log('UPLOAD', `Mode: deep | Categories: products | Discovery: claude | Async: enabled`, 'info');

  const uploadResponse = await fetch(`${MIVAA_API}/api/rag/documents/upload`, {
    method: 'POST',
    body: formData
  });

  if (!uploadResponse.ok) {
    const errorText = await uploadResponse.text();
    throw new Error(`Upload failed: ${uploadResponse.status} - ${errorText}`);
  }

  const result = await uploadResponse.json();

  log('UPLOAD', `✅ Job ID: ${result.job_id}`, 'success');
  log('UPLOAD', `✅ Document ID: ${result.document_id}`, 'success');
  log('UPLOAD', `✅ Status: ${result.status}`, 'success');

  if (result.message) {
    log('UPLOAD', result.message, 'info');
  }

  if (result.status_url) {
    log('UPLOAD', `📍 Status URL: ${result.status_url}`, 'info');
  }

  return {
    job_id: result.job_id,
    document_id: result.document_id
  };
}

async function validateDataSaved(documentId, jobData) {
  const validation = {
    chunks: 0,
    images: 0,
    products: 0,
    relevancies: 0,
    textEmbeddings: 0,
    imageEmbeddings: 0
  };

  try {
    const chunksResponse = await fetch(`${MIVAA_API}/api/rag/chunks?document_id=${documentId}&limit=1000`);
    if (chunksResponse.ok) {
      const chunksData = await chunksResponse.json();
      validation.chunks = chunksData.chunks?.length || 0;
      validation.textEmbeddings = chunksData.chunks?.filter(c => c.embedding).length || 0;
    }

    const imagesResponse = await fetch(`${MIVAA_API}/api/rag/images?document_id=${documentId}&limit=1000`);
    if (imagesResponse.ok) {
      const imagesData = await imagesResponse.json();
      validation.images = imagesData.images?.length || 0;
      validation.imageEmbeddings = imagesData.images?.filter(img => img.visual_clip_embedding_512).length || 0;
    }

    const productsResponse = await fetch(`${MIVAA_API}/api/rag/products?document_id=${documentId}&limit=1000`);
    if (productsResponse.ok) {
      const productsData = await productsResponse.json();
      validation.products = productsData.products?.length || 0;
    }

    const productImageRelResponse = await fetch(`${MIVAA_API}/api/rag/product-image-relationships?document_id=${documentId}&limit=1000`);
    if (productImageRelResponse.ok) {
      const relData = await productImageRelResponse.json();
      validation.relevancies = relData.count || 0;
    }

    const jobChunks = jobData.metadata?.chunks_created || 0;
    const jobImages = jobData.metadata?.images_extracted || 0;
    const jobProducts = jobData.metadata?.products_created || 0;

    const chunksMatch = validation.chunks === jobChunks;
    const imagesMatch = validation.images <= jobImages;
    const productsMatch = validation.products === jobProducts;

    log('VALIDATE', `Chunks: ${validation.chunks}/${jobChunks} ${chunksMatch ? '✅' : '❌'}`, chunksMatch ? 'success' : 'error');
    log('VALIDATE', `  - With Text Embeddings: ${validation.textEmbeddings}`, 'info');
    log('VALIDATE', `Images: ${validation.images}/${jobImages} ${imagesMatch ? '✅' : '❌'}`, imagesMatch ? 'success' : 'info');
    if (validation.images < jobImages) {
      log('VALIDATE', `  ℹ️  Filtered ${jobImages - validation.images} non-material images (expected behavior)`, 'info');
    }
    log('VALIDATE', `  - With Image Embeddings: ${validation.imageEmbeddings}`, 'info');
    log('VALIDATE', `Products: ${validation.products}/${jobProducts} ${productsMatch ? '✅' : '❌'}`, productsMatch ? 'success' : 'error');
    log('VALIDATE', `Relevancies: ${validation.relevancies}`, 'info');

    return {
      valid: chunksMatch && imagesMatch && productsMatch,
      validation,
      expected: { chunks: jobChunks, images: jobImages, products: jobProducts }
    };
  } catch (error) {
    log('VALIDATE', `Validation error: ${error.message}`, 'error');
    return { valid: false, error: error.message };
  }
}

async function monitorProcessingJob(jobId, documentId) {
  const maxAttempts = 480;
  const pollInterval = 15000;
  let lastProgress = 0;
  let lastValidation = null;

  log('MONITOR', `Starting job monitoring for: ${jobId}`, 'info');
  log('MONITOR', `Polling interval: ${pollInterval/1000}s | Max duration: ${(maxAttempts * pollInterval)/1000/60} minutes`, 'info');

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    await new Promise(resolve => setTimeout(resolve, pollInterval));

    const statusResponse = await fetch(`${MIVAA_API}/api/rag/documents/job/${jobId}`);

    if (!statusResponse.ok) {
      log('MONITOR', `API returned ${statusResponse.status}`, 'warning');
      continue;
    }

    const jobData = await statusResponse.json();
    const status = jobData.status;
    const progress = jobData.progress || 0;
    const metadata = jobData.metadata || {};
    const currentStep = metadata.current_step || metadata.stage || 'Processing';

    let progressMsg = `[${attempt}/${maxAttempts}] ${status.toUpperCase()} (${progress}%) - ${currentStep}`;

    if (metadata.chunks_created) progressMsg += ` | Chunks: ${metadata.chunks_created}`;
    if (metadata.images_extracted) progressMsg += ` | Images: ${metadata.images_extracted}`;
    if (metadata.products_created) progressMsg += ` | Products: ${metadata.products_created}`;
    if (metadata.current_page && metadata.total_pages) {
      progressMsg += ` | Page: ${metadata.current_page}/${metadata.total_pages}`;
    }

    log('MONITOR', progressMsg, 'info');

    if (progress >= 40 && progress !== lastProgress && progress % 20 === 0) {
      log('VALIDATE', `Running data validation at ${progress}%...`, 'info');
      lastValidation = await validateDataSaved(documentId, jobData);
      if (!lastValidation.valid) {
        log('VALIDATE', `⚠️ Data mismatch detected at ${progress}%`, 'warning');
      }
    }

    lastProgress = progress;

    if (status === 'completed') {
      log('MONITOR', '✅ Job completed successfully!', 'success');

      if (metadata.chunks_created || metadata.images_extracted || metadata.products_created) {
        log('MONITOR', '📊 Final Statistics:', 'success');
        log('MONITOR', `   📄 Chunks: ${metadata.chunks_created || 0}`, 'info');
        log('MONITOR', `   🖼️  Images: ${metadata.images_extracted || 0}`, 'info');
        log('MONITOR', `   📦 Products: ${metadata.products_created || 0}`, 'info');

        if (metadata.ai_usage) {
          log('MONITOR', '   🤖 AI Usage:', 'info');
          Object.entries(metadata.ai_usage).forEach(([model, count]) => {
            log('MONITOR', `      - ${model}: ${count}`, 'info');
          });
        }
      }

      log('VALIDATE', 'Running final data validation...', 'info');
      const finalValidation = await validateDataSaved(documentId, jobData);

      if (!finalValidation.valid) {
        log('VALIDATE', '❌ CRITICAL: Final validation failed! Data not properly saved!', 'error');
        throw new Error('Data validation failed: ' + JSON.stringify(finalValidation));
      }

      log('VALIDATE', '✅ All data successfully saved to database!', 'success');
      return jobData;
    }

    if (status === 'failed') {
      const error = jobData.error || 'Unknown error';
      log('MONITOR', `❌ Job failed: ${error}`, 'error');
      throw new Error(`Job failed: ${error}`);
    }

    if (status === 'interrupted') {
      log('MONITOR', '⚠️ Job interrupted! Attempting to resume...', 'warning');
      const resumeResponse = await fetch(`${MIVAA_API}/api/rag/documents/job/${jobId}/resume`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });

      if (resumeResponse.ok) {
        log('MONITOR', '✅ Job resumed successfully', 'success');
      } else {
        const resumeError = await resumeResponse.text();
        log('MONITOR', `❌ Resume failed: ${resumeError}`, 'error');
        throw new Error(`Job interrupted and resume failed: ${resumeError}`);
      }
    }
  }

  throw new Error('Job monitoring timed out after 2 hours');
}

async function retrieveNovaProductData(documentId) {
  log('RETRIEVE', `Fetching ALL product data for document: ${documentId}`, 'info');

  const allData = {
    chunks: [],
    images: [],
    products: [],
    chunkImageRelevancies: [],
    productImageRelevancies: [],
    chunkProductRelevancies: []
  };

  const chunksResponse = await fetch(`${MIVAA_API}/api/rag/chunks?document_id=${documentId}&limit=1000`);
  if (chunksResponse.ok) {
    const chunksData = await chunksResponse.json();
    allData.chunks = chunksData.chunks || [];
    log('RETRIEVE', `Found ${allData.chunks.length} total chunks`, 'success');

    const chunksWithEmbeddings = allData.chunks.filter(c => c.embedding).length;
    log('RETRIEVE', `  - ${chunksWithEmbeddings} chunks have text embeddings`, 'info');

    const chunksWithMetadata = allData.chunks.filter(c => c.metadata && Object.keys(c.metadata).length > 0).length;
    log('RETRIEVE', `  - ${chunksWithMetadata} chunks have metadata`, 'info');
  } else {
    const errorText = await chunksResponse.text();
    log('RETRIEVE', `Failed to fetch chunks: ${chunksResponse.status} ${chunksResponse.statusText}`, 'error');
    log('RETRIEVE', `Error details: ${errorText}`, 'error');
  }

  const imagesResponse = await fetch(`${MIVAA_API}/api/rag/images?document_id=${documentId}&limit=1000`);
  if (imagesResponse.ok) {
    const imagesData = await imagesResponse.json();
    allData.images = imagesData.images || [];
    log('RETRIEVE', `Found ${allData.images.length} total images`, 'success');

    const visualEmbeddings = allData.images.filter(img => img.visual_clip_embedding_512).length;
    const colorEmbeddings = allData.images.filter(img => img.color_clip_embedding_512).length;
    const textureEmbeddings = allData.images.filter(img => img.texture_clip_embedding_512).length;
    const applicationEmbeddings = allData.images.filter(img => img.application_clip_embedding_512).length;
    const materialEmbeddings = allData.images.filter(img => img.material_clip_embedding_512).length;

    const totalClipEmbeddings = visualEmbeddings + colorEmbeddings + textureEmbeddings + applicationEmbeddings + materialEmbeddings;

    log('RETRIEVE', `  - CLIP Embeddings Generated:`, 'info');
    log('RETRIEVE', `    • Visual: ${visualEmbeddings}`, 'info');
    log('RETRIEVE', `    • Color: ${colorEmbeddings}`, 'info');
    log('RETRIEVE', `    • Texture: ${textureEmbeddings}`, 'info');
    log('RETRIEVE', `    • Application: ${applicationEmbeddings}`, 'info');
    log('RETRIEVE', `    • Material: ${materialEmbeddings}`, 'info');
    log('RETRIEVE', `    • TOTAL CLIP Embeddings: ${totalClipEmbeddings}`, 'success');
  } else {
    log('RETRIEVE', `Failed to fetch images: ${imagesResponse.status} ${imagesResponse.statusText}`, 'error');
  }

  const productsResponse = await fetch(`${MIVAA_API}/api/rag/products?document_id=${documentId}&limit=1000`);
  if (productsResponse.ok) {
    const productsData = await productsResponse.json();
    allData.products = productsData.products || [];
    log('RETRIEVE', `Found ${allData.products.length} total products`, 'success');

    const productsWithMetadata = allData.products.filter(p => p.metadata && Object.keys(p.metadata).length > 0).length;
    log('RETRIEVE', `  - ${productsWithMetadata} products have metadata`, 'info');
  } else {
    log('RETRIEVE', `Failed to fetch products: ${productsResponse.status} ${productsResponse.statusText}`, 'error');
  }

  const chunkImageRelResponse = await fetch(`${MIVAA_API}/api/rag/relevancies?document_id=${documentId}&limit=1000`);
  if (chunkImageRelResponse.ok) {
    const relData = await chunkImageRelResponse.json();
    allData.chunkImageRelevancies = relData.relevancies || [];
    log('RETRIEVE', `Found ${allData.chunkImageRelevancies.length} chunk-image relevancies`, 'success');
  } else {
    log('RETRIEVE', `Failed to fetch chunk-image relevancies: ${chunkImageRelResponse.status}`, 'error');
  }

  const productImageRelResponse = await fetch(`${MIVAA_API}/api/rag/product-image-relationships?document_id=${documentId}&limit=1000`);
  if (productImageRelResponse.ok) {
    const relData = await productImageRelResponse.json();
    allData.productImageRelevancies = relData.relationships || [];
    log('RETRIEVE', `Found ${allData.productImageRelevancies.length} product-image relevancies`, 'success');
  } else {
    log('RETRIEVE', `Failed to fetch product-image relevancies: ${productImageRelResponse.status}`, 'warning');
  }

  const chunkProductRelResponse = await fetch(`${MIVAA_API}/api/rag/chunk-product-relationships?document_id=${documentId}&limit=1000`);
  if (chunkProductRelResponse.ok) {
    const relData = await chunkProductRelResponse.json();
    allData.chunkProductRelevancies = relData.relationships || [];
    log('RETRIEVE', `Found ${allData.chunkProductRelevancies.length} chunk-product relevancies`, 'success');
  } else {
    log('RETRIEVE', `Failed to fetch chunk-product relevancies: ${chunkProductRelResponse.status}`, 'warning');
  }

  return allData;
}

async function generateDetailedReport(allData, jobResult) {
  logSection('DETAILED PDF PROCESSING REPORT');

  const chunksWithEmbeddings = allData.chunks.filter(c => c.embedding).length;
  const chunksWithMetadata = allData.chunks.filter(c => c.metadata && Object.keys(c.metadata).length > 0).length;

  const visualEmbeddings = allData.images.filter(img => img.visual_clip_embedding_512).length;
  const colorEmbeddings = allData.images.filter(img => img.color_clip_embedding_512).length;
  const textureEmbeddings = allData.images.filter(img => img.texture_clip_embedding_512).length;
  const applicationEmbeddings = allData.images.filter(img => img.application_clip_embedding_512).length;
  const materialEmbeddings = allData.images.filter(img => img.material_clip_embedding_512).length;
  const totalClipEmbeddings = visualEmbeddings + colorEmbeddings + textureEmbeddings + applicationEmbeddings + materialEmbeddings;

  const productsWithMetadata = allData.products.filter(p => p.metadata && Object.keys(p.metadata).length > 0).length;

  const report = {
    timestamp: new Date().toISOString(),
    job: {
      id: jobResult.job_id,
      document_id: jobResult.document_id,
      status: jobResult.status,
      progress: jobResult.progress,
      metadata: jobResult.metadata
    },
    data: allData,
    summary: {
      total_chunks: allData.chunks.length,
      chunks_with_embeddings: chunksWithEmbeddings,
      chunks_with_metadata: chunksWithMetadata,
      total_images: allData.images.length,
      clip_embeddings: {
        visual: visualEmbeddings,
        color: colorEmbeddings,
        texture: textureEmbeddings,
        application: applicationEmbeddings,
        material: materialEmbeddings,
        total: totalClipEmbeddings
      },
      total_products: allData.products.length,
      products_with_metadata: productsWithMetadata,
      relevancies: {
        chunk_image: allData.chunkImageRelevancies.length,
        product_image: allData.productImageRelevancies.length,
        chunk_product: allData.chunkProductRelevancies.length,
        total: allData.chunkImageRelevancies.length + allData.productImageRelevancies.length + allData.chunkProductRelevancies.length
      }
    }
  };

  logSection('📊 FINAL SUMMARY - NOVA PRODUCT TEST RESULTS');

  console.log('\n' + '='.repeat(100));
  console.log('1️⃣  PRODUCTS');
  console.log('='.repeat(100));
  console.log(`   ✅ Total Products: ${report.summary.total_products}`);
  console.log(`   ✅ Products with Metadata: ${productsWithMetadata}`);

  console.log('\n' + '='.repeat(100));
  console.log('2️⃣  CLIP EMBEDDINGS GENERATED');
  console.log('='.repeat(100));
  console.log(`   ✅ Visual Embeddings: ${visualEmbeddings}`);
  console.log(`   ✅ Color Embeddings: ${colorEmbeddings}`);
  console.log(`   ✅ Texture Embeddings: ${textureEmbeddings}`);
  console.log(`   ✅ Application Embeddings: ${applicationEmbeddings}`);
  console.log(`   ✅ Material Embeddings: ${materialEmbeddings}`);
  console.log(`   ✅ TOTAL CLIP Embeddings: ${totalClipEmbeddings}`);

  console.log('\n' + '='.repeat(100));
  console.log('3️⃣  TOTAL IMAGES ADDED TO DB');
  console.log('='.repeat(100));
  console.log(`   ✅ Total Images: ${report.summary.total_images}`);

  console.log('\n' + '='.repeat(100));
  console.log('4️⃣  PRODUCT RELEVANCIES TO IMAGES');
  console.log('='.repeat(100));
  console.log(`   ✅ Total Products: ${report.summary.total_products}`);
  console.log(`   ✅ Product-Image Relevancies: ${report.summary.relevancies.product_image}`);
  console.log(`   📊 Example: ${report.summary.total_products} products → ${report.summary.relevancies.product_image} image relationships`);

  console.log('\n' + '='.repeat(100));
  console.log('5️⃣  TEXT EMBEDDINGS');
  console.log('='.repeat(100));
  console.log(`   ✅ Total Chunks: ${report.summary.total_chunks}`);
  console.log(`   ✅ Chunks with Text Embeddings: ${chunksWithEmbeddings}`);

  console.log('\n' + '='.repeat(100));
  console.log('6️⃣  META GENERATED AND EMBEDDINGS RELATED TO META');
  console.log('='.repeat(100));
  console.log(`   ✅ Chunks with Metadata: ${chunksWithMetadata}`);
  console.log(`   ✅ Products with Metadata: ${productsWithMetadata}`);
  console.log(`   ✅ Total Metadata Generated: ${chunksWithMetadata + productsWithMetadata}`);
  console.log(`   ✅ Metadata Embeddings (text embeddings include metadata): ${chunksWithEmbeddings}`);

  console.log('\n' + '='.repeat(100));
  console.log('7️⃣  ALL RELATIONSHIP COUNTS');
  console.log('='.repeat(100));
  console.log(`   📊 EMBEDDINGS TO PRODUCTS:`);
  console.log(`      • Total Text Embeddings (chunks): ${chunksWithEmbeddings}`);
  console.log(`      • Total CLIP Embeddings (images): ${totalClipEmbeddings}`);
  console.log(`      • Products: ${report.summary.total_products}`);
  console.log(`      • Chunk-Product Relationships: ${report.summary.relevancies.chunk_product}`);
  console.log(`      • Product-Image Relationships: ${report.summary.relevancies.product_image}`);
  console.log(``);
  console.log(`   📊 CHUNKS TO PRODUCTS:`);
  console.log(`      • Total Chunks: ${report.summary.total_chunks}`);
  console.log(`      • Total Products: ${report.summary.total_products}`);
  console.log(`      • Chunk-Product Relationships: ${report.summary.relevancies.chunk_product}`);
  console.log(``);
  console.log(`   📊 CHUNKS TO IMAGES:`);
  console.log(`      • Total Chunks: ${report.summary.total_chunks}`);
  console.log(`      • Total Images: ${report.summary.total_images}`);
  console.log(`      • Chunk-Image Relationships: ${report.summary.relevancies.chunk_image}`);

  console.log('\n' + '='.repeat(100));
  console.log('📊 ALL RELEVANCIES SUMMARY');
  console.log('='.repeat(100));
  console.log(`   ✅ Chunk-Image Relevancies: ${report.summary.relevancies.chunk_image}`);
  console.log(`   ✅ Product-Image Relevancies: ${report.summary.relevancies.product_image}`);
  console.log(`   ✅ Chunk-Product Relevancies: ${report.summary.relevancies.chunk_product}`);
  console.log(`   ✅ TOTAL RELEVANCIES: ${report.summary.relevancies.total}`);

  console.log('\n📝 SAMPLE CHUNKS (First 3):');
  allData.chunks.slice(0, 3).forEach((chunk, idx) => {
    console.log(`\nChunk ${idx + 1}:`);
    console.log(`  ID: ${chunk.id}`);
    console.log(`  Content: ${chunk.content?.substring(0, 150)}...`);
    console.log(`  Page: ${chunk.page_number || 'N/A'}`);
    if (chunk.metadata) {
      console.log(`  Metadata: ${JSON.stringify(chunk.metadata, null, 2)}`);
    }
  });

  console.log('\n🖼️  SAMPLE IMAGES (First 3):');
  allData.images.slice(0, 3).forEach((img, idx) => {
    console.log(`\nImage ${idx + 1}:`);
    console.log(`  ID: ${img.id}`);
    console.log(`  URL: ${img.url || img.storage_path}`);
    console.log(`  Page: ${img.page_number || 'N/A'}`);
    if (img.caption || img.description) {
      console.log(`  Caption: ${img.caption || img.description}`);
    }
    if (img.metadata) {
      console.log(`  Metadata: ${JSON.stringify(img.metadata, null, 2)}`);
    }
  });

  console.log('\n🏷️  ALL PRODUCTS:');
  allData.products.forEach((product, idx) => {
    console.log(`\nProduct ${idx + 1}:`);
    console.log(`  ID: ${product.id}`);
    console.log(`  Name: ${product.name}`);
    console.log(`  Designer: ${product.designer || 'N/A'}`);
    console.log(`  Description: ${product.description?.substring(0, 200) || 'N/A'}...`);
    if (product.metadata) {
      console.log(`  Metadata: ${JSON.stringify(product.metadata, null, 2)}`);
    }
    if (product.page_ranges) {
      console.log(`  Page Ranges: ${JSON.stringify(product.page_ranges)}`);
    }
  });

  console.log('\n🔗 SAMPLE CHUNK-IMAGE RELEVANCIES (First 10):');
  allData.chunkImageRelevancies.slice(0, 10).forEach((rel, idx) => {
    console.log(`\n${idx + 1}. Chunk ${rel.chunk_id} ↔ Image ${rel.image_id}`);
    console.log(`   Relevance Score: ${rel.relevance_score}`);
    console.log(`   Relationship Type: ${rel.relationship_type || 'N/A'}`);
  });

  console.log('\n🔗 SAMPLE PRODUCT-IMAGE RELEVANCIES (First 10):');
  allData.productImageRelevancies.slice(0, 10).forEach((rel, idx) => {
    console.log(`\n${idx + 1}. Product ${rel.product_id} ↔ Image ${rel.image_id}`);
    console.log(`   Relevance Score: ${rel.relevance_score || 'N/A'}`);
    console.log(`   Relationship Type: ${rel.relationship_type || 'N/A'}`);
  });

  if (jobResult.metadata?.ai_usage) {
    console.log('\n🤖 AI MODEL USAGE:');
    Object.entries(jobResult.metadata.ai_usage).forEach(([model, count]) => {
      console.log(`  ${model}: ${count} calls`);
    });
  }

  const reportPath = `/var/www/mivaa-pdf-extractor/test-scripts/pdf-processing-report-${Date.now()}.json`;
  const fs = await import('fs');
  fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
  log('REPORT', `Detailed report saved to: ${reportPath}`, 'success');
}

async function runNovaProductTest() {
  logSection('NOVA PRODUCT FOCUSED END-TO-END TEST');

  console.log(`Product: ${NOVA_PRODUCT.name} by ${NOVA_PRODUCT.designer}`);
  console.log(`PDF: ${HARMONY_PDF_URL}`);
  console.log(`Workspace: ${WORKSPACE_ID}`);
  console.log(`MIVAA API: ${MIVAA_API}\n`);

  try {
    await cleanupOldTestData();

    log('UPLOAD', 'Starting new PDF processing', 'step');
    const uploadResult = await uploadPDFForNovaExtraction();
    const jobId = uploadResult.job_id;
    const documentId = uploadResult.document_id;

    log('UPLOAD', `Job ID: ${jobId}`, 'info');
    log('UPLOAD', `Document ID: ${documentId}`, 'info');

    log('MONITOR', `Monitoring job: ${jobId}`, 'step');
    await monitorProcessingJob(jobId, documentId);

    log('VALIDATE', 'Retrieving ALL product data from document', 'step');
    const allData = await retrieveNovaProductData(documentId);

    log('REPORT', 'Generating detailed report', 'step');
    await generateDetailedReport(allData, { job_id: jobId, document_id: documentId });

    log('COMPLETE', '✅ PDF processing test completed successfully!', 'success');

  } catch (error) {
    log('ERROR', `Test failed: ${error.message}`, 'error');
    console.error(error);
    process.exit(1);
  }
}

runNovaProductTest().catch(error => {
  console.error('Fatal error:', error);
  process.exit(1);
});
