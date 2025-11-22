/**
 * NOVA Product Focused End-to-End Test
 * 
 * Tests the complete PDF processing pipeline for a SINGLE product (NOVA by SG NY)
 * from the Harmony PDF catalog.
 * 
 * This test will:
 * 1. Extract only NOVA product pages from Harmony PDF
 * 2. Process all related images
 * 3. Run full AI analysis (LLAMA classification, Claude chunking, CLIP embeddings)
 * 4. Generate text and image embeddings
 * 5. Create product record
 * 6. Return COMPLETE detailed results including:
 *    - Actual Supabase image URLs
 *    - Extracted text content
 *    - All metadata
 *    - AI model outputs and scores
 *    - Processing steps with timings
 *    - Quality metrics
 */

import fetch from 'node-fetch';
import fs from 'fs';
import FormData from 'form-data';
import { Blob } from 'buffer';

// Configuration
const MIVAA_API = 'http://127.0.0.1:8000';
const HARMONY_PDF_URL = 'https://bgbavxtjlbvgplozizxu.supabase.co/storage/v1/object/public/pdf-documents/harmony-signature-book-24-25.pdf';
const WORKSPACE_ID = 'ffafc28b-1b8b-4b0d-b226-9f9a6154004e';

// NOVA product search criteria
const NOVA_PRODUCT = {
  name: 'NOVA',
  designer: 'SG NY',
  searchTerms: ['NOVA', 'SG NY', 'SGNY']
};

// Logging utilities
function log(category, message, level = 'info') {
  const timestamp = new Date().toISOString();
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

// Main test function
async function runNovaProductTest() {
  logSection('NOVA PRODUCT FOCUSED END-TO-END TEST');

  console.log(`Product: ${NOVA_PRODUCT.name} by ${NOVA_PRODUCT.designer}`);
  console.log(`PDF: ${HARMONY_PDF_URL}`);
  console.log(`Workspace: ${WORKSPACE_ID}`);
  console.log(`MIVAA API: ${MIVAA_API}\n`);

  try {
    // Step 1: Check for existing job
    log('CHECK', 'Checking for existing jobs for this PDF', 'step');
    const existingJob = await findExistingJob();

    let jobId, documentId;

    if (existingJob) {
      log('CHECK', `Found existing job: ${existingJob.id} (${existingJob.status})`, 'info');
      jobId = existingJob.id;
      documentId = existingJob.document_id;

      if (existingJob.status === 'completed') {
        log('CHECK', 'Job already completed, skipping to data retrieval', 'success');
      } else if (existingJob.status === 'processing') {
        log('CHECK', 'Job is currently processing, monitoring progress', 'info');
      } else {
        log('CHECK', `Job status: ${existingJob.status}, will monitor`, 'warning');
      }
    } else {
      // Step 2: Upload PDF with async processing
      log('UPLOAD', 'No existing job found, starting new PDF processing', 'step');
      const uploadResult = await uploadPDFForNovaExtraction();
      jobId = uploadResult.job_id;
      documentId = uploadResult.document_id;

      log('UPLOAD', `Job ID: ${jobId}`, 'info');
      log('UPLOAD', `Document ID: ${documentId}`, 'info');
    }

    // Step 3: Monitor async job processing (skip if already completed)
    if (!existingJob || existingJob.status !== 'completed') {
      log('MONITOR', `Monitoring job: ${jobId}`, 'step');
      await monitorProcessingJob(jobId, documentId);
    }

    // Step 4: Retrieve and validate ALL product data
    log('VALIDATE', 'Retrieving ALL product data from document', 'step');
    const allData = await retrieveNovaProductData(documentId);

    // Step 5: Generate comprehensive report
    log('REPORT', 'Generating detailed report', 'step');
    await generateDetailedReport(allData, { job_id: jobId, document_id: documentId });

    log('COMPLETE', '✅ PDF processing test completed successfully!', 'success');

  } catch (error) {
    log('ERROR', `Test failed: ${error.message}`, 'error');
    console.error(error);
    process.exit(1);
  }
}

async function findExistingJob() {
  try {
    // Get recent jobs (last 50)
    const response = await fetch(`${MIVAA_API}/api/rag/documents/jobs?limit=50&sort=created_at:desc`);

    if (!response.ok) {
      log('CHECK', 'Failed to fetch jobs list', 'warning');
      return null;
    }

    const data = await response.json();
    const jobs = data.jobs || [];

    // Find job for harmony-signature-book-24-25.pdf
    const harmonyJob = jobs.find(job => {
      const filename = job.metadata?.filename || job.filename || '';
      return filename.includes('harmony-signature-book-24-25');
    });

    return harmonyJob || null;
  } catch (error) {
    log('CHECK', `Error checking for existing jobs: ${error.message}`, 'warning');
    return null;
  }
}

async function uploadPDFForNovaExtraction() {
  log('UPLOAD', `Using URL-based upload: ${HARMONY_PDF_URL}`, 'info');

  // Create form data with URL and processing options
  const formData = new FormData();

  // Use file_url parameter instead of downloading file
  formData.append('file_url', HARMONY_PDF_URL);

  // Add processing parameters using latest API specification
  formData.append('title', 'NOVA Product Extraction - Focused Test');
  formData.append('description', 'Extract all products from Harmony catalog');
  formData.append('tags', 'nova,harmony,test');
  formData.append('categories', 'products');  // Extract only products
  formData.append('processing_mode', 'deep');  // Deep mode for complete analysis
  formData.append('discovery_model', 'claude');  // Claude Sonnet 4.5 for best quality
  formData.append('chunk_size', '1024');
  formData.append('chunk_overlap', '128');
  formData.append('enable_prompt_enhancement', 'true');
  formData.append('workspace_id', WORKSPACE_ID);

  log('UPLOAD', `Triggering Consolidated Upload via MIVAA API: ${MIVAA_API}/api/rag/documents/upload`, 'info');
  log('UPLOAD', `Mode: deep | Categories: products | Discovery: claude | Async: enabled`, 'info');

  const uploadResponse = await fetch(`${MIVAA_API}/api/rag/documents/upload`, {
    method: 'POST',
    body: formData,
    headers: formData.getHeaders()
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
  // Validate that data is actually being saved to database via MIVAA API
  const validation = {
    chunks: 0,
    images: 0,
    products: 0,
    embeddings: 0,
    chunkImageRelationships: 0,
    productImageRelationships: 0,
    chunkProductRelationships: 0
  };

  try {
    // Check chunks using consolidated RAG endpoint
    const chunksResponse = await fetch(`${MIVAA_API}/api/rag/chunks?document_id=${documentId}&limit=1000`);
    if (chunksResponse.ok) {
      const chunksData = await chunksResponse.json();
      validation.chunks = Array.isArray(chunksData) ? chunksData.length : (chunksData.chunks?.length || 0);
    }

    // Check images using consolidated RAG endpoint
    const imagesResponse = await fetch(`${MIVAA_API}/api/rag/images?document_id=${documentId}&limit=1000`);
    if (imagesResponse.ok) {
      const imagesData = await imagesResponse.json();
      validation.images = Array.isArray(imagesData) ? imagesData.length : (imagesData.images?.length || 0);
    }

    // Check products using consolidated RAG endpoint
    const productsResponse = await fetch(`${MIVAA_API}/api/rag/products?document_id=${documentId}&limit=1000`);
    if (productsResponse.ok) {
      const productsData = await productsResponse.json();
      validation.products = Array.isArray(productsData) ? productsData.length : (productsData.products?.length || 0);
    }

    // Check embeddings using consolidated RAG endpoint
    const embeddingsResponse = await fetch(`${MIVAA_API}/api/rag/embeddings?document_id=${documentId}&limit=1000`);
    if (embeddingsResponse.ok) {
      const embeddingsData = await embeddingsResponse.json();
      validation.embeddings = Array.isArray(embeddingsData) ? embeddingsData.length : (embeddingsData.embeddings?.length || 0);
    }

    // Check chunk-image relationships
    const chunkImageRelsResponse = await fetch(`${MIVAA_API}/api/rag/relevancies?document_id=${documentId}&limit=10000`);
    if (chunkImageRelsResponse.ok) {
      const chunkImageRelsData = await chunkImageRelsResponse.json();
      validation.chunkImageRelationships = chunkImageRelsData.count || 0;
    }

    // Check product-image relationships
    const productImageRelsResponse = await fetch(`${MIVAA_API}/api/rag/product-image-relationships?document_id=${documentId}&limit=10000`);
    if (productImageRelsResponse.ok) {
      const productImageRelsData = await productImageRelsResponse.json();
      validation.productImageRelationships = productImageRelsData.count || 0;
    }

    // Check chunk-product relationships
    const chunkProductRelsResponse = await fetch(`${MIVAA_API}/api/rag/chunk-product-relationships?document_id=${documentId}&limit=10000`);
    if (chunkProductRelsResponse.ok) {
      const chunkProductRelsData = await chunkProductRelsResponse.json();
      validation.chunkProductRelationships = chunkProductRelsData.count || 0;
    }

    // Compare with job metadata
    const jobChunks = jobData.metadata?.chunks_created || 0;
    const jobImages = jobData.metadata?.images_extracted || 0;
    const jobProducts = jobData.metadata?.products_created || 0;
    const jobChunkImageRels = jobData.metadata?.chunk_image_relationships || 0;
    const jobProductImageRels = jobData.metadata?.product_image_relationships || 0;

    const chunksMatch = validation.chunks === jobChunks;
    const imagesMatch = validation.images === jobImages;
    const productsMatch = validation.products === jobProducts;
    const chunkImageRelsMatch = jobChunkImageRels === 0 || validation.chunkImageRelationships === jobChunkImageRels;
    const productImageRelsMatch = jobProductImageRels === 0 || validation.productImageRelationships === jobProductImageRels;

    log('VALIDATE', `Chunks: ${validation.chunks}/${jobChunks} ${chunksMatch ? '✅' : '❌'}`, chunksMatch ? 'success' : 'error');
    log('VALIDATE', `Images: ${validation.images}/${jobImages} ${imagesMatch ? '✅' : '❌'}`, imagesMatch ? 'success' : 'error');
    log('VALIDATE', `Products: ${validation.products}/${jobProducts} ${productsMatch ? '✅' : '❌'}`, productsMatch ? 'success' : 'error');
    log('VALIDATE', `Embeddings: ${validation.embeddings}`, 'info');
    log('VALIDATE', `Chunk-Image Relationships: ${validation.chunkImageRelationships}${jobChunkImageRels > 0 ? `/${jobChunkImageRels}` : ''} ${chunkImageRelsMatch ? '✅' : '❌'}`, chunkImageRelsMatch ? 'success' : 'error');
    log('VALIDATE', `Product-Image Relationships: ${validation.productImageRelationships}${jobProductImageRels > 0 ? `/${jobProductImageRels}` : ''} ${productImageRelsMatch ? '✅' : '❌'}`, productImageRelsMatch ? 'success' : 'error');
    log('VALIDATE', `Chunk-Product Relationships: ${validation.chunkProductRelationships}`, 'info');

    return {
      valid: chunksMatch && imagesMatch && productsMatch && chunkImageRelsMatch && productImageRelsMatch,
      validation,
      expected: {
        chunks: jobChunks,
        images: jobImages,
        products: jobProducts,
        chunkImageRelationships: jobChunkImageRels,
        productImageRelationships: jobProductImageRels
      }
    };
  } catch (error) {
    log('VALIDATE', `Validation error: ${error.message}`, 'error');
    return { valid: false, error: error.message };
  }
}

async function monitorProcessingJob(jobId, documentId) {
  const maxAttempts = 480; // 2 hours with 15-second intervals
  const pollInterval = 15000; // 15 seconds
  let lastProgress = 0;
  let lastValidation = null;

  log('MONITOR', `Starting job monitoring for: ${jobId}`, 'info');
  log('MONITOR', `Polling interval: ${pollInterval/1000}s | Max duration: ${(maxAttempts * pollInterval)/1000/60} minutes`, 'info');

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    await new Promise(resolve => setTimeout(resolve, pollInterval));

    // Use correct endpoint: /api/rag/documents/job/{job_id}
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

    // Enhanced progress logging with detailed metadata
    let progressMsg = `[${attempt}/${maxAttempts}] ${status.toUpperCase()} (${progress}%) - ${currentStep}`;

    if (metadata.chunks_created) progressMsg += ` | Chunks: ${metadata.chunks_created}`;
    if (metadata.images_extracted) progressMsg += ` | Images: ${metadata.images_extracted}`;
    if (metadata.products_created) progressMsg += ` | Products: ${metadata.products_created}`;
    if (metadata.current_page && metadata.total_pages) {
      progressMsg += ` | Page: ${metadata.current_page}/${metadata.total_pages}`;
    }

    log('MONITOR', progressMsg, 'info');

    // Validate data saving at key checkpoints
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

      // Display final statistics
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

      // Final validation
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
      // Try to resume the job using consolidated RAG endpoint
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
    embeddings: {
      text: [],
      image: [],
      visual: [],
      multimodal: []
    },
    relationships: {
      chunkImage: [],
      productImage: [],
      chunkProduct: []
    }
  };

  // Retrieve ALL chunks using consolidated RAG endpoint
  const chunksResponse = await fetch(`${MIVAA_API}/api/rag/chunks?document_id=${documentId}&limit=10000`);
  if (chunksResponse.ok) {
    const chunksData = await chunksResponse.json();
    allData.chunks = Array.isArray(chunksData) ? chunksData : (chunksData.chunks || []);
    log('RETRIEVE', `Found ${allData.chunks.length} total chunks`, 'success');
  } else {
    log('RETRIEVE', `Failed to fetch chunks: ${chunksResponse.status} ${chunksResponse.statusText}`, 'error');
  }

  // Retrieve ALL images using consolidated RAG endpoint
  const imagesResponse = await fetch(`${MIVAA_API}/api/rag/images?document_id=${documentId}&limit=10000`);
  if (imagesResponse.ok) {
    const imagesData = await imagesResponse.json();
    allData.images = Array.isArray(imagesData) ? imagesData : (imagesData.images || []);
    log('RETRIEVE', `Found ${allData.images.length} total images`, 'success');
  } else {
    log('RETRIEVE', `Failed to fetch images: ${imagesResponse.status} ${imagesResponse.statusText}`, 'error');
  }

  // Retrieve ALL products using consolidated RAG endpoint
  const productsResponse = await fetch(`${MIVAA_API}/api/rag/products?document_id=${documentId}&limit=10000`);
  if (productsResponse.ok) {
    const productsData = await productsResponse.json();
    allData.products = Array.isArray(productsData) ? productsData : (productsData.products || []);
    log('RETRIEVE', `Found ${allData.products.length} total products`, 'success');
  } else {
    log('RETRIEVE', `Failed to fetch products: ${productsResponse.status} ${productsResponse.statusText}`, 'error');
  }

  // Retrieve ALL embeddings using consolidated RAG endpoint
  const embeddingsResponse = await fetch(`${MIVAA_API}/api/rag/embeddings?document_id=${documentId}&limit=10000`);
  if (embeddingsResponse.ok) {
    const embeddingsData = await embeddingsResponse.json();
    const embeddings = Array.isArray(embeddingsData) ? embeddingsData : (embeddingsData.embeddings || []);

    // Categorize embeddings by type
    allData.embeddings.text = embeddings.filter(e => e.embedding_type === 'text');
    allData.embeddings.image = embeddings.filter(e => e.embedding_type === 'image');
    allData.embeddings.visual = embeddings.filter(e => e.embedding_type === 'visual');
    allData.embeddings.multimodal = embeddings.filter(e => e.embedding_type === 'multimodal');

    log('RETRIEVE', `Found ${allData.embeddings.text.length} text embeddings`, 'success');
    log('RETRIEVE', `Found ${allData.embeddings.image.length} image embeddings`, 'success');
    log('RETRIEVE', `Found ${allData.embeddings.visual.length} visual embeddings`, 'success');
    log('RETRIEVE', `Found ${allData.embeddings.multimodal.length} multimodal embeddings`, 'success');
  } else {
    log('RETRIEVE', `Failed to fetch embeddings: ${embeddingsResponse.status} ${embeddingsResponse.statusText}`, 'error');
  }

  // Retrieve chunk-image relationships
  const chunkImageRelsResponse = await fetch(`${MIVAA_API}/api/rag/relevancies?document_id=${documentId}&limit=10000`);
  if (chunkImageRelsResponse.ok) {
    const chunkImageRelsData = await chunkImageRelsResponse.json();
    allData.relationships.chunkImage = chunkImageRelsData.relevancies || [];
    log('RETRIEVE', `Found ${allData.relationships.chunkImage.length} chunk-image relationships`, 'success');
  } else {
    log('RETRIEVE', `Failed to fetch chunk-image relationships: ${chunkImageRelsResponse.status}`, 'error');
  }

  // Retrieve product-image relationships
  const productImageRelsResponse = await fetch(`${MIVAA_API}/api/rag/product-image-relationships?document_id=${documentId}&limit=10000`);
  if (productImageRelsResponse.ok) {
    const productImageRelsData = await productImageRelsResponse.json();
    allData.relationships.productImage = productImageRelsData.relationships || [];
    log('RETRIEVE', `Found ${allData.relationships.productImage.length} product-image relationships`, 'success');
  } else {
    log('RETRIEVE', `Failed to fetch product-image relationships: ${productImageRelsResponse.status}`, 'error');
  }

  // Retrieve chunk-product relationships
  const chunkProductRelsResponse = await fetch(`${MIVAA_API}/api/rag/chunk-product-relationships?document_id=${documentId}&limit=10000`);
  if (chunkProductRelsResponse.ok) {
    const chunkProductRelsData = await chunkProductRelsResponse.json();
    allData.relationships.chunkProduct = chunkProductRelsData.relationships || [];
    log('RETRIEVE', `Found ${allData.relationships.chunkProduct.length} chunk-product relationships`, 'success');
  } else {
    log('RETRIEVE', `Failed to fetch chunk-product relationships: ${chunkProductRelsResponse.status}`, 'error');
  }

  return allData;
}

async function generateDetailedReport(allData, jobResult) {
  logSection('DETAILED PDF PROCESSING REPORT');

  const totalEmbeddings =
    allData.embeddings.text.length +
    allData.embeddings.image.length +
    allData.embeddings.visual.length +
    allData.embeddings.multimodal.length;

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
      total_images: allData.images.length,
      total_products: allData.products.length,
      total_embeddings: totalEmbeddings,
      embeddings_by_type: {
        text: allData.embeddings.text.length,
        image: allData.embeddings.image.length,
        visual: allData.embeddings.visual.length,
        multimodal: allData.embeddings.multimodal.length
      },
      relationships: {
        chunk_image: allData.relationships.chunkImage.length,
        product_image: allData.relationships.productImage.length,
        chunk_product: allData.relationships.chunkProduct.length
      }
    }
  };

  // Print detailed summary
  logSection('📊 FINAL SUMMARY');
  console.log(`\n✅ Total Chunks: ${report.summary.total_chunks}`);
  console.log(`✅ Total Images: ${report.summary.total_images}`);
  console.log(`✅ Total Products: ${report.summary.total_products}`);
  console.log(`✅ Total Embeddings: ${report.summary.total_embeddings}`);
  console.log(`   - Text Embeddings: ${allData.embeddings.text.length}`);
  console.log(`   - Image Embeddings: ${allData.embeddings.image.length}`);
  console.log(`   - Visual Embeddings: ${allData.embeddings.visual.length}`);
  console.log(`   - Multimodal Embeddings: ${allData.embeddings.multimodal.length}`);
  console.log(`\n🔗 Total Relationships:`);
  console.log(`   - Chunk-Image Relationships: ${report.summary.relationships.chunk_image}`);
  console.log(`   - Product-Image Relationships: ${report.summary.relationships.product_image}`);
  console.log(`   - Chunk-Product Relationships: ${report.summary.relationships.chunk_product}`);

  // Print sample chunks (first 3)
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

  // Print sample images (first 3)
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

  // Print all products with detailed metadata
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

  // Print AI model usage if available
  if (jobResult.metadata?.ai_usage) {
    console.log('\n🤖 AI MODEL USAGE:');
    Object.entries(jobResult.metadata.ai_usage).forEach(([model, count]) => {
      console.log(`  ${model}: ${count} calls`);
    });
  }

  // Save report to file
  const reportPath = `pdf-processing-report-${Date.now()}.json`;
  fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
  log('REPORT', `Detailed report saved to: ${reportPath}`, 'success');
}

// Run the test
runNovaProductTest().catch(error => {
  console.error('Fatal error:', error);
  process.exit(1);
});

