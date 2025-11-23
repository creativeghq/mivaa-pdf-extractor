/**
 * NOVA End-to-End Test - Comprehensive PDF Processing Validation
 *
 * Tests the complete PDF processing pipeline with all metrics:
 * - Product extraction and metadata
 * - Image processing and embeddings (5 types per image)
 * - Chunk creation and text embeddings
 * - Relevancy relationships (chunk-image, product-image, chunk-product)
 * - AI model usage tracking
 */

import fetch from 'node-fetch';
import fs from 'fs';
import FormData from 'form-data';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Configuration
const MIVAA_API = 'http://127.0.0.1:8000';
const API_BASE = MIVAA_API;
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

// Cleanup function
async function cleanupOldTestData() {
  log('CLEANUP', 'Deleting all old test data from database...', 'step');

  try {
    const response = await fetch(`${MIVAA_API}/api/rag/documents/jobs?limit=100&sort=created_at:desc`);
    if (!response.ok) {
      log('CLEANUP', 'Failed to fetch jobs list', 'warning');
      return;
    }

    const data = await response.json();
    const jobs = data.jobs || [];

    log('CLEANUP', `Found ${jobs.length} existing jobs`, 'info');

    for (const job of jobs) {
      if (job.document_id) {
        try {
          const deleteResponse = await fetch(`${MIVAA_API}/api/rag/documents/${job.document_id}`, {
            method: 'DELETE'
          });

          if (deleteResponse.ok) {
            log('CLEANUP', `Deleted document ${job.document_id}`, 'success');
          }
        } catch (error) {
          log('CLEANUP', `Failed to delete document ${job.document_id}: ${error.message}`, 'warning');
        }
      }
    }

    log('CLEANUP', 'Cleanup completed', 'success');
  } catch (error) {
    log('CLEANUP', `Cleanup error: ${error.message}`, 'warning');
  }
}

// Main test function
async function runNovaProductTest() {
  logSection('NOVA END-TO-END TEST - COMPREHENSIVE VALIDATION');

  console.log(`PDF: ${HARMONY_PDF_URL}`);
  console.log(`Workspace: ${WORKSPACE_ID}`);
  console.log(`MIVAA API: ${MIVAA_API}\n`);

  try {
    // Step 0: Clean up old test data
    await cleanupOldTestData();

    // Step 1: Upload PDF
    log('UPLOAD', 'Starting new PDF processing', 'step');
    const uploadResult = await uploadPDFForNovaExtraction();
    const jobId = uploadResult.job_id;
    const documentId = uploadResult.document_id;

    log('UPLOAD', `Job ID: ${jobId}`, 'info');
    log('UPLOAD', `Document ID: ${documentId}`, 'info');

    // Step 2: Monitor processing
    log('MONITOR', `Monitoring job: ${jobId}`, 'step');
    await monitorProcessingJob(jobId, documentId);

    // Step 3: Retrieve data
    log('VALIDATE', 'Retrieving ALL product data from document', 'step');
    const allData = await retrieveNovaProductData(documentId);

    // Step 4: Generate comprehensive report
    log('REPORT', 'Generating detailed report', 'step');
    await generateDetailedReport(allData, { job_id: jobId, document_id: documentId });

    log('COMPLETE', '✅ PDF processing test completed successfully!', 'success');

  } catch (error) {
    log('ERROR', `Test failed: ${error.message}`, 'error');
    console.error(error);
    process.exit(1);
  }
}




async function uploadPDFForNovaExtraction() {
  log('UPLOAD', 'Downloading PDF from URL...', 'info');

  // Download PDF
  const pdfResponse = await fetch(HARMONY_PDF_URL);
  if (!pdfResponse.ok) {
    throw new Error(`Failed to download PDF: ${pdfResponse.status}`);
  }

  const pdfBuffer = Buffer.from(await pdfResponse.arrayBuffer());

  // Save to temp file
  const tempPath = `/tmp/harmony-test-${Date.now()}.pdf`;
  fs.writeFileSync(tempPath, pdfBuffer);

  log('UPLOAD', `PDF downloaded (${(pdfBuffer.length / 1024 / 1024).toFixed(2)} MB)`, 'info');

  // Create form data with file stream
  const formData = new FormData();
  formData.append('file', fs.createReadStream(tempPath), {
    filename: 'harmony-signature-book-24-25.pdf',
    contentType: 'application/pdf'
  });

  // Add processing parameters
  formData.append('title', 'NOVA Product Extraction - End-to-End Test');
  formData.append('description', 'Extract all products from Harmony catalog');
  formData.append('tags', 'nova,harmony,test,e2e');
  formData.append('categories', 'products');
  formData.append('processing_mode', 'deep');
  formData.append('discovery_model', 'claude');
  formData.append('chunk_size', '1024');
  formData.append('chunk_overlap', '128');
  formData.append('enable_prompt_enhancement', 'true');
  formData.append('workspace_id', WORKSPACE_ID);

  log('UPLOAD', 'Uploading to MIVAA API...', 'info');

  // Upload
  const uploadResponse = await fetch(`${MIVAA_API}/api/rag/documents/upload`, {
    method: 'POST',
    body: formData,
    headers: formData.getHeaders()
  });

  if (!uploadResponse.ok) {
    const errorText = await uploadResponse.text();
    throw new Error(`Upload failed: ${uploadResponse.status}\n${errorText}`);
  }

  const result = await uploadResponse.json();
  log('UPLOAD', `✅ Upload successful!`, 'success');

  // Cleanup temp file
  try {
    fs.unlinkSync(tempPath);
  } catch (e) {
    // Ignore cleanup errors
  }

  return result;
}

async function monitorProcessingJob(jobId, documentId) {
  const maxWaitTime = 2 * 60 * 60 * 1000; // 2 hours
  const pollInterval = 10000; // 10 seconds
  const startTime = Date.now();

  log('MONITOR', 'Starting job monitoring...', 'info');

  while (Date.now() - startTime < maxWaitTime) {
    try {
      const response = await fetch(`${MIVAA_API}/api/rag/documents/jobs/${jobId}`);

      if (!response.ok) {
        log('MONITOR', `Failed to fetch job status: ${response.status}`, 'warning');
        await new Promise(resolve => setTimeout(resolve, pollInterval));
        continue;
      }

      const jobData = await response.json();
      const status = jobData.status;
      const progress = jobData.progress || 0;
      const stage = jobData.last_checkpoint?.stage || 'UNKNOWN';

      log('MONITOR', `Status: ${status} | Progress: ${progress}% | Stage: ${stage}`, 'info');

      if (status === 'completed') {
        log('MONITOR', '✅ Job completed successfully!', 'success');

        // Log final metrics
        if (jobData.metadata) {
          log('MONITOR', `Final Metrics:`, 'data');
          log('MONITOR', `  - Products: ${jobData.metadata.products_created || 0}`, 'data');
          log('MONITOR', `  - Images: ${jobData.metadata.images_extracted || 0}`, 'data');
          log('MONITOR', `  - Chunks: ${jobData.metadata.chunks_created || 0}`, 'data');
        }

        return jobData;
      }

      if (status === 'failed') {
        const error = jobData.error || 'Unknown error';
        throw new Error(`Job failed: ${error}`);
      }

      // Show progress details
      if (jobData.last_checkpoint?.metadata) {
        const meta = jobData.last_checkpoint.metadata;
        if (meta.current_step && meta.total_steps) {
          log('MONITOR', `  Step ${meta.current_step}/${meta.total_steps}: ${meta.step_name || ''}`, 'info');
        }
      }

    } catch (error) {
      log('MONITOR', `Monitoring error: ${error.message}`, 'warning');
    }

    await new Promise(resolve => setTimeout(resolve, pollInterval));
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

  // Retrieve ALL chunks
  const chunksResponse = await fetch(`${MIVAA_API}/api/rag/chunks?document_id=${documentId}&limit=10000`);
  if (chunksResponse.ok) {
    const chunksData = await chunksResponse.json();
    allData.chunks = chunksData.chunks || [];
    log('RETRIEVE', `Found ${allData.chunks.length} total chunks`, 'success');

    const chunksWithEmbeddings = allData.chunks.filter(c => c.embedding).length;
    log('RETRIEVE', `  - ${chunksWithEmbeddings} chunks have text embeddings`, 'info');
  } else {
    log('RETRIEVE', `Failed to fetch chunks: ${chunksResponse.status}`, 'error');
  }


  // Retrieve ALL images
  const imagesResponse = await fetch(`${MIVAA_API}/api/rag/images?document_id=${documentId}&limit=10000`);
  if (imagesResponse.ok) {
    const imagesData = await imagesResponse.json();
    allData.images = imagesData.images || [];
    log('RETRIEVE', `Found ${allData.images.length} total images`, 'success');

    // Query embeddings table instead of document_images columns (columns were dropped)
    const embeddingsResponse = await fetch(`${API_BASE}/api/supabase/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        table: 'embeddings',
        filters: { entity_type: 'image' }
      })
    });
    const embeddingsData = await embeddingsResponse.json();
    const imageEmbeddings = embeddingsData.data || [];

    // Count embeddings by type
    const visualEmbeddings = imageEmbeddings.filter(e => e.embedding_type === 'visual_512').length;
    const colorEmbeddings = imageEmbeddings.filter(e => e.embedding_type === 'color_512').length;
    const textureEmbeddings = imageEmbeddings.filter(e => e.embedding_type === 'texture_512').length;
    const styleEmbeddings = imageEmbeddings.filter(e => e.embedding_type === 'style_512').length;
    const materialEmbeddings = imageEmbeddings.filter(e => e.embedding_type === 'material_512').length;

    const totalClipEmbeddings = visualEmbeddings + colorEmbeddings + textureEmbeddings + styleEmbeddings + materialEmbeddings;

    log('RETRIEVE', `  - CLIP Embeddings Generated:`, 'info');
    log('RETRIEVE', `    • Visual: ${visualEmbeddings}`, 'info');
    log('RETRIEVE', `    • Color: ${colorEmbeddings}`, 'info');
    log('RETRIEVE', `    • Texture: ${textureEmbeddings}`, 'info');
    log('RETRIEVE', `    • Style: ${styleEmbeddings}`, 'info');
    log('RETRIEVE', `    • Material: ${materialEmbeddings}`, 'info');
    log('RETRIEVE', `    • TOTAL: ${totalClipEmbeddings}`, 'info');
  } else {
    log('RETRIEVE', `Failed to fetch images: ${imagesResponse.status}`, 'error');
  }

  // Retrieve ALL products
  const productsResponse = await fetch(`${MIVAA_API}/api/rag/products?document_id=${documentId}&limit=10000`);
  if (productsResponse.ok) {
    const productsData = await productsResponse.json();
    allData.products = productsData.products || [];
    log('RETRIEVE', `Found ${allData.products.length} total products`, 'success');

    const productsWithMetadata = allData.products.filter(p => p.metadata && Object.keys(p.metadata).length > 0).length;
    log('RETRIEVE', `  - ${productsWithMetadata} products have metadata`, 'info');
  } else {
    log('RETRIEVE', `Failed to fetch products: ${productsResponse.status}`, 'error');
  }

  // Retrieve chunk-image relevancies
  const chunkImageRelsResponse = await fetch(`${MIVAA_API}/api/rag/relevancies/chunk-image?document_id=${documentId}&limit=10000`);
  if (chunkImageRelsResponse.ok) {
    const relsData = await chunkImageRelsResponse.json();
    allData.chunkImageRelevancies = relsData.relevancies || [];
    log('RETRIEVE', `Found ${allData.chunkImageRelevancies.length} chunk-image relevancies`, 'success');
  }

  // Retrieve product-image relevancies
  const productImageRelsResponse = await fetch(`${MIVAA_API}/api/rag/relevancies/product-image?document_id=${documentId}&limit=10000`);
  if (productImageRelsResponse.ok) {
    const relsData = await productImageRelsResponse.json();
    allData.productImageRelevancies = relsData.relevancies || [];
    log('RETRIEVE', `Found ${allData.productImageRelevancies.length} product-image relevancies`, 'success');
  }

  // Retrieve chunk-product relevancies
  const chunkProductRelsResponse = await fetch(`${MIVAA_API}/api/rag/relevancies/chunk-product?document_id=${documentId}&limit=10000`);
  if (chunkProductRelsResponse.ok) {
    const relsData = await chunkProductRelsResponse.json();
    allData.chunkProductRelevancies = relsData.relevancies || [];
    log('RETRIEVE', `Found ${allData.chunkProductRelevancies.length} chunk-product relevancies`, 'success');
  }

  return allData;
}

async function generateDetailedReport(allData, jobResult) {
  logSection('DETAILED PDF PROCESSING REPORT');

  const chunksWithEmbeddings = allData.chunks.filter(c => c.embedding).length;
  const chunksWithMetadata = allData.chunks.filter(c => c.metadata && Object.keys(c.metadata).length > 0).length;

  // Query embeddings table for image embeddings (document_images columns were dropped)
  const embeddingsResponse = await fetch(`${API_BASE}/api/supabase/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      table: 'embeddings',
      filters: { entity_type: 'image' }
    })
  });
  const embeddingsData = await embeddingsResponse.json();
  const imageEmbeddings = embeddingsData.data || [];

  const visualEmbeddings = imageEmbeddings.filter(e => e.embedding_type === 'visual_512').length;
  const colorEmbeddings = imageEmbeddings.filter(e => e.embedding_type === 'color_512').length;
  const textureEmbeddings = imageEmbeddings.filter(e => e.embedding_type === 'texture_512').length;
  const styleEmbeddings = imageEmbeddings.filter(e => e.embedding_type === 'style_512').length;
  const materialEmbeddings = imageEmbeddings.filter(e => e.embedding_type === 'material_512').length;
  const totalClipEmbeddings = visualEmbeddings + colorEmbeddings + textureEmbeddings + styleEmbeddings + materialEmbeddings;

  const productsWithMetadata = allData.products.filter(p => p.metadata && Object.keys(p.metadata).length > 0).length;

  const report = {
    timestamp: new Date().toISOString(),
    job: {
      id: jobResult.job_id,
      document_id: jobResult.document_id
    },
    summary: {
      total_chunks: allData.chunks.length,
      chunks_with_embeddings: chunksWithEmbeddings,
      chunks_with_metadata: chunksWithMetadata,
      total_images: allData.images.length,
      clip_embeddings: {
        visual: visualEmbeddings,
        color: colorEmbeddings,
        texture: textureEmbeddings,
        style: styleEmbeddings,
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

  // Print detailed summary
  logSection('📊 FINAL SUMMARY - NOVA END-TO-END TEST RESULTS');

  console.log('\n' + '='.repeat(100));
  console.log('1️⃣  PRODUCTS');
  console.log('='.repeat(100));
  console.log(`   ✅ Total Products: ${report.summary.total_products}`);
  console.log(`   ✅ Products with Metadata: ${productsWithMetadata}`);

  console.log('\n' + '='.repeat(100));
  console.log('2️⃣  CLIP EMBEDDINGS (5 types per image)');
  console.log('='.repeat(100));
  console.log(`   ✅ Visual Embeddings: ${report.summary.clip_embeddings.visual}`);
  console.log(`   ✅ Color Embeddings: ${report.summary.clip_embeddings.color}`);
  console.log(`   ✅ Texture Embeddings: ${report.summary.clip_embeddings.texture}`);
  console.log(`   ✅ Style Embeddings: ${report.summary.clip_embeddings.style}`);
  console.log(`   ✅ Material Embeddings: ${report.summary.clip_embeddings.material}`);
  console.log(`   ✅ TOTAL CLIP Embeddings: ${report.summary.clip_embeddings.total}`);

  console.log('\n' + '='.repeat(100));
  console.log('3️⃣  IMAGES');
  console.log('='.repeat(100));
  console.log(`   ✅ Total Images in DB: ${report.summary.total_images}`);
  console.log(`   ✅ Expected: ${report.summary.total_images} images × 5 embedding types = ${report.summary.total_images * 5} embeddings`);
  console.log(`   ✅ Actual: ${report.summary.clip_embeddings.total} embeddings`);
  const embeddingsMatch = report.summary.clip_embeddings.total === report.summary.total_images * 5;
  console.log(`   ${embeddingsMatch ? '✅' : '❌'} Embeddings Match: ${embeddingsMatch ? 'YES' : 'NO'}`);

  console.log('\n' + '='.repeat(100));
  console.log('4️⃣  PRODUCT-TO-IMAGE RELEVANCIES');
  console.log('='.repeat(100));
  console.log(`   ✅ Total Relevancies: ${report.summary.relevancies.product_image}`);

  // Group relevancies by product
  const relevanciesByProduct = {};
  allData.productImageRelevancies.forEach(rel => {
    if (!relevanciesByProduct[rel.product_id]) {
      relevanciesByProduct[rel.product_id] = [];
    }
    relevanciesByProduct[rel.product_id].push(rel);
  });

  console.log(`   ✅ Products with Images: ${Object.keys(relevanciesByProduct).length}`);
  Object.entries(relevanciesByProduct).forEach(([productId, rels]) => {
    const product = allData.products.find(p => p.id === productId);
    const productName = product?.name || product?.metadata?.name || 'Unknown';
    console.log(`      • ${productName}: ${rels.length} images`);
  });

  console.log('\n' + '='.repeat(100));
  console.log('5️⃣  TOTAL EMBEDDINGS');
  console.log('='.repeat(100));
  const totalEmbeddings = report.summary.chunks_with_embeddings + report.summary.clip_embeddings.total;
  console.log(`   ✅ Text Embeddings (chunks): ${report.summary.chunks_with_embeddings}`);
  console.log(`   ✅ Image Embeddings (CLIP): ${report.summary.clip_embeddings.total}`);
  console.log(`   ✅ TOTAL EMBEDDINGS: ${totalEmbeddings}`);

  console.log('\n' + '='.repeat(100));
  console.log('6️⃣  ALL RELATIONSHIP COUNTS');
  console.log('='.repeat(100));
  console.log(`   ✅ Embeddings-to-Products: ${report.summary.total_products} products`);
  console.log(`   ✅ Chunks-to-Products: ${report.summary.relevancies.chunk_product} relationships`);
  console.log(`   ✅ Chunks-to-Images: ${report.summary.relevancies.chunk_image} relationships`);
  console.log(`   ✅ Products-to-Images: ${report.summary.relevancies.product_image} relationships`);
  console.log(`   ✅ TOTAL RELATIONSHIPS: ${report.summary.relevancies.total}`);

  console.log('\n' + '='.repeat(100));
  console.log('7️⃣  META GENERATED & META-RELATED EMBEDDINGS');
  console.log('='.repeat(100));
  console.log(`   ✅ Products with Metadata: ${productsWithMetadata}`);
  console.log(`   ✅ Chunks with Metadata: ${chunksWithMetadata}`);
  console.log(`   ✅ Meta-related Embeddings: ${report.summary.clip_embeddings.color + report.summary.clip_embeddings.texture + report.summary.clip_embeddings.style + report.summary.clip_embeddings.material}`);

  console.log('\n' + '='.repeat(100));

  // Save report to file
  const reportPath = path.join(__dirname, `pdf-processing-report-${Date.now()}.json`);
  fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
  log('REPORT', `Detailed report saved to: ${reportPath}`, 'success');
}

// Run the test
runNovaProductTest().catch(error => {
  console.error('Fatal error:', error);
  process.exit(1);
});


