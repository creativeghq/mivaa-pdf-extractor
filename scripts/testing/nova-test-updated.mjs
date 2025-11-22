/**
 * NOVA Product Focused End-to-End Test
 * 
 * Tests the complete PDF processing pipeline with comprehensive metrics
 */

import fetch from 'node-fetch';
import fs from 'fs';
import FormData from 'form-data';
import { Blob } from 'buffer';

// Configuration - Use LOCAL API on server
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

// Cleanup function to delete all old test data
async function cleanupOldTestData() {
  log('CLEANUP', 'Deleting all old test data from database...', 'step');

  try {
    // Find all Harmony PDF documents
    const response = await fetch(`${MIVAA_API}/api/rag/documents/jobs?limit=100&sort=created_at:desc`);
    if (!response.ok) {
      log('CLEANUP', 'Failed to fetch jobs list', 'warning');
      return;
    }

    const data = await response.json();
    const jobs = data.jobs || [];

    // Filter for Harmony PDF jobs
    const harmonyJobs = jobs.filter(job => {
      const filename = job.metadata?.filename || job.filename || '';
      return filename.includes('harmony-signature-book-24-25');
    });

    if (harmonyJobs.length === 0) {
      log('CLEANUP', 'No old Harmony PDF jobs found', 'info');
      return;
    }

    log('CLEANUP', `Found ${harmonyJobs.length} old Harmony PDF jobs to delete`, 'info');

    // Delete each job and its associated data
    for (const job of harmonyJobs) {
      const jobId = job.id;
      const documentId = job.document_id;

      log('CLEANUP', `Deleting job ${jobId} and document ${documentId}...`, 'info');

      try {
        // Delete job (this should cascade to related data via foreign keys)
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

    // Wait a bit for database cleanup to complete
    await new Promise(resolve => setTimeout(resolve, 2000));

  } catch (error) {
    log('CLEANUP', `Error during cleanup: ${error.message}`, 'error');
  }
}

// Main test function
async function runNovaProductTest() {
  logSection('NOVA PRODUCT FOCUSED END-TO-END TEST');

  console.log(`Product: ${NOVA_PRODUCT.name} by ${NOVA_PRODUCT.designer}`);
  console.log(`PDF: ${HARMONY_PDF_URL}`);
  console.log(`Workspace: ${WORKSPACE_ID}`);
  console.log(`MIVAA API: ${MIVAA_API}\n`);

  try {
    // Step 0: Clean up old test data
    await cleanupOldTestData();

    // Step 1: Upload PDF with async processing (always start fresh)
    log('UPLOAD', 'Starting new PDF processing', 'step');
    const uploadResult = await uploadPDFForNovaExtraction();
    const jobId = uploadResult.job_id;
    const documentId = uploadResult.document_id;

    log('UPLOAD', `Job ID: ${jobId}`, 'info');
    log('UPLOAD', `Document ID: ${documentId}`, 'info');

    // Step 2: Monitor async job processing
    log('MONITOR', `Monitoring job: ${jobId}`, 'step');
    await monitorProcessingJob(jobId, documentId);

    // Step 3: Retrieve and validate ALL product data
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
  log('UPLOAD', 'Uploading Harmony PDF for processing...', 'info');

  // Download PDF from URL
  const pdfResponse = await fetch(HARMONY_PDF_URL);
  if (!pdfResponse.ok) {
    throw new Error(`Failed to download PDF: ${pdfResponse.status} ${pdfResponse.statusText}`);
  }

  const pdfBuffer = await pdfResponse.arrayBuffer();
  const pdfBlob = new Blob([pdfBuffer], { type: 'application/pdf' });

  // Create form data
  const formData = new FormData();
  formData.append('file', pdfBlob, 'harmony-signature-book-24-25.pdf');

  // Add processing parameters using latest API specification
  formData.append('title', 'NOVA Product Extraction - Focused Test');
  formData.append('description', 'Extract all products from Harmony catalog');
  formData.append('tags', 'nova,harmony,test');
  formData.append('categories', 'products');  // Extract only products
  formData.append('processing_mode', 'deep');  // Deep mode for complete analysis
  formData.append('discovery_model', 'claude');  // Claude Sonnet 4.5 for product discovery
  formData.append('chunk_size', '1024');
  formData.append('chunk_overlap', '128');
  formData.append('enable_prompt_enhancement', 'true');
  formData.append('workspace_id', WORKSPACE_ID);

  // Upload and start async processing
  const uploadResponse = await fetch(`${MIVAA_API}/api/rag/documents/upload`, {
    method: 'POST',
    body: formData
  });

  if (!uploadResponse.ok) {
    const errorText = await uploadResponse.text();
    throw new Error(`Upload failed: ${uploadResponse.status} ${uploadResponse.statusText}\n${errorText}`);
  }

  const result = await uploadResponse.json();
  log('UPLOAD', `✅ Upload successful! Job ID: ${result.job_id}`, 'success');

  return result;
}

async function monitorProcessingJob(jobId, documentId) {
  const maxWaitTime = 2 * 60 * 60 * 1000; // 2 hours
  const pollInterval = 10000; // 10 seconds
  const startTime = Date.now();

  log('MONITOR', 'Starting job monitoring...', 'info');

  while (Date.now() - startTime < maxWaitTime) {
    // Get job status
    const statusResponse = await fetch(`${MIVAA_API}/api/rag/documents/jobs/${jobId}`);
    
    if (!statusResponse.ok) {
      log('MONITOR', `Failed to get job status: ${statusResponse.status}`, 'warning');
      await new Promise(resolve => setTimeout(resolve, pollInterval));
      continue;
    }

    const jobData = await statusResponse.json();
    const status = jobData.status;
    const progress = jobData.progress || 0;
    const currentStage = jobData.last_checkpoint?.stage || 'unknown';

    log('MONITOR', `Status: ${status} | Progress: ${progress}% | Stage: ${currentStage}`, 'info');

    // Check if completed
    if (status === 'completed' && progress === 100) {
      log('MONITOR', '✅ Job completed successfully!', 'success');
      
      // Display final metrics
      if (jobData.metadata) {
        log('MONITOR', 'Final Metrics:', 'data');
        log('MONITOR', `  - Total Pages: ${jobData.metadata.total_pages || 'N/A'}`, 'info');
        log('MONITOR', `  - Products Found: ${jobData.metadata.products_count || 'N/A'}`, 'info');
        log('MONITOR', `  - Images Extracted: ${jobData.metadata.images_count || 'N/A'}`, 'info');
        log('MONITOR', `  - Chunks Created: ${jobData.metadata.chunks_count || 'N/A'}`, 'info');
      }
      
      return jobData;
    }

    // Check for errors
    if (status === 'failed') {
      const error = jobData.error || 'Unknown error';
      log('MONITOR', `❌ Job failed: ${error}`, 'error');
      throw new Error(`Job failed: ${error}`);
    }

    if (status === 'interrupted') {
      log('MONITOR', '⚠️ Job interrupted! Attempting to resume...', 'warning');
      // Try to resume the job
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

    // Wait before next poll
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

    const chunksWithMetadata = allData.chunks.filter(c => c.metadata && Object.keys(c.metadata).length > 0).length;
    log('RETRIEVE', `  - ${chunksWithMetadata} chunks have metadata`, 'info');
  } else {
    const errorText = await chunksResponse.text();
    log('RETRIEVE', `Failed to fetch chunks: ${chunksResponse.status}`, 'error');
  }

  // Retrieve ALL images
  const imagesResponse = await fetch(`${MIVAA_API}/api/rag/images?document_id=${documentId}&limit=10000`);
  if (imagesResponse.ok) {
    const imagesData = await imagesResponse.json();
    allData.images = imagesData.images || [];
    log('RETRIEVE', `Found ${allData.images.length} total images`, 'success');

    // Count CLIP embeddings (5 types per image)
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
  const chunkImageRelResponse = await fetch(`${MIVAA_API}/api/rag/relevancies?document_id=${documentId}&limit=10000`);
  if (chunkImageRelResponse.ok) {
    const relData = await chunkImageRelResponse.json();
    allData.chunkImageRelevancies = relData.relevancies || [];
    log('RETRIEVE', `Found ${allData.chunkImageRelevancies.length} chunk-image relevancies`, 'success');
  } else {
    log('RETRIEVE', `Failed to fetch chunk-image relevancies: ${chunkImageRelResponse.status}`, 'error');
  }

  // Retrieve product-image relevancies
  const productImageRelResponse = await fetch(`${MIVAA_API}/api/rag/product-image-relationships?document_id=${documentId}&limit=10000`);
  if (productImageRelResponse.ok) {
    const relData = await productImageRelResponse.json();
    allData.productImageRelevancies = relData.relationships || [];
    log('RETRIEVE', `Found ${allData.productImageRelevancies.length} product-image relevancies`, 'success');
  } else {
    log('RETRIEVE', `Failed to fetch product-image relevancies: ${productImageRelResponse.status}`, 'warning');
  }

  // Retrieve chunk-product relevancies
  const chunkProductRelResponse = await fetch(`${MIVAA_API}/api/rag/chunk-product-relationships?document_id=${documentId}&limit=10000`);
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

  // Count CLIP embeddings (5 types per image)
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

  // Print detailed summary with requested metrics
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
