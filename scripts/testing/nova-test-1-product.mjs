/**
 * NOVA Single Product Test - Extract ONLY 1 product for fast validation
 */

import fetch from 'node-fetch';
import FormData from 'form-data';

const MIVAA_API = 'http://127.0.0.1:8000';
const HARMONY_PDF_URL = 'https://bgbavxtjlbvgplozizxu.supabase.co/storage/v1/object/public/pdf-documents/harmony-signature-book-24-25.pdf';
const WORKSPACE_ID = 'ffafc28b-1b8b-4b0d-b226-9f9a6154004e';

function log(category, message, level = 'info') {
  const emoji = {
    'step': '📋',
    'info': '📝',
    'success': '✅',
    'error': '❌',
    'warning': '⚠️',
    'data': '📊'
  }[level] || '📝';
  
  const timestamp = new Date().toISOString().split('T')[1].split('.')[0];
  console.log(`[${timestamp}] ${emoji} [${category}] ${message}`);
}

function logSection(title) {
  console.log('\n' + '='.repeat(100));
  console.log(`🎯 ${title}`);
  console.log('='.repeat(100));
}

async function uploadPDFForSingleProduct() {
  const formData = new FormData();
  formData.append('file_url', HARMONY_PDF_URL);
  formData.append('workspace_id', WORKSPACE_ID);
  formData.append('title', 'Harmony - SINGLE PRODUCT TEST');
  formData.append('description', 'Testing with ONLY 1 product extraction');
  formData.append('discovery_model', 'claude-vision');
  formData.append('categories', 'products');
  formData.append('agent_prompt', 'Extract ONLY the FIRST product you find. Stop after finding 1 product. Do not extract any other products.');

  const response = await fetch(`${MIVAA_API}/api/rag/documents/upload`, {
    method: 'POST',
    body: formData
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Upload failed: ${response.status} ${response.statusText}\n${errorText}`);
  }

  const result = await response.json();
  return result;
}

async function monitorProcessingJob(jobId) {
  const maxAttempts = 240; // 20 minutes max
  let attempts = 0;
  let lastProgress = -1;
  let lastStage = '';

  while (attempts < maxAttempts) {
    attempts++;
    
    const response = await fetch(`${MIVAA_API}/api/rag/documents/job/${jobId}`);
    const job = await response.json();

    const progress = job.progress || 0;
    const status = job.status;
    const stage = job.metadata?.current_stage || 'N/A';
    const products = job.metadata?.products_created || 0;
    const chunks = job.metadata?.chunks_created || 0;
    const images = job.metadata?.images_extracted || 0;

    // Log on progress change OR stage change
    if (progress !== lastProgress || stage !== lastStage) {
      log('PROGRESS', `${status.toUpperCase()} (${progress}%) - ${stage} | Products: ${products} | Chunks: ${chunks} | Images: ${images}`);
      lastProgress = progress;
      lastStage = stage;
    }

    if (status === 'completed') {
      log('COMPLETE', '✅ Job completed successfully!', 'success');
      return job;
    }

    if (status === 'failed') {
      log('ERROR', `❌ Job failed: ${job.error}`, 'error');
      throw new Error(`Job failed: ${job.error}`);
    }

    await new Promise(resolve => setTimeout(resolve, 3000)); // Check every 3 seconds
  }

  throw new Error('Job monitoring timeout after 20 minutes');
}

async function retrieveProductData(documentId) {
  const response = await fetch(`${MIVAA_API}/api/rag/documents/${documentId}/products`);
  if (!response.ok) {
    throw new Error(`Failed to retrieve products: ${response.status}`);
  }
  return await response.json();
}

async function generateDetailedReport(data, jobInfo) {
  logSection('📊 SINGLE PRODUCT TEST RESULTS');
  
  log('REPORT', `Job ID: ${jobInfo.job_id}`, 'data');
  log('REPORT', `Document ID: ${jobInfo.document_id}`, 'data');
  
  const products = data.products || [];
  const chunks = data.chunks || [];
  const images = data.images || [];
  
  log('REPORT', `Total Products: ${products.length}`, 'data');
  log('REPORT', `Total Chunks: ${chunks.length}`, 'data');
  log('REPORT', `Total Images: ${images.length}`, 'data');
  
  if (products.length > 0) {
    console.log('\n📦 PRODUCT DETAILS:');
    products.forEach((product, idx) => {
      console.log(`\n  Product ${idx + 1}:`);
      console.log(`    Name: ${product.name || 'N/A'}`);
      console.log(`    Designer: ${product.metadata?.designer || 'N/A'}`);
      console.log(`    Pages: ${product.metadata?.page_range || 'N/A'}`);
      console.log(`    Images: ${product.metadata?.image_count || 0}`);
    });
  }
  
  console.log('\n');
}

async function runSingleProductTest() {
  logSection('🚀 SINGLE PRODUCT EXTRACTION TEST - CLAUDE VISION');
  console.log(`PDF: ${HARMONY_PDF_URL}`);
  console.log(`Workspace: ${WORKSPACE_ID}`);
  console.log(`MIVAA API: ${MIVAA_API}`);
  console.log(`Goal: Extract ONLY 1 product for fast validation\n`);

  try {
    log('UPLOAD', 'Starting PDF processing with 1-product limit', 'step');
    const uploadResult = await uploadPDFForSingleProduct();
    const jobId = uploadResult.job_id;
    const documentId = uploadResult.document_id;

    log('UPLOAD', `Job ID: ${jobId}`, 'info');
    log('UPLOAD', `Document ID: ${documentId}`, 'info');

    log('MONITOR', `Monitoring job progress...`, 'step');
    const finalJob = await monitorProcessingJob(jobId);

    log('VALIDATE', 'Retrieving product data', 'step');
    const allData = await retrieveProductData(documentId);

    await generateDetailedReport(allData, { job_id: jobId, document_id: documentId });

    log('SUCCESS', '✅ Single product test completed successfully!', 'success');

  } catch (error) {
    log('ERROR', `Test failed: ${error.message}`, 'error');
    console.error(error);
    process.exit(1);
  }
}

runSingleProductTest().catch(error => {
  console.error('Fatal error:', error);
  process.exit(1);
});
