/**
 * NOVA Product Focused End-to-End Test
 * Tests the complete PDF processing pipeline from Harmony PDF catalog
 */

import fetch from 'node-fetch';
import FormData from 'form-data';

const MIVAA_API = 'http://127.0.0.1:8000';
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

async function uploadPDFForNovaExtraction() {
  const formData = new FormData();
  formData.append('file_url', HARMONY_PDF_URL);
  formData.append('workspace_id', WORKSPACE_ID);
  formData.append('title', 'Harmony Signature Book 24-25 - NOVA Test');
  formData.append('description', 'Testing NOVA product extraction with Claude Vision');
  formData.append('discovery_model', 'claude-vision');
  formData.append('categories', 'products');

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

async function monitorProcessingJob(jobId, documentId) {
  const maxAttempts = 480;
  let attempts = 0;
  let lastProgress = -1;

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

    if (progress !== lastProgress) {
      log('MONITOR', `[${attempts}/${maxAttempts}] ${status.toUpperCase()} (${progress}%) - ${stage} | Products: ${products} | Chunks: ${chunks} | Images: ${images}`);
      lastProgress = progress;
    }

    if (status === 'completed') {
      log('MONITOR', '✅ Job completed successfully!', 'success');
      return job;
    }

    if (status === 'failed') {
      log('MONITOR', `❌ Job failed: ${job.error}`, 'error');
      throw new Error(`Job failed: ${job.error}`);
    }

    await new Promise(resolve => setTimeout(resolve, 5000));
  }

  throw new Error('Job monitoring timeout');
}

async function retrieveNovaProductData(documentId) {
  const response = await fetch(`${MIVAA_API}/api/rag/documents/${documentId}/products`);
  if (!response.ok) {
    throw new Error(`Failed to retrieve products: ${response.status}`);
  }
  return await response.json();
}

async function generateDetailedReport(data, jobInfo) {
  logSection('COMPREHENSIVE TEST REPORT');
  
  log('REPORT', `Job ID: ${jobInfo.job_id}`, 'data');
  log('REPORT', `Document ID: ${jobInfo.document_id}`, 'data');
  
  log('REPORT', `Total Products: ${data.products?.length || 0}`, 'data');
  log('REPORT', `Total Chunks: ${data.chunks?.length || 0}`, 'data');
  log('REPORT', `Total Images: ${data.images?.length || 0}`, 'data');
  
  console.log('\n');
}

async function runNovaProductTest() {
  logSection('NOVA PRODUCT EXTRACTION TEST - CLAUDE VISION');
  console.log(`Product: ${NOVA_PRODUCT.name} by ${NOVA_PRODUCT.designer}`);
  console.log(`PDF: ${HARMONY_PDF_URL}`);
  console.log(`Workspace: ${WORKSPACE_ID}`);
  console.log(`MIVAA API: ${MIVAA_API}\n`);

  try {
    log('UPLOAD', 'Starting new PDF processing with Claude Vision', 'step');
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
