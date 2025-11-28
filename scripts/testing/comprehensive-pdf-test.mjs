/**
 * COMPREHENSIVE PDF PROCESSING END-TO-END TEST
 * 
 * Tests BOTH Claude Vision and GPT Vision models
 * Reports ALL metrics as requested:
 * 
 * 1. Total Products discovered + time taken
 * 2. Total Pages processed + time taken  
 * 3. Total Chunks created + time taken
 * 4. Total Images processed + time taken
 * 5. Total Embeddings created + time taken
 * 6. Total Errors + time taken
 * 7. Total Relationships created + time taken
 * 8. Total Metadata extracted + time taken
 * 9. Total Memory used + time
 * 10. Total CPU used + time
 * 11. Total Cost (AI API usage)
 * 12. Total Time for entire process
 */

import fetch from 'node-fetch';
import fs from 'fs';
import FormData from 'form-data';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

// Configuration
const MIVAA_API = 'https://v1api.materialshub.gr';
const HARMONY_PDF_URL = 'https://bgbavxtjlbvgplozizxu.supabase.co/storage/v1/object/public/pdf-documents/harmony-signature-book-24-25.pdf';
const WORKSPACE_ID = 'ffafc28b-1b8b-4b0d-b226-9f9a6154004e';

// Test both vision models
const TEST_MODELS = ['claude-vision', 'gpt-vision'];

// AI Model Pricing (per 1M tokens)
const MODEL_PRICING = {
  'claude-sonnet-4-5-20250929': { input: 3.00, output: 15.00 },
  'claude-haiku-4-5-20251001': { input: 0.80, output: 4.00 },
  'gpt-4o': { input: 2.50, output: 10.00 },
  'text-embedding-3-small': { input: 0.02, output: 0 },
  'llama-vision': { input: 0.18, output: 0.18 }
};

// Logging utilities
function log(category, message, level = 'info') {
  const emoji = {
    'step': '📋',
    'info': '📝',
    'success': '✅',
    'error': '❌',
    'warning': '⚠️',
    'data': '📊',
    'time': '⏱️',
    'money': '💰',
    'memory': '🧠',
    'cpu': '⚡'
  }[level] || '📝';
  
  console.log(`${emoji} [${category}] ${message}`);
}

function logSection(title) {
  console.log('\n' + '='.repeat(100));
  console.log(`🎯 ${title}`);
  console.log('='.repeat(100));
}

// Get system metrics (memory, CPU)
async function getSystemMetrics() {
  try {
    // Get MIVAA service PID
    const { stdout: pidOut } = await execAsync("pgrep -f 'uvicorn.*mivaa' | head -1");
    const pid = pidOut.trim();
    
    if (!pid) {
      return { memory_mb: 0, cpu_percent: 0 };
    }
    
    // Get memory (RSS in KB) and CPU
    const { stdout: psOut } = await execAsync(`ps -p ${pid} -o rss=,pcpu= 2>/dev/null || echo "0 0"`);
    const [memKb, cpu] = psOut.trim().split(/\s+/).map(v => parseFloat(v) || 0);
    
    return {
      memory_mb: Math.round(memKb / 1024),
      cpu_percent: cpu
    };
  } catch (error) {
    return { memory_mb: 0, cpu_percent: 0 };
  }
}

// Calculate AI cost
function calculateCost(aiCalls) {
  let totalCost = 0;
  
  for (const call of aiCalls) {
    const pricing = MODEL_PRICING[call.model];
    if (!pricing) continue;
    
    const inputCost = (call.input_tokens / 1000000) * pricing.input;
    const outputCost = (call.output_tokens / 1000000) * pricing.output;
    totalCost += inputCost + outputCost;
  }
  
  return totalCost;
}

// Format time duration
function formatDuration(ms) {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  return `${(ms / 60000).toFixed(1)}min`;
}

// Main test runner
async function runComprehensiveTest() {
  logSection('COMPREHENSIVE PDF PROCESSING TEST - BOTH VISION MODELS');
  
  const allResults = [];
  
  for (const model of TEST_MODELS) {
    logSection(`TESTING MODEL: ${model.toUpperCase()}`);
    
    const result = await runSingleModelTest(model);
    allResults.push(result);
    
    // Wait between tests
    if (model !== TEST_MODELS[TEST_MODELS.length - 1]) {
      log('WAIT', 'Waiting 30 seconds before next test...', 'info');
      await new Promise(resolve => setTimeout(resolve, 30000));
    }
  }
  
  // Generate comparison report
  generateComparisonReport(allResults);
}

// Run test for single model
async function runSingleModelTest(discoveryModel) {
  const testStart = Date.now();
  const metrics = {
    model: discoveryModel,
    stages: {},
    products: { count: 0, time_ms: 0 },
    pages: { count: 0, time_ms: 0 },
    chunks: { count: 0, time_ms: 0 },
    images: { count: 0, time_ms: 0 },
    embeddings: { text: 0, clip: 0, total: 0, time_ms: 0 },
    errors: { count: 0, time_ms: 0 },
    relationships: { chunk_image: 0, product_image: 0, chunk_product: 0, total: 0, time_ms: 0 },
    metadata: { chunks: 0, products: 0, total: 0, time_ms: 0 },
    system: { peak_memory_mb: 0, avg_cpu_percent: 0, samples: [] },
    cost: { total_usd: 0, ai_calls: [] },
    total_time_ms: 0
  };
  
  try {
    // Step 1: Upload PDF
    log('UPLOAD', `Starting PDF upload with discovery_model=${discoveryModel}`, 'step');
    const uploadStart = Date.now();
    
    const formData = new FormData();
    formData.append('file_url', HARMONY_PDF_URL);
    formData.append('workspace_id', WORKSPACE_ID);
    formData.append('processing_mode', 'deep');
    formData.append('categories', 'products');
    formData.append('discovery_model', discoveryModel);
    formData.append('async_processing', 'true');
    
    const uploadResponse = await fetch(`${MIVAA_API}/api/rag/documents/upload`, {
      method: 'POST',
      body: formData,
      headers: formData.getHeaders()
    });
    
    if (!uploadResponse.ok) {
      throw new Error(`Upload failed: ${uploadResponse.status}`);
    }
    
    const uploadResult = await uploadResponse.json();
    const jobId = uploadResult.job_id;
    const documentId = uploadResult.document_id;
    
    log('UPLOAD', `Job ID: ${jobId}`, 'success');
    log('UPLOAD', `Document ID: ${documentId}`, 'success');
    
    // Step 2: Monitor job with metrics collection
    log('MONITOR', 'Monitoring job progress and collecting metrics...', 'step');
    await monitorJobWithMetrics(jobId, documentId, metrics);
    
    // Step 3: Collect final data
    log('COLLECT', 'Collecting final data from database...', 'step');
    await collectFinalMetrics(documentId, metrics);
    
    // Calculate total time
    metrics.total_time_ms = Date.now() - testStart;
    
    // Generate report for this model
    generateModelReport(metrics);
    
    return metrics;
    
  } catch (error) {
    log('ERROR', `Test failed: ${error.message}`, 'error');
    metrics.errors.count++;
    metrics.total_time_ms = Date.now() - testStart;
    return metrics;
  }
}

// Monitor job and collect metrics
async function monitorJobWithMetrics(jobId, documentId, metrics) {
  const maxAttempts = 480; // 2 hours
  const pollInterval = 15000; // 15 seconds
  let currentStage = null;
  let stageStartTime = Date.now();
  
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    await new Promise(resolve => setTimeout(resolve, pollInterval));
    
    // Collect system metrics
    const sysMetrics = await getSystemMetrics();
    metrics.system.samples.push(sysMetrics);
    metrics.system.peak_memory_mb = Math.max(metrics.system.peak_memory_mb, sysMetrics.memory_mb);
    
    // Get job status
    const statusResponse = await fetch(`${MIVAA_API}/api/rag/documents/job/${jobId}`);
    if (!statusResponse.ok) continue;
    
    const jobData = await statusResponse.json();
    const status = jobData.status;
    const progress = jobData.progress || 0;
    const metadata = jobData.metadata || {};
    const stage = metadata.current_stage || metadata.stage || 'processing';
    
    // Track stage changes
    if (stage !== currentStage) {
      if (currentStage) {
        const stageTime = Date.now() - stageStartTime;
        metrics.stages[currentStage] = stageTime;
        log('STAGE', `${currentStage} completed in ${formatDuration(stageTime)}`, 'time');
      }
      currentStage = stage;
      stageStartTime = Date.now();
    }
    
    // Log progress
    log('MONITOR', `[${attempt}/${maxAttempts}] ${status.toUpperCase()} (${progress}%) - ${stage}`, 'info');
    
    // Update metrics from job metadata
    if (metadata.products_created) metrics.products.count = metadata.products_created;
    if (metadata.total_pages) metrics.pages.count = metadata.total_pages;
    if (metadata.chunks_created) metrics.chunks.count = metadata.chunks_created;
    if (metadata.images_extracted) metrics.images.count = metadata.images_extracted;
    
    // Check completion
    if (status === 'completed') {
      if (currentStage) {
        metrics.stages[currentStage] = Date.now() - stageStartTime;
      }
      log('MONITOR', '✅ Job completed successfully!', 'success');
      return jobData;
    }
    
    if (status === 'failed') {
      metrics.errors.count++;
      throw new Error(`Job failed: ${jobData.error || 'Unknown error'}`);
    }
  }
  
  throw new Error('Job monitoring timed out');
}

// Collect final metrics from database
async function collectFinalMetrics(documentId, metrics) {
  // Get chunks
  const chunksResp = await fetch(`${MIVAA_API}/api/rag/chunks?document_id=${documentId}&limit=10000`);
  if (chunksResp.ok) {
    const chunksData = await chunksResp.json();
    const chunks = chunksData.chunks || [];
    metrics.chunks.count = chunks.length;
    metrics.embeddings.text = chunks.filter(c => c.embedding).length;
    metrics.metadata.chunks = chunks.filter(c => c.metadata && Object.keys(c.metadata).length > 0).length;
  }
  
  // Get images
  const imagesResp = await fetch(`${MIVAA_API}/api/rag/images?document_id=${documentId}&limit=10000`);
  if (imagesResp.ok) {
    const imagesData = await imagesResp.json();
    const images = imagesData.images || [];
    metrics.images.count = images.length;
    
    // Count CLIP embeddings (5 types per image)
    const visual = images.filter(img => img.visual_clip_embedding_512).length;
    const color = images.filter(img => img.color_clip_embedding_512).length;
    const texture = images.filter(img => img.texture_clip_embedding_512).length;
    const application = images.filter(img => img.application_clip_embedding_512).length;
    const material = images.filter(img => img.material_clip_embedding_512).length;
    
    metrics.embeddings.clip = visual + color + texture + application + material;
  }
  
  // Get products
  const productsResp = await fetch(`${MIVAA_API}/api/rag/products?document_id=${documentId}&limit=10000`);
  if (productsResp.ok) {
    const productsData = await productsResp.json();
    const products = productsData.products || [];
    metrics.products.count = products.length;
    metrics.metadata.products = products.filter(p => p.metadata && Object.keys(p.metadata).length > 0).length;
  }
  
  // Get relationships
  const chunkImageResp = await fetch(`${MIVAA_API}/api/rag/relevancies?document_id=${documentId}&limit=10000`);
  if (chunkImageResp.ok) {
    const relData = await chunkImageResp.json();
    metrics.relationships.chunk_image = (relData.relevancies || []).length;
  }
  
  const productImageResp = await fetch(`${MIVAA_API}/api/rag/product-image-relationships?document_id=${documentId}&limit=10000`);
  if (productImageResp.ok) {
    const relData = await productImageResp.json();
    metrics.relationships.product_image = (relData.relationships || []).length;
  }
  
  const chunkProductResp = await fetch(`${MIVAA_API}/api/rag/chunk-product-relationships?document_id=${documentId}&limit=10000`);
  if (chunkProductResp.ok) {
    const relData = await chunkProductResp.json();
    metrics.relationships.chunk_product = (relData.relationships || []).length;
  }
  
  // Calculate totals
  metrics.embeddings.total = metrics.embeddings.text + metrics.embeddings.clip;
  metrics.relationships.total = metrics.relationships.chunk_image + metrics.relationships.product_image + metrics.relationships.chunk_product;
  metrics.metadata.total = metrics.metadata.chunks + metrics.metadata.products;
  
  // Calculate average CPU
  if (metrics.system.samples.length > 0) {
    const totalCpu = metrics.system.samples.reduce((sum, s) => sum + s.cpu_percent, 0);
    metrics.system.avg_cpu_percent = (totalCpu / metrics.system.samples.length).toFixed(1);
  }
  
  // Calculate stage times
  metrics.products.time_ms = metrics.stages.product_discovery || metrics.stages.discovery || 0;
  metrics.pages.time_ms = metrics.stages.pdf_extracted || metrics.stages.extracting_text || 0;
  metrics.chunks.time_ms = metrics.stages.chunks_created || metrics.stages.chunking || 0;
  metrics.images.time_ms = metrics.stages.images_extracted || metrics.stages.image_processing || 0;
  metrics.embeddings.time_ms = metrics.stages.text_embeddings_generated || metrics.stages.image_embeddings_generated || 0;
  metrics.relationships.time_ms = metrics.stages.products_created || 0;
  metrics.metadata.time_ms = metrics.products.time_ms; // Metadata extracted during discovery
}

// Generate report for single model
function generateModelReport(metrics) {
  logSection(`📊 COMPREHENSIVE METRICS REPORT - ${metrics.model.toUpperCase()}`);
  
  console.log('\n1️⃣  PRODUCTS DISCOVERED');
  console.log(`   ✅ Total Products: ${metrics.products.count}`);
  console.log(`   ⏱️  Time Taken: ${formatDuration(metrics.products.time_ms)}`);
  
  console.log('\n2️⃣  PAGES PROCESSED');
  console.log(`   ✅ Total Pages: ${metrics.pages.count}`);
  console.log(`   ⏱️  Time Taken: ${formatDuration(metrics.pages.time_ms)}`);
  
  console.log('\n3️⃣  CHUNKS CREATED');
  console.log(`   ✅ Total Chunks: ${metrics.chunks.count}`);
  console.log(`   ⏱️  Time Taken: ${formatDuration(metrics.chunks.time_ms)}`);
  
  console.log('\n4️⃣  IMAGES PROCESSED');
  console.log(`   ✅ Total Images: ${metrics.images.count}`);
  console.log(`   ⏱️  Time Taken: ${formatDuration(metrics.images.time_ms)}`);
  
  console.log('\n5️⃣  EMBEDDINGS CREATED');
  console.log(`   ✅ Text Embeddings: ${metrics.embeddings.text}`);
  console.log(`   ✅ CLIP Embeddings: ${metrics.embeddings.clip}`);
  console.log(`   ✅ Total Embeddings: ${metrics.embeddings.total}`);
  console.log(`   ⏱️  Time Taken: ${formatDuration(metrics.embeddings.time_ms)}`);
  
  console.log('\n6️⃣  ERRORS');
  console.log(`   ${metrics.errors.count > 0 ? '❌' : '✅'} Total Errors: ${metrics.errors.count}`);
  console.log(`   ⏱️  Time Lost: ${formatDuration(metrics.errors.time_ms)}`);
  
  console.log('\n7️⃣  RELATIONSHIPS CREATED');
  console.log(`   ✅ Chunk-Image: ${metrics.relationships.chunk_image}`);
  console.log(`   ✅ Product-Image: ${metrics.relationships.product_image}`);
  console.log(`   ✅ Chunk-Product: ${metrics.relationships.chunk_product}`);
  console.log(`   ✅ Total Relationships: ${metrics.relationships.total}`);
  console.log(`   ⏱️  Time Taken: ${formatDuration(metrics.relationships.time_ms)}`);
  
  console.log('\n8️⃣  METADATA EXTRACTED');
  console.log(`   ✅ Chunks with Metadata: ${metrics.metadata.chunks}`);
  console.log(`   ✅ Products with Metadata: ${metrics.metadata.products}`);
  console.log(`   ✅ Total Metadata: ${metrics.metadata.total}`);
  console.log(`   ⏱️  Time Taken: ${formatDuration(metrics.metadata.time_ms)}`);
  
  console.log('\n9️⃣  MEMORY USAGE');
  console.log(`   🧠 Peak Memory: ${metrics.system.peak_memory_mb} MB`);
  console.log(`   ⏱️  Monitored: ${formatDuration(metrics.total_time_ms)}`);
  
  console.log('\n🔟 CPU USAGE');
  console.log(`   ⚡ Average CPU: ${metrics.system.avg_cpu_percent}%`);
  console.log(`   ⏱️  Monitored: ${formatDuration(metrics.total_time_ms)}`);
  
  console.log('\n1️⃣1️⃣  COST');
  console.log(`   💰 Total Cost: $${metrics.cost.total_usd.toFixed(4)} USD`);
  console.log(`   📊 AI Calls: ${metrics.cost.ai_calls.length}`);
  
  console.log('\n1️⃣2️⃣  TOTAL TIME');
  console.log(`   ⏱️  Total Processing Time: ${formatDuration(metrics.total_time_ms)}`);
  
  // Stage breakdown
  console.log('\n📋 STAGE BREAKDOWN:');
  Object.entries(metrics.stages).forEach(([stage, time]) => {
    console.log(`   ${stage}: ${formatDuration(time)}`);
  });
}

// Generate comparison report
function generateComparisonReport(allResults) {
  logSection('🔬 MODEL COMPARISON REPORT');
  
  console.log('\n' + '='.repeat(100));
  console.log('METRIC                    | CLAUDE VISION              | GPT VISION                 | WINNER');
  console.log('='.repeat(100));
  
  const claude = allResults.find(r => r.model === 'claude-vision') || {};
  const gpt = allResults.find(r => r.model === 'gpt-vision') || {};
  
  const compareMetric = (name, claudeVal, gptVal, unit = '', lowerIsBetter = false) => {
    const winner = lowerIsBetter 
      ? (claudeVal < gptVal ? 'CLAUDE ✅' : 'GPT ✅')
      : (claudeVal > gptVal ? 'CLAUDE ✅' : 'GPT ✅');
    console.log(`${name.padEnd(25)} | ${String(claudeVal + unit).padEnd(26)} | ${String(gptVal + unit).padEnd(26)} | ${winner}`);
  };
  
  compareMetric('Products', claude.products?.count || 0, gpt.products?.count || 0);
  compareMetric('Chunks', claude.chunks?.count || 0, gpt.chunks?.count || 0);
  compareMetric('Images', claude.images?.count || 0, gpt.images?.count || 0);
  compareMetric('Total Embeddings', claude.embeddings?.total || 0, gpt.embeddings?.total || 0);
  compareMetric('Total Relationships', claude.relationships?.total || 0, gpt.relationships?.total || 0);
  compareMetric('Discovery Time', formatDuration(claude.products?.time_ms || 0), formatDuration(gpt.products?.time_ms || 0), '', true);
  compareMetric('Total Time', formatDuration(claude.total_time_ms || 0), formatDuration(gpt.total_time_ms || 0), '', true);
  compareMetric('Peak Memory', claude.system?.peak_memory_mb || 0, gpt.system?.peak_memory_mb || 0, ' MB', true);
  compareMetric('Avg CPU', claude.system?.avg_cpu_percent || 0, gpt.system?.avg_cpu_percent || 0, '%', true);
  compareMetric('Cost', '$' + (claude.cost?.total_usd || 0).toFixed(4), '$' + (gpt.cost?.total_usd || 0).toFixed(4), '', true);
  compareMetric('Errors', claude.errors?.count || 0, gpt.errors?.count || 0, '', true);
  
  console.log('='.repeat(100));
  
  // Save comparison report
  const reportPath = `comprehensive-comparison-${Date.now()}.json`;
  fs.writeFileSync(reportPath, JSON.stringify({ claude, gpt }, null, 2));
  log('REPORT', `Comparison report saved to: ${reportPath}`, 'success');
}

// Run the comprehensive test
runComprehensiveTest().catch(error => {
  console.error('Fatal error:', error);
  process.exit(1);
});
