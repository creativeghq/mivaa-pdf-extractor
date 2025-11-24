/**
 * Complete Database Reset Script
 *
 * This script:
 * 1. Deletes knowledge base data (chunks, embeddings, products, images, etc.)
 * 2. PRESERVES user data (users, profiles, workspaces, API keys)
 * 3. Deletes all files from storage buckets EXCEPT pdf-documents folder
 * 4. Verifies cleanup was successful
 * 5. Reports storage and resource usage
 */

const SUPABASE_URL = 'https://bgbavxtjlbvgplozizxu.supabase.co';
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;

if (!SUPABASE_SERVICE_ROLE_KEY) {
  console.error('❌ SUPABASE_SERVICE_ROLE_KEY environment variable is required');
  process.exit(1);
}

const TABLES_TO_CLEAR = [
  'embeddings',
  'document_images',
  'document_chunks',
  'products',
  'background_jobs',
  'documents',
  'ai_analysis_queue',
  'processed_documents',
  'job_progress',
  'materials_catalog',
  'material_visual_analysis',
  'processing_results',
  'quality_metrics_daily',
  'quality_scoring_logs',
  'analytics_events',
  'agent_tasks',
  'generation_3d',
  'scraped_materials_temp',
  'scraping_sessions',
  'scraping_pages'
];

const BUCKETS_CONFIG = [
  { name: 'pdf-tiles', excludeFolders: [] },
  { name: 'pdf-documents', excludeFolders: ['*'] },
  { name: 'material-images', excludeFolders: [] }
];

async function makeSupabaseRequest(method, path, body = null) {
  const url = `${SUPABASE_URL}${path}`;
  const options = {
    method,
    headers: {
      'Authorization': `Bearer ${SUPABASE_SERVICE_ROLE_KEY}`,
      'apikey': SUPABASE_SERVICE_ROLE_KEY,
      'Prefer': 'return=minimal'
    }
  };

  if (body) {
    options.headers['Content-Type'] = 'application/json';
    options.body = JSON.stringify(body);
  }

  const response = await fetch(url, options);

  if (method === 'DELETE' && (response.status === 200 || response.status === 204)) {
    return { success: true };
  }

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Supabase API error ${response.status}: ${text}`);
  }

  const text = await response.text();
  if (!text) {
    return { success: true };
  }

  try {
    return JSON.parse(text);
  } catch (e) {
    return { success: true, raw: text };
  }
}

async function clearTable(tableName) {
  console.log(`\n🗑️  Clearing table: ${tableName}`);

  try {
    const countBefore = await makeSupabaseRequest('GET', `/rest/v1/${tableName}?select=count`, null);
    const count = countBefore?.[0]?.count || 0;

    if (count === 0) {
      console.log(`   ✅ Table ${tableName} is already empty`);
      return { table: tableName, deleted: 0 };
    }

    console.log(`   📊 Found ${count} rows to delete`);
    await makeSupabaseRequest('DELETE', `/rest/v1/${tableName}?id=neq.00000000-0000-0000-0000-000000000000`, null);
    console.log(`   ✅ Deleted ${count} rows from ${tableName}`);
    return { table: tableName, deleted: count };
  } catch (error) {
    console.error(`   ❌ Failed to clear ${tableName}: ${error.message}`);
    return { table: tableName, deleted: 0, error: error.message };
  }
}

async function listBucketFiles(bucketName, path = '') {
  try {
    const response = await makeSupabaseRequest('POST', `/storage/v1/object/list/${bucketName}`, {
      prefix: path,
      limit: 1000,
      offset: 0
    });
    return response || [];
  } catch (error) {
    console.error(`   ❌ Failed to list files in ${bucketName}/${path}: ${error.message}`);
    return [];
  }
}

async function deleteFile(bucketName, filePath) {
  try {
    await makeSupabaseRequest('DELETE', `/storage/v1/object/${bucketName}`, {
      prefixes: [filePath]
    });
    return true;
  } catch (error) {
    console.error(`   ⚠️  Failed to delete ${bucketName}/${filePath}: ${error.message}`);
    return false;
  }
}

async function listAllFilesRecursively(bucketName, prefix = '') {
  const allFiles = [];

  async function listFolder(folderPath) {
    const items = await listBucketFiles(bucketName, folderPath);

    for (const item of items) {
      const fullPath = folderPath ? `${folderPath}/${item.name}` : item.name;

      if (!item.metadata || item.metadata.mimetype === 'application/x-directory') {
        await listFolder(fullPath);
      } else {
        allFiles.push(fullPath);
      }
    }
  }

  await listFolder(prefix);
  return allFiles;
}

async function clearBucket(bucketConfig) {
  const { name: bucketName, excludeFolders = [] } = bucketConfig;
  console.log(`\n🗑️  Clearing bucket: ${bucketName}`);

  if (excludeFolders.length > 0) {
    console.log(`   🔒 Preserving folders: ${excludeFolders.join(', ')}`);
  }

  try {
    const allFiles = await listAllFilesRecursively(bucketName);

    if (allFiles.length === 0) {
      console.log(`   ✅ Bucket ${bucketName} is already empty`);
      return { bucket: bucketName, deleted: 0, skipped: 0 };
    }

    console.log(`   📊 Found ${allFiles.length} files to process`);

    let deleted = 0;
    let failed = 0;
    let skipped = 0;

    for (const filePath of allFiles) {
      let shouldSkip = false;
      for (const excludedFolder of excludeFolders) {
        if (excludedFolder === '*' || filePath.startsWith(excludedFolder)) {
          shouldSkip = true;
          skipped++;
          break;
        }
      }

      if (shouldSkip) {
        if (skipped % 10 === 0 && skipped > 0) {
          console.log(`   🔒 Skipped ${skipped} files in preserved folders...`);
        }
        continue;
      }

      const success = await deleteFile(bucketName, filePath);
      if (success) {
        deleted++;
        if (deleted % 10 === 0) {
          console.log(`   🔄 Deleted ${deleted} files...`);
        }
      } else {
        failed++;
      }
    }

    console.log(`   ✅ Deleted ${deleted} files from ${bucketName}`);
    if (skipped > 0) {
      console.log(`   🔒 Preserved ${skipped} files in excluded folders`);
    }
    if (failed > 0) {
      console.log(`   ⚠️  Failed to delete ${failed} files`);
    }

    return { bucket: bucketName, deleted, failed, skipped };
  } catch (error) {
    console.error(`   ❌ Failed to clear bucket ${bucketName}: ${error.message}`);
    return { bucket: bucketName, deleted: 0, failed: 0, skipped: 0, error: error.message };
  }
}

async function getStorageUsage() {
  try {
    console.log('\n📊 Checking storage usage...');

    const bucketStats = [];
    for (const bucketConfig of BUCKETS_CONFIG) {
      const files = await listBucketFiles(bucketConfig.name);
      let totalSize = 0;
      let fileCount = 0;

      for (const file of files) {
        if (!file.name.endsWith('/')) {
          totalSize += file.metadata?.size || 0;
          fileCount++;
        }
      }

      bucketStats.push({
        bucket: bucketConfig.name,
        files: fileCount,
        size_mb: (totalSize / 1024 / 1024).toFixed(2)
      });
    }

    return bucketStats;
  } catch (error) {
    console.error(`❌ Failed to get storage usage: ${error.message}`);
    return [];
  }
}

async function main() {
  console.log('═══════════════════════════════════════════════════════════');
  console.log('🔄 KNOWLEDGE BASE RESET (PRESERVING USER DATA)');
  console.log('═══════════════════════════════════════════════════════════');
  console.log('');
  console.log('✅ PRESERVED:');
  console.log('   • Users & Authentication');
  console.log('   • Profiles & Workspaces');
  console.log('   • API Keys & Usage Logs');
  console.log('   • PDF files in pdf-documents folder');
  console.log('');
  console.log('🗑️  WILL DELETE:');
  console.log('   • PDF Processing Data (chunks, embeddings, images)');
  console.log('   • Products & Materials Catalog');
  console.log('   • Background Jobs & Processing Results');
  console.log('   • Analytics & Agent Tasks');
  console.log('   • 3D Generation History');
  console.log('   • Storage files (except pdf-documents folder)');
  console.log('');
  console.log('═══════════════════════════════════════════════════════════');
  console.log(`📅 Started: ${new Date().toISOString()}`);

  const results = {
    tables: [],
    buckets: [],
    storage_before: [],
    storage_after: []
  };

  console.log('\n📊 STEP 1: Check storage usage BEFORE cleanup');
  results.storage_before = await getStorageUsage();
  console.log('\nStorage usage BEFORE:');
  console.table(results.storage_before);

  console.log('\n🗑️  STEP 2: Clear knowledge base tables');
  console.log(`   📋 Clearing ${TABLES_TO_CLEAR.length} tables (preserving user data)...`);
  for (const tableName of TABLES_TO_CLEAR) {
    const result = await clearTable(tableName);
    results.tables.push(result);
  }

  console.log('\n🗑️  STEP 3: Clear storage buckets');
  console.log('   🔒 Preserving pdf-documents folder and all files inside...');
  for (const bucketConfig of BUCKETS_CONFIG) {
    const result = await clearBucket(bucketConfig);
    results.buckets.push(result);
  }

  console.log('\n📊 STEP 4: Check storage usage AFTER cleanup');
  results.storage_after = await getStorageUsage();
  console.log('\nStorage usage AFTER:');
  console.table(results.storage_after);

  console.log('\n═══════════════════════════════════════════════════════════');
  console.log('📊 CLEANUP SUMMARY');
  console.log('═══════════════════════════════════════════════════════════');

  const totalRowsDeleted = results.tables.reduce((sum, r) => sum + (r.deleted || 0), 0);
  const totalFilesDeleted = results.buckets.reduce((sum, r) => sum + (r.deleted || 0), 0);
  const totalFilesFailed = results.buckets.reduce((sum, r) => sum + (r.failed || 0), 0);
  const totalFilesSkipped = results.buckets.reduce((sum, r) => sum + (r.skipped || 0), 0);

  console.log(`\n✅ Database rows deleted: ${totalRowsDeleted}`);
  console.log(`✅ Storage files deleted: ${totalFilesDeleted}`);
  if (totalFilesSkipped > 0) {
    console.log(`🔒 Storage files preserved: ${totalFilesSkipped} (pdf-documents folder)`);
  }
  if (totalFilesFailed > 0) {
    console.log(`⚠️  Storage files failed: ${totalFilesFailed}`);
  }

  console.log('\n📋 Table cleanup details:');
  console.table(results.tables);

  console.log('\n📦 Bucket cleanup details:');
  console.table(results.buckets);

  console.log('\n✅ PRESERVED DATA:');
  console.log('   • Users, Profiles, Workspaces remain intact');
  console.log('   • API Keys and authentication preserved');
  console.log(`   • ${totalFilesSkipped} files preserved in pdf-documents folder`);

  console.log(`\n📅 Completed: ${new Date().toISOString()}`);
  console.log('═══════════════════════════════════════════════════════════');
}

main().catch(error => {
  console.error('\n❌ FATAL ERROR:', error);
  process.exit(1);
});
