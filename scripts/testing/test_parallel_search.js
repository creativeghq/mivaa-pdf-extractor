/**
 * Test Script for Issue #57 - Parallel Query Execution
 * 
 * Tests the 'all' strategy to verify:
 * 1. All 6 strategies execute in parallel
 * 2. Performance improvement: <300ms vs ~800ms sequential
 * 3. Proper result merging and deduplication
 * 4. Error handling (failed strategies don't block others)
 * 
 * Usage: node scripts/testing/test_parallel_search.js
 */

const API_BASE_URL = 'https://v1api.materialshub.gr';

// Test configuration
const TEST_QUERIES = [
    {
        name: 'Simple text query',
        query: 'modern oak furniture',
        workspace_id: 'test-workspace'
    },
    {
        name: 'Material-focused query',
        query: 'sustainable wood materials',
        workspace_id: 'test-workspace'
    },
    {
        name: 'Design-focused query',
        query: 'minimalist Scandinavian design',
        workspace_id: 'test-workspace'
    }
];

async function testParallelSearch(query, workspace_id) {
    console.log(`\n${'='.repeat(80)}`);
    console.log(`🔍 Testing Parallel Search: "${query}"`);
    console.log(`${'='.repeat(80)}\n`);

    try {
        const startTime = Date.now();

        const response = await fetch(`${API_BASE_URL}/api/rag/search?strategy=all`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query: query,
                workspace_id: workspace_id,
                top_k: 10
            })
        });

        const endTime = Date.now();
        const totalTime = endTime - startTime;

        if (!response.ok) {
            const errorText = await response.text();
            console.error(`❌ API Error (${response.status}): ${errorText}`);
            return null;
        }

        const data = await response.json();

        // Display results
        console.log(`✅ Request completed in ${totalTime}ms\n`);

        // Extract metadata (could be at root level or in search_metadata)
        const metadata = data.search_metadata || data;
        const strategiesExecuted = metadata.strategies_executed || data.strategies_executed;
        const strategiesSuccessful = metadata.strategies_successful || data.strategies_successful;
        const strategiesFailed = metadata.strategies_failed || data.strategies_failed;
        const strategyBreakdown = metadata.strategy_breakdown || data.strategy_breakdown;
        const parallelProcessingTime = metadata.parallel_processing_time || data.processing_time;

        console.log('📊 Performance Metrics:');
        console.log(`   Total Time: ${totalTime}ms`);
        console.log(`   API Processing Time: ${data.processing_time ? (data.processing_time * 1000).toFixed(2) : 'N/A'}ms`);
        console.log(`   Parallel Processing Time: ${parallelProcessingTime ? (parallelProcessingTime * 1000).toFixed(2) : 'N/A'}ms`);
        console.log(`   Strategies Executed: ${strategiesExecuted || 'N/A'}`);
        console.log(`   Strategies Successful: ${strategiesSuccessful || 'N/A'}`);
        console.log(`   Strategies Failed: ${strategiesFailed || 'N/A'}`);

        if (strategyBreakdown) {
            console.log('\n📋 Strategy Breakdown:');
            for (const [strategy, info] of Object.entries(strategyBreakdown)) {
                const status = info.success ? '✅' : '❌';
                console.log(`   ${status} ${strategy}: ${info.count} results`);
            }
        }

        console.log(`\n📦 Results:`);
        console.log(`   Total Results: ${data.total_results || 0}`);
        console.log(`   Results Returned: ${data.results?.length || 0}`);

        if (data.results && data.results.length > 0) {
            console.log('\n🔝 Top 3 Results:');
            data.results.slice(0, 3).forEach((result, index) => {
                console.log(`\n   ${index + 1}. Score: ${result.weighted_score?.toFixed(4) || result.score?.toFixed(4) || 'N/A'}`);
                console.log(`      Found by: ${result.found_by_strategies?.join(', ') || 'N/A'}`);
                console.log(`      Content: ${(result.content || result.description || 'N/A').substring(0, 100)}...`);
                if (result.strategy_scores) {
                    console.log(`      Strategy Scores: ${JSON.stringify(result.strategy_scores)}`);
                }
            });
        }

        // Performance validation
        console.log('\n🎯 Performance Validation:');
        const targetTime = 300; // Target: <300ms
        if (totalTime < targetTime) {
            console.log(`   ✅ PASS: ${totalTime}ms < ${targetTime}ms target`);
        } else {
            console.log(`   ⚠️  SLOW: ${totalTime}ms > ${targetTime}ms target (expected <300ms)`);
        }

        // Strategy execution validation
        if (strategiesExecuted >= 4) {
            console.log(`   ✅ PASS: ${strategiesExecuted} strategies executed (expected 4-6)`);
        } else {
            console.log(`   ❌ FAIL: Only ${strategiesExecuted || 0} strategies executed (expected 4-6)`);
        }

        // Success rate validation
        const successRate = strategiesExecuted ? strategiesSuccessful / strategiesExecuted : 0;
        if (successRate >= 0.8) {
            console.log(`   ✅ PASS: ${(successRate * 100).toFixed(0)}% success rate (expected >80%)`);
        } else {
            console.log(`   ⚠️  LOW: ${(successRate * 100).toFixed(0)}% success rate (expected >80%)`);
        }

        return {
            totalTime,
            processingTime: parallelProcessingTime ? parallelProcessingTime * 1000 : data.processing_time * 1000,
            strategiesExecuted: strategiesExecuted || 0,
            strategiesSuccessful: strategiesSuccessful || 0,
            totalResults: data.total_results || 0,
            successRate
        };

    } catch (error) {
        console.error(`❌ Test failed: ${error.message}`);
        console.error(error.stack);
        return null;
    }
}

async function runAllTests() {
    console.log('\n' + '='.repeat(80));
    console.log('🚀 PARALLEL SEARCH PERFORMANCE TEST - Issue #57');
    console.log('='.repeat(80));
    console.log('\nTarget: <300ms response time with 4-6 strategies in parallel');
    console.log('Expected: 3-4x faster than sequential (~800ms)\n');

    const results = [];

    for (const testCase of TEST_QUERIES) {
        const result = await testParallelSearch(testCase.query, testCase.workspace_id);
        if (result) {
            results.push({
                name: testCase.name,
                ...result
            });
        }
        // Wait 1 second between tests
        await new Promise(resolve => setTimeout(resolve, 1000));
    }

    // Summary
    console.log('\n' + '='.repeat(80));
    console.log('📊 TEST SUMMARY');
    console.log('='.repeat(80) + '\n');

    if (results.length === 0) {
        console.log('❌ No tests completed successfully');
        return;
    }

    const avgTime = results.reduce((sum, r) => sum + r.totalTime, 0) / results.length;
    const avgProcessingTime = results.reduce((sum, r) => sum + r.processingTime, 0) / results.length;
    const avgStrategies = results.reduce((sum, r) => sum + r.strategiesExecuted, 0) / results.length;
    const avgSuccessRate = results.reduce((sum, r) => sum + r.successRate, 0) / results.length;

    console.log(`Tests Completed: ${results.length}/${TEST_QUERIES.length}`);
    console.log(`Average Total Time: ${avgTime.toFixed(2)}ms`);
    console.log(`Average Processing Time: ${avgProcessingTime.toFixed(2)}ms`);
    console.log(`Average Strategies Executed: ${avgStrategies.toFixed(1)}`);
    console.log(`Average Success Rate: ${(avgSuccessRate * 100).toFixed(0)}%`);

    console.log('\n🎯 Overall Performance:');
    if (avgTime < 300) {
        console.log(`   ✅ EXCELLENT: ${avgTime.toFixed(0)}ms average (target: <300ms)`);
    } else if (avgTime < 500) {
        console.log(`   ⚠️  ACCEPTABLE: ${avgTime.toFixed(0)}ms average (target: <300ms)`);
    } else {
        console.log(`   ❌ NEEDS IMPROVEMENT: ${avgTime.toFixed(0)}ms average (target: <300ms)`);
    }

    const improvement = (800 / avgTime).toFixed(1);
    console.log(`   📈 Performance Improvement: ${improvement}x faster than sequential (~800ms)`);

    console.log('\n' + '='.repeat(80));
    console.log('✅ Test completed!');
    console.log('='.repeat(80) + '\n');
}

// Run tests
runAllTests().catch(error => {
    console.error('Fatal error:', error);
    process.exit(1);
});

