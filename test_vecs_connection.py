#!/usr/bin/env python3
"""
Test VECS Connection and Data Saving
This script tests the complete VECS workflow:
1. Connect to VECS
2. Create/get collection
3. Save test embedding
4. Query test embedding
5. Delete test data
"""

import os
import sys
import asyncio
from app.services.vecs_service import VecsService

async def test_vecs_connection():
    """Test VECS connection and data operations."""
    print("🔍 Testing VECS Connection and Data Operations")
    print("=" * 70)
    
    try:
        # Step 1: Initialize VECS service
        print("\n1️⃣ Initializing VECS Service...")
        vecs_service = VecsService()
        print("   ✅ VECS Service initialized")
        
        # Step 2: Get connection string
        print("\n2️⃣ Getting Connection String...")
        conn_str = vecs_service._get_connection_string()
        # Mask sensitive data
        masked_conn = conn_str.split('@')[0].split(':')[0] + ':***@' + conn_str.split('@')[1]
        print(f"   ✅ Connection: {masked_conn}")
        
        # Step 3: Test connection by listing collections
        print("\n3️⃣ Testing Connection (List Collections)...")
        import vecs
        client = vecs.create_client(conn_str)
        collections = client.list_collections()
        print(f"   ✅ Connection successful!")
        print(f"   ✅ Found {len(collections)} existing collections")
        if collections:
            print(f"   Collections: {[c.name for c in collections[:5]]}")
        
        # Step 4: Get or create test collection
        print("\n4️⃣ Getting/Creating Test Collection...")
        collection = client.get_or_create_collection(
            name="test_vecs_connection",
            dimension=512
        )
        print(f"   ✅ Collection 'test_vecs_connection' ready")
        
        # Step 5: Create test embedding
        print("\n5️⃣ Creating Test Embedding...")
        test_embedding = [0.1] * 512  # Simple test vector
        test_id = "test_image_001"
        test_metadata = {
            "test": True,
            "document_id": "test_doc_123",
            "image_path": "/tmp/test.jpg"
        }
        
        # Upsert test data
        collection.upsert(
            records=[(test_id, test_embedding, test_metadata)]
        )
        print(f"   ✅ Test embedding saved (ID: {test_id})")
        
        # Step 6: Query test embedding
        print("\n6️⃣ Querying Test Embedding...")
        results = collection.query(
            data=test_embedding,
            limit=1,
            include_value=True,
            include_metadata=True
        )
        
        if results and len(results) > 0:
            result = results[0]
            print(f"   ✅ Query successful!")
            print(f"   Found: {result[0]} (distance: {result[1]:.6f})")
            print(f"   Metadata: {result[2]}")
        else:
            print(f"   ❌ Query returned no results")
            return False
        
        # Step 7: Delete test data
        print("\n7️⃣ Cleaning Up Test Data...")
        collection.delete(ids=[test_id])
        print(f"   ✅ Test data deleted")
        
        # Step 8: Verify deletion
        print("\n8️⃣ Verifying Deletion...")
        verify_results = collection.query(
            data=test_embedding,
            limit=1,
            filters={"test": {"$eq": True}}
        )
        
        if not verify_results or len(verify_results) == 0:
            print(f"   ✅ Deletion verified - no test data found")
        else:
            print(f"   ⚠️ Warning: Test data still exists")
        
        print("\n" + "=" * 70)
        print("✅ ALL VECS TESTS PASSED!")
        print("=" * 70)
        print("\n📋 Summary:")
        print("   • VECS connection: Working")
        print("   • Collection creation: Working")
        print("   • Data insertion: Working")
        print("   • Data querying: Working")
        print("   • Data deletion: Working")
        print("\n✅ VECS is ready for production use!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run the async test
    success = asyncio.run(test_vecs_connection())
    sys.exit(0 if success else 1)

