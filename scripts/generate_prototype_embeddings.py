#!/usr/bin/env python3
"""
Generate CLIP text embeddings for all prototype values in material_properties table.

This script:
1. Fetches all properties with prototype_descriptions
2. For each prototype value, generates a CLIP text embedding (512D)
3. Stores the embedding in text_embedding_512 column
"""

import asyncio
import sys
from pathlib import Path
from typing import List, Dict, Any
import json
import httpx
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.supabase_client import get_supabase_client
from datetime import datetime


async def generate_text_embedding_512d(text: str) -> List[float]:
    """Generate 512D text embedding using OpenAI."""
    openai_api_key = os.getenv('OPENAI_API_KEY')

    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {openai_api_key}"},
            json={
                "model": "text-embedding-3-small",
                "input": text[:8191],  # OpenAI limit
                "encoding_format": "float",
                "dimensions": 512  # Request 512D embeddings
            },
            timeout=30.0
        )

        if response.status_code == 200:
            data = response.json()
            return data["data"][0]["embedding"]
        else:
            raise Exception(f"OpenAI API error: {response.status_code} - {response.text}")


async def generate_prototype_embeddings():
    """Generate CLIP embeddings for all prototype values."""
    supabase = get_supabase_client()

    print("🚀 Generating CLIP embeddings for prototype values...")

    # Fetch all properties with prototypes
    result = supabase.client.table('material_properties').select(
        'id, property_key, name, prototype_descriptions'
    ).not_.is_('prototype_descriptions', 'null').execute()

    properties = result.data
    print(f"   Found {len(properties)} properties with prototypes")

    total_embeddings = 0

    for prop in properties:
        property_key = prop['property_key']
        property_name = prop['name']
        prototypes = prop['prototype_descriptions']

        print(f"\n📝 Processing: {property_name} ({property_key})")
        print(f"   Prototype values: {len(prototypes)}")

        # Collect all text variations for this property
        all_texts = []
        for prototype_value, variations in prototypes.items():
            # Include the prototype value itself
            all_texts.append(prototype_value)
            # Include all variations
            all_texts.extend(variations)

        print(f"   Total text variations: {len(all_texts)}")

        # Generate a single combined embedding representing all prototype values
        # This creates a semantic "center" for the property
        combined_text = ", ".join(all_texts[:50])  # Limit to first 50 to avoid token limits

        print(f"   Generating 512D embedding for combined text...")
        try:
            embedding = await generate_text_embedding_512d(combined_text)

            # Store embedding in database
            supabase.client.table('material_properties').update({
                'text_embedding_512': embedding,
                'prototype_updated_at': datetime.utcnow().isoformat()
            }).eq('id', prop['id']).execute()

            total_embeddings += 1
            print(f"   ✅ Embedding generated and stored (512D, {len(embedding)} dimensions)")
        except Exception as e:
            print(f"   ❌ Failed to generate embedding: {e}")

    print(f"\n✅ Complete!")
    print(f"   Total embeddings generated: {total_embeddings}/{len(properties)}")


if __name__ == "__main__":
    asyncio.run(generate_prototype_embeddings())

