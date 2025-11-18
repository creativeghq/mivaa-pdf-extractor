"""
Search Query Tracking Service

Tracks all search queries to identify:
1. Zero-result queries (users searching for things we don't have)
2. Unmatched terms (words that don't match any prototypes)
3. Patterns in user search behavior

This data feeds into the prototype discovery system to continuously improve
metadata validation and search relevance.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio

from app.core.supabase_client import get_supabase_client
from app.services.metadata_prototype_validator import get_metadata_validator

logger = logging.getLogger(__name__)


class SearchQueryTracker:
    """Tracks search queries and identifies missing prototypes."""
    
    def __init__(self):
        self.supabase = get_supabase_client()
        self.logger = logging.getLogger(__name__)
    
    async def track_query(
        self,
        workspace_id: str,
        query_text: str,
        query_metadata: Optional[Dict[str, Any]] = None,
        search_type: str = "multi_vector",
        result_count: int = 0,
        response_time_ms: int = 0,
        user_id: Optional[str] = None
    ):
        """Track a search query and analyze for missing prototypes.
        
        Args:
            workspace_id: Workspace performing the search
            query_text: Natural language query
            query_metadata: Metadata filters used (e.g., {"finish": "shiny"})
            search_type: Type of search performed
            result_count: Number of results returned
            response_time_ms: Query execution time
            user_id: Optional user ID
        """
        try:
            # Extract terms from query
            searched_terms = self._extract_search_terms(query_text, query_metadata)
            
            # Validate metadata terms against prototypes
            validation_results = {}
            matched_terms = []
            unmatched_terms = []
            
            if query_metadata:
                validator = get_metadata_validator()
                await validator.load_prototypes()
                
                for property_key, value in query_metadata.items():
                    if property_key in validator._prototype_cache:
                        # Validate this term
                        validated_value, validation_info = await validator._validate_field(
                            field_key=property_key,
                            field_value=str(value),
                            confidence_threshold=0.80
                        )
                        
                        validation_results[property_key] = validation_info
                        
                        if validation_info.get("prototype_matched"):
                            matched_terms.append(str(value))
                        else:
                            unmatched_terms.append(str(value))
                            # Track unmatched term for frequency analysis
                            await self._track_unmatched_term(
                                term=str(value),
                                property_key=property_key,
                                workspace_id=workspace_id
                            )
            
            # Insert tracking record
            await self.supabase.client.table('search_query_tracking').insert({
                'workspace_id': workspace_id,
                'user_id': user_id,
                'query_text': query_text,
                'query_metadata': query_metadata,
                'search_type': search_type,
                'result_count': result_count,
                'zero_results': result_count == 0,
                'searched_terms': searched_terms,
                'matched_terms': matched_terms,
                'unmatched_terms': unmatched_terms,
                'validation_attempted': len(validation_results) > 0,
                'validation_results': validation_results,
                'response_time_ms': response_time_ms,
                'timestamp': datetime.utcnow().isoformat()
            }).execute()
            
            # Log zero-result queries for immediate attention
            if result_count == 0:
                self.logger.warning(
                    f"Zero-result query: '{query_text}' with filters {query_metadata}. "
                    f"Unmatched terms: {unmatched_terms}"
                )
        
        except Exception as e:
            # Don't fail the search if tracking fails
            self.logger.error(f"Failed to track search query: {e}")
    
    async def _track_unmatched_term(
        self,
        term: str,
        property_key: str,
        workspace_id: str
    ):
        """Track an unmatched term for frequency analysis."""
        try:
            # Upsert into unmatched_term_frequency
            result = await self.supabase.client.rpc(
                'upsert_unmatched_term',
                {
                    'p_term': term,
                    'p_property_key': property_key,
                    'p_workspace_id': workspace_id
                }
            ).execute()
            
        except Exception as e:
            # If RPC doesn't exist, do manual upsert
            try:
                existing = await self.supabase.client.table('unmatched_term_frequency').select('*').eq(
                    'term', term
                ).eq('property_key', property_key).execute()
                
                if existing.data:
                    # Update frequency
                    await self.supabase.client.table('unmatched_term_frequency').update({
                        'frequency_count': existing.data[0]['frequency_count'] + 1,
                        'last_seen_at': datetime.utcnow().isoformat(),
                        'workspace_ids': list(set(existing.data[0].get('workspace_ids', []) + [workspace_id]))
                    }).eq('id', existing.data[0]['id']).execute()
                else:
                    # Insert new
                    await self.supabase.client.table('unmatched_term_frequency').insert({
                        'term': term,
                        'property_key': property_key,
                        'frequency_count': 1,
                        'workspace_ids': [workspace_id]
                    }).execute()
            except Exception as e2:
                self.logger.error(f"Failed to track unmatched term: {e2}")
    
    def _extract_search_terms(
        self,
        query_text: str,
        query_metadata: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Extract searchable terms from query."""
        terms = []
        
        # Add metadata values
        if query_metadata:
            for value in query_metadata.values():
                if isinstance(value, str):
                    terms.append(value.lower())
                elif isinstance(value, list):
                    terms.extend([str(v).lower() for v in value])
        
        # Add significant words from query text (simple tokenization)
        if query_text:
            words = query_text.lower().split()
            # Filter out common words
            stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'show', 'find', 'search'}
            terms.extend([w for w in words if w not in stopwords and len(w) > 2])
        
        return list(set(terms))  # Deduplicate


# Singleton instance
_tracker_instance: Optional[SearchQueryTracker] = None


def get_search_tracker() -> SearchQueryTracker:
    """Get singleton tracker instance."""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = SearchQueryTracker()
    return _tracker_instance

