"""
Tests for Prompt Enhancement Service

Tests the prompt enhancement pipeline including:
- Custom prompt retrieval
- Intent parsing
- Context enrichment
- Prompt building
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from app.services.prompt_enhancement_service import PromptEnhancementService, EnhancedPrompt


@pytest.fixture
def mock_supabase():
    """Mock Supabase client"""
    mock = Mock()
    mock.client = Mock()
    return mock


@pytest.fixture
def service(mock_supabase):
    """Create PromptEnhancementService with mocked Supabase"""
    with patch('app.services.prompt_enhancement_service.get_supabase_client', return_value=mock_supabase):
        return PromptEnhancementService()


@pytest.mark.asyncio
async def test_parse_agent_intent_extract_products():
    """Test parsing 'extract products' intent"""
    service = PromptEnhancementService()
    
    intent = await service.parse_agent_intent("extract products")
    
    assert intent['action'] == 'extract'
    assert intent['target'] == 'products'


@pytest.mark.asyncio
async def test_parse_agent_intent_search():
    """Test parsing search intent"""
    service = PromptEnhancementService()
    
    intent = await service.parse_agent_intent("search for NOVA")
    
    assert intent['action'] == 'search'
    assert intent['query'] == 'NOVA'


@pytest.mark.asyncio
async def test_parse_agent_intent_certificates():
    """Test parsing certificate extraction intent"""
    service = PromptEnhancementService()
    
    intent = await service.parse_agent_intent("extract certificates")
    
    assert intent['action'] == 'extract'
    assert intent['target'] == 'certificates'


@pytest.mark.asyncio
async def test_get_default_prompt():
    """Test getting default prompt template"""
    service = PromptEnhancementService()
    
    prompt = service.get_default_prompt('discovery', 'products')
    
    assert 'product' in prompt.lower()
    assert 'extract' in prompt.lower() or 'analyze' in prompt.lower()


@pytest.mark.asyncio
async def test_get_category_guidelines():
    """Test getting category-specific guidelines"""
    service = PromptEnhancementService()
    
    guidelines = service._get_category_guidelines('products')
    
    assert 'product' in guidelines.lower()
    assert len(guidelines) > 0


@pytest.mark.asyncio
async def test_enhance_prompt_with_defaults(service, mock_supabase):
    """Test prompt enhancement with default templates"""
    # Mock no custom prompts
    mock_result = Mock()
    mock_result.data = []
    mock_supabase.client.table.return_value.select.return_value.eq.return_value.eq.return_value.eq.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = mock_result
    
    # Mock extraction config
    mock_config = Mock()
    mock_config.data = [{
        'enabled_categories': ['products'],
        'quality_threshold': 0.7,
        'chunk_size': 1000
    }]
    mock_supabase.client.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value = mock_config
    
    enhanced = await service.enhance_prompt(
        agent_prompt="extract products",
        stage="discovery",
        category="products",
        workspace_id="test-workspace-id"
    )
    
    assert isinstance(enhanced, EnhancedPrompt)
    assert enhanced.original_prompt == "extract products"
    assert len(enhanced.enhanced_prompt) > 0
    assert enhanced.stage == "discovery"
    assert enhanced.category == "products"


@pytest.mark.asyncio
async def test_enhance_prompt_with_custom_template(service, mock_supabase):
    """Test prompt enhancement with custom template"""
    # Mock custom prompt
    mock_result = Mock()
    mock_result.data = [{
        'prompt_template': 'Custom prompt for {category}',
        'system_prompt': 'System instructions',
        'version': 2
    }]
    mock_supabase.client.table.return_value.select.return_value.eq.return_value.eq.return_value.eq.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = mock_result
    
    # Mock extraction config
    mock_config = Mock()
    mock_config.data = [{
        'enabled_categories': ['products'],
        'quality_threshold': 0.8,
        'chunk_size': 1000
    }]
    
    enhanced = await service.enhance_prompt(
        agent_prompt="extract products",
        stage="discovery",
        category="products",
        workspace_id="test-workspace-id"
    )
    
    assert enhanced.prompt_version == 2
    assert 'Custom prompt' in enhanced.template_used


@pytest.mark.asyncio
async def test_enhance_prompt_with_context(service, mock_supabase):
    """Test prompt enhancement with additional context"""
    # Mock no custom prompts
    mock_result = Mock()
    mock_result.data = []
    mock_supabase.client.table.return_value.select.return_value.eq.return_value.eq.return_value.eq.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = mock_result
    
    # Mock extraction config
    mock_config = Mock()
    mock_config.data = [{
        'enabled_categories': ['products'],
        'quality_threshold': 0.7
    }]
    
    context = {
        'document_context': {
            'filename': 'test.pdf',
            'page_count': 50
        },
        'previous_results': ['result1', 'result2']
    }
    
    enhanced = await service.enhance_prompt(
        agent_prompt="extract products",
        stage="discovery",
        category="products",
        workspace_id="test-workspace-id",
        context=context
    )
    
    assert 'test.pdf' in enhanced.enhanced_prompt or 'test.pdf' in str(enhanced.context_added)


@pytest.mark.asyncio
async def test_format_workspace_settings():
    """Test workspace settings formatting"""
    service = PromptEnhancementService()
    
    settings = {
        'quality_threshold': 0.8,
        'chunk_size': 1500,
        'enabled_categories': ['products', 'certificates']
    }
    
    formatted = service._format_workspace_settings(settings)
    
    assert '0.8' in formatted or 'quality_threshold' in formatted
    assert len(formatted) > 0


@pytest.mark.asyncio
async def test_build_enhanced_prompt():
    """Test building enhanced prompt from template"""
    service = PromptEnhancementService()
    
    template = "Extract {category} with quality threshold {quality_threshold}"
    context = {
        'quality_threshold': 0.9,
        'category_guidelines': 'Focus on products'
    }
    intent = {'action': 'extract', 'target': 'products'}
    
    enhanced = await service.build_enhanced_prompt(
        agent_prompt="extract products",
        template=template,
        context=context,
        intent=intent
    )
    
    assert '0.9' in enhanced
    assert len(enhanced) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

