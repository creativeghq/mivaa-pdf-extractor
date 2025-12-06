"""
Prompt Enhancement Service

Enhances agent prompts with context, admin customizations, and detailed instructions.
Transforms simple prompts like "extract products" into detailed extraction prompts.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass

from app.services.supabase_client import get_supabase_client
from app.services.unified_prompt_service import UnifiedPromptService

logger = logging.getLogger(__name__)


@dataclass
class EnhancedPrompt:
    """Enhanced prompt with metadata"""
    original_prompt: str
    enhanced_prompt: str
    template_used: str
    context_added: Dict[str, Any]
    confidence_score: float
    stage: str
    category: str
    prompt_version: int


class PromptEnhancementService:
    """Service for enhancing agent prompts with context and customizations"""

    def __init__(self):
        self.supabase = get_supabase_client()
        self.prompt_service = UnifiedPromptService()
        
    async def enhance_prompt(
        self,
        agent_prompt: str,
        stage: str,
        category: str,
        workspace_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> EnhancedPrompt:
        """
        Enhance agent prompt with context and admin customizations
        
        Args:
            agent_prompt: Simple prompt from agent (e.g., "extract products")
            stage: Extraction stage (discovery, chunking, image_analysis, entity_creation)
            category: Content category (products, certificates, logos, specifications)
            workspace_id: Workspace ID for custom prompts
            context: Additional context (document_context, previous_results, etc.)
            
        Returns:
            EnhancedPrompt with original, enhanced prompt, and metadata
        """
        try:
            logger.info(f"Enhancing prompt for stage={stage}, category={category}")
            
            # 1. Get custom prompt from database
            custom_prompt = await self.get_custom_prompt(workspace_id, stage, category)
            
            # 2. Parse agent intent
            intent = await self.parse_agent_intent(agent_prompt)
            
            # 3. Build enhancement context
            enhancement_context = await self._build_enhancement_context(
                workspace_id, category, context or {}
            )
            
            # 4. Get template (custom or default)
            template = custom_prompt['template'] if custom_prompt else self.get_default_prompt(stage, category)
            prompt_version = custom_prompt['version'] if custom_prompt else 1
            
            # 5. Build enhanced prompt
            enhanced = await self.build_enhanced_prompt(
                agent_prompt,
                template,
                enhancement_context,
                intent
            )
            
            return EnhancedPrompt(
                original_prompt=agent_prompt,
                enhanced_prompt=enhanced,
                template_used=template,
                context_added=enhancement_context,
                confidence_score=0.9,  # Can be calculated based on context quality
                stage=stage,
                category=category,
                prompt_version=prompt_version
            )
            
        except Exception as e:
            logger.error(f"Error enhancing prompt: {str(e)}")
            # Fallback to default prompt
            return EnhancedPrompt(
                original_prompt=agent_prompt,
                enhanced_prompt=self.get_default_prompt(stage, category),
                template_used="default",
                context_added={},
                confidence_score=0.5,
                stage=stage,
                category=category,
                prompt_version=1
            )
    
    async def get_custom_prompt(
        self,
        workspace_id: str,
        stage: str,
        category: str
    ) -> Optional[Dict[str, Any]]:
        """Get prompt from unified prompts table"""
        try:
            # Use unified prompt service to get extraction prompt
            prompt_data = await self.prompt_service.get_extraction_prompt(
                workspace_id=workspace_id,
                stage=stage,
                category=category
            )

            if prompt_data:
                return {
                    'template': prompt_data.get('prompt_text') or prompt_data.get('system_prompt'),
                    'version': prompt_data.get('version', 1),
                    'is_custom': prompt_data.get('is_custom', False)
                }

            return None

        except Exception as e:
            logger.error(f"Error fetching custom prompt: {str(e)}")
            return None
    
    def get_default_prompt(self, stage: str, category: str) -> str:
        """Get default prompt template"""
        from app.services.prompt_templates import DEFAULT_PROMPTS
        
        key = f"{stage}_{category}"
        return DEFAULT_PROMPTS.get(key, DEFAULT_PROMPTS.get(f"{stage}_default", "Extract content from this document."))
    
    async def parse_agent_intent(self, agent_prompt: str) -> Dict[str, Any]:
        """
        Parse agent intent from natural language
        
        Examples:
        - "extract products" -> {action: "extract", target: "products"}
        - "search for NOVA" -> {action: "search", query: "NOVA"}
        """
        prompt_lower = agent_prompt.lower()
        
        intent = {
            'action': 'extract',  # Default action
            'target': None,
            'query': None,
            'filters': []
        }
        
        # Detect action
        if 'search' in prompt_lower or 'find' in prompt_lower:
            intent['action'] = 'search'
        elif 'extract' in prompt_lower or 'get' in prompt_lower:
            intent['action'] = 'extract'
        elif 'analyze' in prompt_lower:
            intent['action'] = 'analyze'
        
        # Detect target
        if 'product' in prompt_lower:
            intent['target'] = 'products'
        elif 'certificate' in prompt_lower:
            intent['target'] = 'certificates'
        elif 'logo' in prompt_lower:
            intent['target'] = 'logos'
        elif 'specification' in prompt_lower or 'spec' in prompt_lower:
            intent['target'] = 'specifications'
        
        # Extract search query (text after "for" or quoted text)
        if intent['action'] == 'search':
            if ' for ' in prompt_lower:
                intent['query'] = agent_prompt.split(' for ', 1)[1].strip()
            elif '"' in agent_prompt:
                # Extract quoted text
                import re
                matches = re.findall(r'"([^"]*)"', agent_prompt)
                if matches:
                    intent['query'] = matches[0]
        
        return intent
    
    async def build_enhanced_prompt(
        self,
        agent_prompt: str,
        template: str,
        context: Dict[str, Any],
        intent: Dict[str, Any]
    ) -> str:
        """Build final enhanced prompt by combining template with context"""
        
        # Replace template variables
        enhanced = template
        
        # Add context variables
        if '{workspace_settings}' in enhanced:
            settings_str = self._format_workspace_settings(context.get('workspace_settings', {}))
            enhanced = enhanced.replace('{workspace_settings}', settings_str)
        
        if '{quality_threshold}' in enhanced:
            threshold = context.get('quality_threshold', 0.7)
            enhanced = enhanced.replace('{quality_threshold}', str(threshold))
        
        if '{category_guidelines}' in enhanced:
            guidelines = context.get('category_guidelines', '')
            enhanced = enhanced.replace('{category_guidelines}', guidelines)
        
        if '{document_context}' in enhanced:
            doc_context = context.get('document_context', {})
            doc_str = f"Document: {doc_context.get('filename', 'Unknown')}, Pages: {doc_context.get('page_count', 'Unknown')}"
            enhanced = enhanced.replace('{document_context}', doc_str)
        
        # Add intent-specific instructions
        if intent.get('query'):
            enhanced += f"\n\nSpecific search query: {intent['query']}"
        
        return enhanced
    
    async def _build_enhancement_context(
        self,
        workspace_id: str,
        category: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build enhancement context from workspace settings and provided context"""
        
        # Get workspace extraction config
        config = await self._get_extraction_config(workspace_id)
        
        return {
            'workspace_settings': config,
            'quality_threshold': config.get('quality_threshold', 0.7),
            'category_guidelines': self._get_category_guidelines(category),
            'document_context': context.get('document_context', {}),
            'previous_results': context.get('previous_results', [])
        }
    
    async def _get_extraction_config(self, workspace_id: str) -> Dict[str, Any]:
        """Get extraction configuration for workspace"""
        try:
            result = self.supabase.client.table('extraction_config')\
                .select('*')\
                .eq('workspace_id', workspace_id)\
                .limit(1)\
                .execute()
            
            if result.data and len(result.data) > 0:
                return result.data[0]
            
            # Return defaults
            return {
                'enabled_categories': ['products'],
                'quality_threshold': 0.7,
                'chunk_size': 1000,
                'chunk_overlap': 200
            }
            
        except Exception as e:
            logger.error(f"Error fetching extraction config: {str(e)}")
            return {}
    
    def _get_category_guidelines(self, category: str) -> str:
        """Get category-specific guidelines"""
        guidelines = {
            'products': "Focus on product names, variants, dimensions, materials, finishes, designers, and specifications.",
            'certificates': "Extract certification names, types, standards, validity periods, and issuing authorities.",
            'logos': "Identify brand logos, manufacturer marks, certification logos, and quality marks.",
            'specifications': "Extract technical specifications, dimensions, materials, performance data, and compliance information."
        }
        return guidelines.get(category, "Extract relevant information from the document.")
    
    def _format_workspace_settings(self, settings: Dict[str, Any]) -> str:
        """Format workspace settings for prompt"""
        if not settings:
            return "No specific workspace settings"
        
        formatted = []
        for key, value in settings.items():
            if key in ['quality_threshold', 'chunk_size', 'enabled_categories']:
                formatted.append(f"{key}: {value}")
        
        return ", ".join(formatted) if formatted else "Default settings"

