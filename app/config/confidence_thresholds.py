"""
Confidence Threshold Configuration

Defines confidence thresholds for different AI tasks to enable smart routing,
fallback strategies, and escalation decisions.

Last Updated: 2025-10-27
"""

from typing import Dict, Optional
from decimal import Decimal


class ConfidenceThresholds:
    """
    Confidence thresholds for AI task routing and escalation.
    
    Thresholds are on a 0.0-1.0 scale where:
    - 0.0-0.5: Low confidence (requires escalation or fallback)
    - 0.5-0.7: Medium confidence (acceptable for non-critical tasks)
    - 0.7-0.85: Good confidence (acceptable for most tasks)
    - 0.85-1.0: High confidence (excellent quality)
    """
    
    # Material Classification Thresholds
    MATERIAL_CLASSIFICATION = {
        "minimum_acceptable": 0.70,  # Below this, use fallback
        "good": 0.80,                # Good quality result
        "excellent": 0.90,           # Excellent quality result
        "escalation_threshold": 0.70,  # Below this, escalate to better model
    }
    
    # Product Extraction Thresholds
    PRODUCT_EXTRACTION = {
        "minimum_acceptable": 0.75,  # Products must be high confidence
        "good": 0.85,
        "excellent": 0.92,
        "escalation_threshold": 0.75,
    }
    
    # Image Analysis Thresholds
    IMAGE_ANALYSIS = {
        "minimum_acceptable": 0.65,  # Images can be more subjective
        "good": 0.75,
        "excellent": 0.88,
        "escalation_threshold": 0.65,
    }
    
    # Text Chunking Quality Thresholds
    CHUNKING_QUALITY = {
        "minimum_acceptable": 0.70,
        "good": 0.80,
        "excellent": 0.90,
        "escalation_threshold": 0.70,
    }
    
    # Embedding Quality Thresholds
    EMBEDDING_QUALITY = {
        "minimum_acceptable": 0.75,
        "good": 0.85,
        "excellent": 0.93,
        "escalation_threshold": 0.75,
    }
    
    # Product Enrichment Thresholds
    PRODUCT_ENRICHMENT = {
        "minimum_acceptable": 0.72,
        "good": 0.82,
        "excellent": 0.91,
        "escalation_threshold": 0.72,
    }
    
    # RAG Search Thresholds
    RAG_SEARCH = {
        "minimum_acceptable": 0.68,
        "good": 0.78,
        "excellent": 0.88,
        "escalation_threshold": 0.68,
    }
    
    # Document Classification Thresholds
    DOCUMENT_CLASSIFICATION = {
        "minimum_acceptable": 0.73,
        "good": 0.83,
        "excellent": 0.92,
        "escalation_threshold": 0.73,
    }
    
    # Metadata Extraction Thresholds
    METADATA_EXTRACTION = {
        "minimum_acceptable": 0.71,
        "good": 0.81,
        "excellent": 0.90,
        "escalation_threshold": 0.71,
    }
    
    # Vision Analysis Thresholds
    VISION_ANALYSIS = {
        "minimum_acceptable": 0.67,
        "good": 0.77,
        "excellent": 0.87,
        "escalation_threshold": 0.67,
    }
    
    @classmethod
    def get_threshold(cls, task_type: str, threshold_level: str = "minimum_acceptable") -> float:
        """
        Get threshold for a specific task type and level.
        
        Args:
            task_type: Type of AI task (e.g., 'material_classification')
            threshold_level: Level of threshold ('minimum_acceptable', 'good', 'excellent', 'escalation_threshold')
            
        Returns:
            Threshold value (0.0-1.0)
        """
        task_type_upper = task_type.upper()
        
        if hasattr(cls, task_type_upper):
            thresholds = getattr(cls, task_type_upper)
            return thresholds.get(threshold_level, 0.70)  # Default to 0.70
        
        # Default thresholds if task type not found
        return {
            "minimum_acceptable": 0.70,
            "good": 0.80,
            "excellent": 0.90,
            "escalation_threshold": 0.70,
        }.get(threshold_level, 0.70)
    
    @classmethod
    def should_escalate(cls, task_type: str, confidence_score: float) -> bool:
        """
        Determine if a result should be escalated to a better model.
        
        Args:
            task_type: Type of AI task
            confidence_score: Confidence score of the result (0.0-1.0)
            
        Returns:
            True if should escalate, False otherwise
        """
        escalation_threshold = cls.get_threshold(task_type, "escalation_threshold")
        return confidence_score < escalation_threshold
    
    @classmethod
    def is_acceptable(cls, task_type: str, confidence_score: float) -> bool:
        """
        Determine if a result meets minimum acceptable quality.
        
        Args:
            task_type: Type of AI task
            confidence_score: Confidence score of the result (0.0-1.0)
            
        Returns:
            True if acceptable, False otherwise
        """
        minimum_threshold = cls.get_threshold(task_type, "minimum_acceptable")
        return confidence_score >= minimum_threshold
    
    @classmethod
    def get_quality_level(cls, task_type: str, confidence_score: float) -> str:
        """
        Get quality level description for a confidence score.
        
        Args:
            task_type: Type of AI task
            confidence_score: Confidence score of the result (0.0-1.0)
            
        Returns:
            Quality level: 'excellent', 'good', 'acceptable', 'poor'
        """
        excellent = cls.get_threshold(task_type, "excellent")
        good = cls.get_threshold(task_type, "good")
        minimum = cls.get_threshold(task_type, "minimum_acceptable")
        
        if confidence_score >= excellent:
            return "excellent"
        elif confidence_score >= good:
            return "good"
        elif confidence_score >= minimum:
            return "acceptable"
        else:
            return "poor"


class EscalationRules:
    """
    Rules for escalating AI tasks to more powerful (and expensive) models.
    """
    
    # Model escalation chain (cheapest to most expensive)
    MODEL_CHAIN = [
        "qwen3-vl-32b",           # Primary vision model (HF Endpoint)
        "claude-haiku-4-5",       # Balanced
        "claude-sonnet-4-5",      # Powerful
        "gpt-5",                  # Most powerful (for critical tasks)
    ]

    # Cost multipliers for each model (relative to Qwen3-VL-32B)
    COST_MULTIPLIERS = {
        "qwen3-vl-32b": 1.0,      # Baseline (32B only, 8B removed)
        "claude-haiku-4-5": 2.0,
        "claude-sonnet-4-5": 7.5,
        "gpt-5": 12.5,
    }
    
    # Maximum escalation attempts before giving up
    MAX_ESCALATION_ATTEMPTS = 2
    
    # Tasks that are critical and can use expensive models
    CRITICAL_TASKS = {
        "product_extraction",
        "material_classification",
        "safety_information",
        "compliance_data",
        "technical_specifications",
        "pricing_data",
    }
    
    # Tasks that should never escalate beyond Haiku (cost-sensitive)
    COST_SENSITIVE_TASKS = {
        "image_analysis",
        "chunking_quality",
        "rag_search",
    }
    
    @classmethod
    def get_next_model(cls, current_model: str, task_type: str) -> Optional[str]:
        """
        Get the next model in the escalation chain.
        
        Args:
            current_model: Current model being used
            task_type: Type of task being performed
            
        Returns:
            Next model to try, or None if no escalation available
        """
        # Normalize model name
        current_model_normalized = current_model.lower().replace("_", "-")
        
        # Find current model in chain
        try:
            current_index = cls.MODEL_CHAIN.index(current_model_normalized)
        except ValueError:
            # Model not in chain, start from beginning
            return cls.MODEL_CHAIN[0]
        
        # Check if we can escalate
        if current_index >= len(cls.MODEL_CHAIN) - 1:
            return None  # Already at top of chain
        
        # Check if task is cost-sensitive
        if task_type.lower() in cls.COST_SENSITIVE_TASKS:
            # Don't escalate beyond Haiku
            haiku_index = cls.MODEL_CHAIN.index("claude-haiku-4-5")
            if current_index >= haiku_index:
                return None
        
        # Return next model
        return cls.MODEL_CHAIN[current_index + 1]
    
    @classmethod
    def can_use_expensive_model(cls, task_type: str) -> bool:
        """
        Check if a task can use expensive models (GPT-5, Sonnet).
        
        Args:
            task_type: Type of task
            
        Returns:
            True if expensive models are allowed
        """
        return task_type.lower() in cls.CRITICAL_TASKS
    
    @classmethod
    def get_cost_multiplier(cls, model: str) -> float:
        """
        Get cost multiplier for a model relative to Qwen3-VL-8B.
        
        Args:
            model: Model name
            
        Returns:
            Cost multiplier
        """
        model_normalized = model.lower().replace("_", "-")
        return cls.COST_MULTIPLIERS.get(model_normalized, 1.0)


