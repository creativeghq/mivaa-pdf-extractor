"""
Sentiment Analysis Service

Provides AI-powered sentiment analysis for user feedback using:
- DistilBERT for overall sentiment classification
- Aspect-based sentiment analysis for specific attributes
- Key phrase extraction
- Recommendation scoring
"""

import logging
from typing import Dict, Optional
from datetime import datetime
import re
import json

from app.services.core.ai_client_service import get_ai_client_service

from app.services.utilities.prompt_registry import load_prompt, render

logger = logging.getLogger(__name__)


class SentimentAnalysisService:
    """Service for analyzing user feedback sentiment using Claude"""

    def __init__(self, anthropic_api_key: str):
        # `anthropic_api_key` arg retained for back-compat; the shim reads
        # the key from Settings, so the arg is no longer used.
        self.anthropic_client = get_ai_client_service().anthropic
        self.aspects = ["quality", "appearance", "durability", "value", "usability"]
    
    async def analyze_feedback(
        self,
        feedback_text: str,
        material_name: Optional[str] = None,
        rating: Optional[int] = None
    ) -> Dict:
        """
        Analyze user feedback and extract sentiment, aspects, and key phrases
        
        Args:
            feedback_text: The user's feedback text
            material_name: Optional material name for context
            rating: Optional numerical rating (1-5)
        
        Returns:
            Dictionary with sentiment analysis results
        """
        try:
            # Use Claude for sentiment analysis (excellent for text understanding)
            sentiment_result = await self._analyze_with_claude(feedback_text, material_name, rating)

            return sentiment_result

        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            # Return fallback result
            return self._get_fallback_sentiment(feedback_text, rating)
    
    async def _analyze_with_claude(
        self,
        feedback_text: str,
        material_name: Optional[str],
        rating: Optional[int]
    ) -> Dict:
        """Use Claude for comprehensive sentiment analysis"""
        
        prompt = render(await load_prompt("extraction", "sentiment"),
            material_name=material_name or 'Unknown', rating=rating, feedback_text=feedback_text)

        try:
            from app.services.core.claude_helper import tracked_claude_call
            response = tracked_claude_call(
                task="sentiment_analysis",
                model="claude-haiku-4-5",
                max_tokens=500,
                temperature=0.1,
                system="You are an expert sentiment analysis AI. Analyze user feedback and provide structured JSON responses.",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
            )

            # Parse JSON response
            result = json.loads(response.content[0].text)

            # Add metadata
            result["analyzed_at"] = datetime.utcnow().isoformat()
            result["model_used"] = "claude-haiku-4-5"

            return result

        except Exception as e:
            logger.error(f"Claude sentiment analysis failed: {e}")
            return self._get_fallback_sentiment(feedback_text, rating)
    
    def _get_fallback_sentiment(self, feedback_text: str, rating: Optional[int]) -> Dict:
        """Fallback sentiment analysis using simple heuristics"""
        
        # Determine sentiment from rating if available
        if rating is not None:
            if rating >= 4:
                sentiment = "positive"
                confidence = 0.7
            elif rating <= 2:
                sentiment = "negative"
                confidence = 0.7
            else:
                sentiment = "neutral"
                confidence = 0.6
        else:
            # Simple keyword-based sentiment
            positive_words = ["good", "great", "excellent", "love", "beautiful", "perfect", "amazing", "wonderful"]
            negative_words = ["bad", "poor", "terrible", "hate", "ugly", "awful", "disappointing", "worst"]
            
            text_lower = feedback_text.lower()
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            if positive_count > negative_count:
                sentiment = "positive"
                confidence = min(0.5 + (positive_count * 0.1), 0.8)
            elif negative_count > positive_count:
                sentiment = "negative"
                confidence = min(0.5 + (negative_count * 0.1), 0.8)
            else:
                sentiment = "neutral"
                confidence = 0.5
        
        # Extract simple key phrases (sentences)
        sentences = re.split(r'[.!?]+', feedback_text)
        key_phrases = [s.strip() for s in sentences if len(s.strip()) > 10][:3]
        
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "aspects": {
                "quality": 0.5,
                "appearance": 0.5,
                "durability": 0.5,
                "value": 0.5,
                "usability": 0.5
            },
            "key_phrases": key_phrases,
            "recommendation_score": rating * 2 if rating else 5.0,
            "analyzed_at": datetime.utcnow().isoformat(),
            "model_used": "fallback-heuristic"
        }


