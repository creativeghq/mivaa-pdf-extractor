"""
Sentiment Analysis Service

Provides AI-powered sentiment analysis for user feedback using:
- DistilBERT for overall sentiment classification
- Aspect-based sentiment analysis for specific attributes
- Key phrase extraction
- Recommendation scoring
"""

import logging
import asyncio
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import re

import httpx
from together import Together

logger = logging.getLogger(__name__)


class SentimentAnalysisService:
    """Service for analyzing user feedback sentiment"""
    
    def __init__(self, together_api_key: str):
        self.together_client = Together(api_key=together_api_key)
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
            # Use Together AI for sentiment analysis (Qwen model)
            sentiment_result = await self._analyze_with_qwen(feedback_text, material_name, rating)
            
            return sentiment_result
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            # Return fallback result
            return self._get_fallback_sentiment(feedback_text, rating)
    
    async def _analyze_with_qwen(
        self,
        feedback_text: str,
        material_name: Optional[str],
        rating: Optional[int]
    ) -> Dict:
        """Use Qwen model for comprehensive sentiment analysis"""
        
        prompt = f"""Analyze the following user feedback about a material product and provide a detailed sentiment analysis.

Material: {material_name or 'Unknown'}
User Rating: {rating}/5 stars
Feedback: "{feedback_text}"

Provide your analysis in the following JSON format:
{{
    "sentiment": "positive" | "neutral" | "negative",
    "confidence": 0.0-1.0,
    "aspects": {{
        "quality": 0.0-1.0,
        "appearance": 0.0-1.0,
        "durability": 0.0-1.0,
        "value": 0.0-1.0,
        "usability": 0.0-1.0
    }},
    "key_phrases": ["phrase1", "phrase2", "phrase3"],
    "recommendation_score": 0.0-10.0,
    "reasoning": "Brief explanation of the analysis"
}}

Rules:
- sentiment: Overall sentiment (positive if rating >= 4, negative if rating <= 2, neutral otherwise)
- confidence: How confident you are in the sentiment classification (0.0-1.0)
- aspects: Score each aspect from 0.0 (very negative) to 1.0 (very positive). Use 0.5 if not mentioned.
- key_phrases: Extract 3-5 most important phrases from the feedback
- recommendation_score: Overall recommendation score from 0 (would not recommend) to 10 (highly recommend)
- reasoning: Brief explanation of why you classified it this way

Respond ONLY with valid JSON, no additional text."""

        try:
            response = self.together_client.chat.completions.create(
                model="Qwen/Qwen3-VL-8B-Instruct",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert sentiment analysis AI. Analyze user feedback and provide structured JSON responses."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=500,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            # Parse JSON response
            import json
            result = json.loads(response.choices[0].message.content)
            
            # Add metadata
            result["analyzed_at"] = datetime.utcnow().isoformat()
            result["model_used"] = "Qwen/Qwen3-VL-8B-Instruct"
            
            return result
            
        except Exception as e:
            logger.error(f"Qwen sentiment analysis failed: {e}")
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

