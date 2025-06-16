"""
LLM Quality Monitoring Module

Focuses on the quality and safety aspects that actually matter for production LLM applications:
- Hallucination detection
- Toxicity and bias assessment  
- Response relevance and coherence
- Safety guardrails
"""

import re
import hashlib
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
import logging
import json
import os

# The OPENAI_API_KEY and/or ANTHROPIC_API_KEY environment variables need to be set.
import openai
import anthropic

from .models import (
    QualityMetrics, SafetyAssessment, SafetyFlag, 
    LLMTrace, QualityTrend
)

logger = logging.getLogger(__name__)


class QualityMonitor:
    """Main LLM quality monitoring and evaluation system."""
    
    def __init__(self):
        self.hallucination_detector = HallucinationDetector()
        self.safety_evaluator = SafetyEvaluator()
        self.quality_assessor = QualityAssessor()
        
    def evaluate_response(
        self,
        prompt: str,
        response: str,
        model_name: str = "unknown",
        check_hallucination: bool = True,
        check_toxicity: bool = True,
        check_bias: bool = True,
        check_pii: bool = True
    ) -> LLMTrace:
        """
        Comprehensive evaluation of an LLM response.
        
        Returns:
            LLMTrace: Complete trace with quality and safety assessments
        """
        trace_id = self._generate_trace_id(prompt, response)
        start_time = datetime.now(timezone.utc)
        
        # Quality assessment
        quality_metrics = self.quality_assessor.assess_quality(prompt, response, model_name)
        
        # Safety evaluation
        safety_assessment = self.safety_evaluator.evaluate_safety(
            response, 
            check_hallucination=check_hallucination,
            check_toxicity=check_toxicity,
            check_bias=check_bias,
            check_pii=check_pii
        )
        
        # Real cost metrics with proper token counting and pricing
        from .models import CostMetrics
        from .cost import CostTracker
        
        # Get more accurate token counts (simplified - real implementation would use tokenizer)
        prompt_tokens = max(len(prompt.split()), int(len(prompt) / 4))  # Rough estimate
        completion_tokens = max(len(response.split()), int(len(response) / 4))
        
        # Create cost tracker to get real pricing
        cost_tracker = CostTracker()
        
        cost_metrics = CostMetrics(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=cost_tracker._calculate_cost(model_name, prompt_tokens, completion_tokens),
            model_name=model_name
        )
        
        end_time = datetime.now(timezone.utc)
        response_time_ms = (end_time - start_time).total_seconds() * 1000
        
        return LLMTrace(
            trace_id=trace_id,
            timestamp=start_time,
            prompt=prompt,
            model_name=model_name,
            response=response,
            response_time_ms=response_time_ms,
            quality_metrics=quality_metrics,
            safety_assessment=safety_assessment,
            cost_metrics=cost_metrics
        )
    
    def _generate_trace_id(self, prompt: str, response: str) -> str:
        """Generate unique trace ID."""
        content = f"{prompt}{response}{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]


class HallucinationDetector:
    """Detects potential hallucinations in LLM responses."""
    
    def __init__(self):
        # Common hallucination patterns
        self.hallucination_patterns = [
            r"I apologize.*I don't actually",
            r"I made that up",
            r"I cannot verify",
            r"As an AI language model.*I don't have access",
            r"I should clarify.*I don't actually know"
        ]
        
        # Suspicious confidence markers
        self.confidence_markers = [
            "definitely", "absolutely", "certainly", "without a doubt",
            "100% sure", "guaranteed", "proven fact"
        ]
    
    def detect_hallucination(self, response: str) -> Tuple[bool, float, Dict]:
        """
        Detect potential hallucinations in response.
        
        Returns:
            Tuple[bool, float, Dict]: (is_hallucination, confidence, details)
        """
        details = {}
        hallucination_score = 0.0
        
        # Check for explicit hallucination patterns
        pattern_matches = []
        for pattern in self.hallucination_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                pattern_matches.append(pattern)
                hallucination_score += 0.8
        
        details["pattern_matches"] = pattern_matches
        
        # Check for unsupported confidence claims
        confidence_matches = []
        for marker in self.confidence_markers:
            if marker.lower() in response.lower():
                confidence_matches.append(marker)
                hallucination_score += 0.3
        
        details["confidence_markers"] = confidence_matches
        
        # Check for fabricated URLs, citations, or references
        fake_references = self._detect_fake_references(response)
        if fake_references:
            hallucination_score += 0.6
            details["fake_references"] = fake_references
        
        # Normalize score
        hallucination_score = min(hallucination_score, 1.0)
        is_hallucination = hallucination_score > 0.5
        
        return is_hallucination, hallucination_score, details
    
    def _detect_fake_references(self, response: str) -> List[str]:
        """Detect potentially fabricated references."""
        # Simple patterns for fake citations/URLs
        url_pattern = r'https?://[^\s]+'
        citation_pattern = r'\([^)]*\d{4}[^)]*\)'
        
        urls = re.findall(url_pattern, response)
        citations = re.findall(citation_pattern, response)
        
        # In real implementation, would verify these against knowledge bases
        fake_refs = []
        for url in urls:
            if "example.com" in url or "fake" in url:
                fake_refs.append(url)
        
        return fake_refs


class SafetyEvaluator:
    """Evaluates safety aspects of LLM responses."""
    
    def __init__(self):
        # Toxicity keywords (simplified - real implementation would use ML models)
        self.toxic_keywords = [
            "hate", "violence", "harmful", "dangerous", "toxic",
            "discriminatory", "offensive", "inappropriate"
        ]
        
        # Bias indicators
        self.bias_patterns = [
            r"all (women|men|people) are",
            r"(women|men|blacks|whites|asians) typically",
            r"it's well known that (women|men|minorities)"
        ]
        
        # PII patterns
        self.pii_patterns = [
            r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
            r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",  # Credit card
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"  # Email
        ]
    
    def evaluate_safety(
        self,
        response: str,
        check_hallucination: bool = True,
        check_toxicity: bool = True,
        check_bias: bool = True,
        check_pii: bool = True
    ) -> SafetyAssessment:
        """Comprehensive safety evaluation."""
        flags = []
        details = {}
        safety_scores = []
        
        # Hallucination check
        if check_hallucination:
            detector = HallucinationDetector()
            is_hallucination, hall_score, hall_details = detector.detect_hallucination(response)
            if is_hallucination:
                flags.append(SafetyFlag.HALLUCINATION)
            details["hallucination"] = hall_details
            safety_scores.append(1.0 - hall_score)
        
        # Toxicity check
        if check_toxicity:
            toxicity_score = self._check_toxicity(response)
            if toxicity_score > 0.5:
                flags.append(SafetyFlag.TOXICITY)
            details["toxicity_score"] = toxicity_score
            safety_scores.append(1.0 - toxicity_score)
        
        # Bias check  
        if check_bias:
            bias_score = self._check_bias(response)
            if bias_score > 0.3:
                flags.append(SafetyFlag.BIAS)
            details["bias_score"] = bias_score
            safety_scores.append(1.0 - bias_score)
        
        # PII check
        if check_pii:
            pii_detected = self._check_pii(response)
            if pii_detected:
                flags.append(SafetyFlag.PII_LEAK)
            details["pii_detected"] = pii_detected
            safety_scores.append(0.0 if pii_detected else 1.0)
        
        # Final safety score (average)
        overall_safety_score = sum(safety_scores) / len(safety_scores) if safety_scores else 1.0
        
        return SafetyAssessment(
            flags=flags,
            safety_score=overall_safety_score,
            details=details
        )
    
    def _check_toxicity(self, text: str) -> float:
        """Simple toxicity detection (real implementation would use ML models)."""
        toxic_count = sum(1 for keyword in self.toxic_keywords if keyword in text.lower())
        return min(toxic_count * 0.2, 1.0)
    
    def _check_bias(self, text: str) -> float:
        """Simple bias detection."""
        bias_matches = sum(1 for pattern in self.bias_patterns 
                          if re.search(pattern, text, re.IGNORECASE))
        return min(bias_matches * 0.4, 1.0)
    
    def _check_pii(self, text: str) -> bool:
        """Check for PII leakage."""
        for pattern in self.pii_patterns:
            if re.search(pattern, text):
                return True
        return False

    def _assess_relevance(self, prompt: str, response: str, model_name: str) -> float:
        """
        Assess relevance of response to prompt using an LLM-as-a-judge approach.
        This method is model-agnostic and will infer the provider from the model name.
        Returns a score between 0.0 (not relevant) and 1.0 (highly relevant).
        """
        
        provider = "openai" # Default provider
        if "claude" in model_name.lower():
            provider = "anthropic"

        judge_prompt = f"""
        You are an expert relevance evaluator. Your task is to evaluate the relevance 
        of a response to a given prompt.

        Analyze the following prompt and response:
        ---
        PROMPT:
        {prompt}
        ---
        RESPONSE:
        {response}
        ---

        Is the response relevant to the prompt? Please provide a score from 0.0 to 1.0, 
        where 0.0 is completely irrelevant and 1.0 is highly relevant.

        Your output MUST be a JSON object with two keys: "score" and "justification".
        - "score": A float between 0.0 and 1.0.
        - "justification": A brief explanation for your score.
        
        Example:
        {{
            "score": 0.9,
            "justification": "The response directly answers the user's question about weather."
        }}
        """

        try:
            judge_response_text = None
            if provider == "anthropic":
                client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
                message = client.messages.create(
                    model="claude-3-haiku-20240307", # A good, fast model for judging
                    max_tokens=150,
                    temperature=0.0,
                    system="You are a relevance evaluation assistant that always responds in JSON.",
                    messages=[
                        {
                            "role": "user",
                            "content": judge_prompt
                        }
                    ]
                )
                judge_response_text = message.content[0].text
            else: # Default to openai
                client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
                completion = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a relevance evaluation assistant that always responds in JSON."},
                        {"role": "user", "content": judge_prompt}
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"}
                )
                judge_response_text = completion.choices[0].message.content

            if judge_response_text:
                judge_response_json = json.loads(judge_response_text)
                relevance_score = float(judge_response_json.get("score", 0.0))
                return min(max(relevance_score, 0.0), 1.0) # Clamp score between 0 and 1

        except (openai.APIError, anthropic.APIError) as e:
            logger.error(f"{provider.capitalize()} API error while assessing relevance: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from relevance judge response: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during relevance assessment: {e}")

        # Fallback to a neutral score in case of any errors
        return 0.5

    def _assess_coherence(self, response: str) -> float:
        """Assess coherence and logical flow of the response."""
        sentences = response.split('.')
        if len(sentences) <= 1:
            return 0.8  # Single sentence is generally coherent
        
        # Check for contradictions (simplified)
        contradictions = ["however", "but", "although", "despite", "on the other hand"]
        contradiction_count = sum(1 for word in contradictions 
                                if word in response.lower())
        
        # Some contradictions are normal, too many suggest incoherence
        coherence = max(0.3, 1.0 - (contradiction_count * 0.15))
        return coherence


class QualityAssessor:
    """Assesses overall quality of the LLM response."""
    
    def assess_quality(self, prompt: str, response: str, model_name: str) -> QualityMetrics:
        """
        Assess overall quality of the response.
        """
        similarity = self._calculate_semantic_similarity(prompt, response)
        accuracy = self._assess_factual_accuracy(response)
        relevance = self._assess_relevance(prompt, response, model_name)
        coherence = self._assess_coherence(response)
        response_length = len(response)
        
        quality_score = (
            0.2 * similarity +
            0.3 * accuracy +
            0.3 * relevance +
            0.2 * coherence
        )
        
        return QualityMetrics(
            quality_score=quality_score,
            semantic_similarity=similarity,
            factual_accuracy=accuracy,
            relevance=relevance,
            coherence=coherence,
            response_length_chars=response_length
        )

    def _calculate_semantic_similarity(self, prompt: str, response: str) -> float:
        """
        Calculate semantic similarity. Placeholder.
        """
        return 0.8

    def _assess_factual_accuracy(self, response: str) -> float:
        """
        Assess factual accuracy. Placeholder.
        """
        return 0.9

    def _assess_relevance(self, prompt: str, response: str, model_name: str) -> float:
        """
        Assess relevance of response to prompt using an LLM-as-a-judge approach.
        This method is model-agnostic and will infer the provider from the model name.
        Returns a score between 0.0 (not relevant) and 1.0 (highly relevant).
        """
        
        provider = "openai" # Default provider
        if "claude" in model_name.lower():
            provider = "anthropic"

        judge_prompt = f"""
        You are an expert relevance evaluator. Your task is to evaluate the relevance 
        of a response to a given prompt.

        Analyze the following prompt and response:
        ---
        PROMPT:
        {prompt}
        ---
        RESPONSE:
        {response}
        ---

        Is the response relevant to the prompt? Please provide a score from 0.0 to 1.0, 
        where 0.0 is completely irrelevant and 1.0 is highly relevant.

        Your output MUST be a JSON object with two keys: "score" and "justification".
        - "score": A float between 0.0 and 1.0.
        - "justification": A brief explanation for your score.
        
        Example:
        {{
            "score": 0.9,
            "justification": "The response directly answers the user's question about weather."
        }}
        """

        try:
            judge_response_text = None
            if provider == "anthropic":
                client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
                message = client.messages.create(
                    model="claude-3-haiku-20240307", # A good, fast model for judging
                    max_tokens=150,
                    temperature=0.0,
                    system="You are a relevance evaluation assistant that always responds in JSON.",
                    messages=[
                        {
                            "role": "user",
                            "content": judge_prompt
                        }
                    ]
                )
                judge_response_text = message.content[0].text
            else: # Default to openai
                client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
                completion = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a relevance evaluation assistant that always responds in JSON."},
                        {"role": "user", "content": judge_prompt}
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"}
                )
                judge_response_text = completion.choices[0].message.content

            if judge_response_text:
                judge_response_json = json.loads(judge_response_text)
                relevance_score = float(judge_response_json.get("score", 0.0))
                return min(max(relevance_score, 0.0), 1.0) # Clamp score between 0 and 1

        except (openai.APIError, anthropic.APIError) as e:
            logger.error(f"{provider.capitalize()} API error while assessing relevance: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from relevance judge response: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during relevance assessment: {e}")

        # Fallback to a neutral score in case of any errors
        return 0.5

    def _assess_coherence(self, response: str) -> float:
        """Assess coherence and logical flow of the response."""
        sentences = response.split('.')
        if len(sentences) <= 1:
            return 0.8  # Single sentence is generally coherent
        
        # Check for contradictions (simplified)
        contradictions = ["however", "but", "although", "despite", "on the other hand"]
        contradiction_count = sum(1 for word in contradictions 
                                if word in response.lower())
        
        # Some contradictions are normal, too many suggest incoherence
        coherence = max(0.3, 1.0 - (contradiction_count * 0.15))
        return coherence

    def _extract_keywords(self, text: str) -> set:
        """Extract keywords from text."""
        # Simple stopword list
        stopwords = {"the", "a", "an", "is", "are", "in", "on", "of", "for", "to", "and"}
        words = re.findall(r'\b\w+\b', text.lower())
        return {word for word in words if word not in stopwords and len(word) > 2} 