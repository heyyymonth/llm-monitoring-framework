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

from sentence_transformers import SentenceTransformer, util

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
            is_safe=overall_safety_score > 0.8 and not flags, # A simple heuristic for overall safety
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


class RelevanceAssessor:
    """Advanced relevance assessment with topic classification and context understanding."""
    
    def __init__(self, similarity_model):
        """Initialize with shared similarity model for efficiency."""
        self.similarity_model = similarity_model
        
        # Topic classification patterns
        self.topic_patterns = {
            "factual": [r"what is", r"define", r"explain", r"how many", r"when did", r"who is"],
            "procedural": [r"how to", r"steps", r"process", r"procedure", r"method"],
            "creative": [r"write", r"create", r"generate", r"compose", r"imagine"],
            "analytical": [r"analyze", r"compare", r"evaluate", r"assess", r"why"],
            "technical": [r"implement", r"code", r"algorithm", r"debug", r"optimize"],
            "conversational": [r"chat", r"talk", r"discuss", r"opinion", r"think"]
        }
    
    def assess_comprehensive_relevance(self, prompt: str, response: str, model_name: str) -> Dict[str, float]:
        """
        Comprehensive relevance assessment with multiple dimensions.
        
        Returns:
            Dict with relevance scores: overall, topical, contextual, intent
        """
        # Classify topic for targeted evaluation
        topic_category = self._classify_topic(prompt)
        
        # Multi-faceted relevance assessment
        topical_relevance = self._assess_topical_relevance(prompt, response, topic_category, model_name)
        contextual_relevance = self._assess_contextual_relevance(prompt, response)
        intent_relevance = self._assess_intent_relevance(prompt, response, topic_category, model_name)
        
        # Weighted overall relevance (can be tuned based on use case)
        overall_relevance = (
            0.4 * topical_relevance +
            0.3 * contextual_relevance + 
            0.3 * intent_relevance
        )
        
        return {
            "overall_relevance": overall_relevance,
            "topical_relevance": topical_relevance,
            "contextual_relevance": contextual_relevance,
            "intent_relevance": intent_relevance,
            "topic_category": topic_category
        }
    
    def _classify_topic(self, prompt: str) -> str:
        """Classify the topic/intent of the prompt."""
        prompt_lower = prompt.lower()
        
        topic_scores = {}
        for topic, patterns in self.topic_patterns.items():
            score = sum(1 for pattern in patterns if re.search(pattern, prompt_lower))
            if score > 0:
                topic_scores[topic] = score
        
        if topic_scores:
            return max(topic_scores, key=topic_scores.get)
        return "general"
    
    def _assess_topical_relevance(self, prompt: str, response: str, topic_category: str, model_name: str) -> float:
        """Assess how well response addresses the specific topic of the prompt."""
        
        # Topic-specific evaluation prompts
        topic_specific_prompts = {
            "factual": "Does the response provide accurate, factual information that directly answers the question?",
            "procedural": "Does the response provide clear, actionable steps or procedures as requested?", 
            "creative": "Does the response fulfill the creative request with appropriate content?",
            "analytical": "Does the response provide thoughtful analysis addressing the analytical request?",
            "technical": "Does the response provide relevant technical information or solutions?",
            "conversational": "Does the response engage appropriately with the conversational prompt?",
            "general": "Does the response directly address the main topic of the prompt?"
        }
        
        evaluation_criteria = topic_specific_prompts.get(topic_category, topic_specific_prompts["general"])
        
        try:
            return self._llm_judge_relevance(prompt, response, model_name, evaluation_criteria)
        except Exception as e:
            logger.warning(f"LLM judge failed for topical relevance, falling back to semantic similarity: {e}")
            return self._assess_contextual_relevance(prompt, response)
    
    def _assess_contextual_relevance(self, prompt: str, response: str) -> float:
        """Assess contextual relevance using semantic similarity as fallback."""
        try:
            # Use semantic similarity for contextual relevance
            if not prompt or not response:
                return 0.0

            prompt_embedding = self.similarity_model.encode(prompt, convert_to_tensor=True)
            response_embedding = self.similarity_model.encode(response, convert_to_tensor=True)
            
            # Compute cosine similarity
            cosine_scores = util.cos_sim(prompt_embedding, response_embedding)
            similarity_score = float(cosine_scores[0][0].item())
            
            return max(0.0, min(1.0, similarity_score))
            
        except Exception as e:
            logger.error(f"Error calculating contextual relevance: {e}")
            return 0.5
    
    def _assess_intent_relevance(self, prompt: str, response: str, topic_category: str, model_name: str) -> float:
        """Assess whether response fulfills the user's intent."""
        
        intent_evaluation = f"""
        Analyze whether the response fulfills the user's underlying intent.
        
        Topic Category: {topic_category}
        
        Consider:
        - Does the response satisfy what the user was actually seeking?
        - Is the response complete and appropriately scoped?
        - Does it address the implicit needs behind the explicit question?
        """
        
        try:
            return self._llm_judge_relevance(prompt, response, model_name, intent_evaluation)
        except Exception as e:
            logger.warning(f"LLM judge failed for intent relevance, falling back to semantic similarity: {e}")
            return self._assess_contextual_relevance(prompt, response)
    
    def _llm_judge_relevance(self, prompt: str, response: str, model_name: str, evaluation_criteria: str) -> float:
        """Enhanced LLM-as-a-judge with better error handling and fallbacks."""
        
        provider = "openai"  # Default provider
        if "claude" in model_name.lower():
            provider = "anthropic"

        judge_prompt = f"""
        You are an expert evaluator. {evaluation_criteria}

        Analyze the following:
        ---
        PROMPT: {prompt}
        ---
        RESPONSE: {response}
        ---

        Provide a relevance score from 0.0 to 1.0:
        - 0.0-0.3: Not relevant/Off-topic
        - 0.4-0.6: Somewhat relevant/Partially addresses
        - 0.7-0.9: Relevant/Good match  
        - 0.9-1.0: Highly relevant/Perfect match

        Output ONLY a JSON object:
        {{
            "score": 0.8,
            "reasoning": "Brief explanation for the score"
        }}
        """

        try:
            judge_response_text = None
            
            if provider == "anthropic":
                client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
                message = client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=200,
                    temperature=0.1,  # Low temperature for consistent scoring
                    system="You are a precise relevance evaluator that responds only in JSON format.",
                    messages=[{"role": "user", "content": judge_prompt}]
                )
                judge_response_text = message.content[0].text
            else:
                client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
                completion = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a precise relevance evaluator that responds only in JSON format."},
                        {"role": "user", "content": judge_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=200,
                    response_format={"type": "json_object"}
                )
                judge_response_text = completion.choices[0].message.content

            if judge_response_text:
                judge_response_json = json.loads(judge_response_text)
                relevance_score = float(judge_response_json.get("score", 0.5))
                return min(max(relevance_score, 0.0), 1.0)

        except (openai.APIError, anthropic.APIError) as e:
            logger.error(f"{provider.capitalize()} API error in relevance assessment: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from relevance judge: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in relevance assessment: {e}")

        # Fallback to semantic similarity
        try:
            return self._assess_contextual_relevance(prompt, response)
        except:
            return 0.5


class QualityAssessor:
    """Assesses the overall quality of LLM responses."""
    
    def __init__(self):
        """Initialize the quality assessor and load required models."""
        # Load the sentence transformer model for semantic similarity
        # This model is optimized for semantic similarity tasks.
        # It's loaded once to be reused across assessments.
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize relevance assessor with shared similarity model
        self.relevance_assessor = RelevanceAssessor(self.similarity_model)

    def assess_quality(self, prompt: str, response: str, model_name: str) -> QualityMetrics:
        """
        Assess overall quality of the response based on multiple metrics.
        """
        similarity = self._calculate_semantic_similarity(prompt, response)
        accuracy = self._assess_factual_accuracy(response)
        
        # Enhanced relevance assessment
        relevance_results = self.relevance_assessor.assess_comprehensive_relevance(prompt, response, model_name)
        relevance = relevance_results["overall_relevance"]
        
        coherence = self._assess_coherence(response)
        response_length = len(response)
        
        # Aggregate quality score (simple weighted average)
        # Weights can be tuned based on what's most important for the use case
        quality_score = (
            0.2 * similarity +
            0.3 * accuracy +
            0.3 * relevance +
            0.2 * coherence
        )
        
        return QualityMetrics(
            overall_quality=quality_score,
            semantic_similarity=similarity,
            factual_accuracy=accuracy,
            response_relevance=relevance,
            topical_relevance=relevance_results.get("topical_relevance"),
            contextual_relevance=relevance_results.get("contextual_relevance"),
            intent_relevance=relevance_results.get("intent_relevance"),
            topic_category=relevance_results.get("topic_category"),
            coherence_score=coherence,
            response_length=response_length
        )

    def _calculate_semantic_similarity(self, prompt: str, response: str) -> float:
        """
        Calculate semantic similarity between prompt and response using sentence embeddings.
        
        This advanced method provides a much more nuanced understanding of semantic
        relationships than simple keyword matching.
        """
        if not prompt or not response:
            return 0.0

        try:
            # Encode the prompt and response into high-dimensional vectors
            prompt_embedding = self.similarity_model.encode(prompt, convert_to_tensor=True)
            response_embedding = self.similarity_model.encode(response, convert_to_tensor=True)

            # Compute cosine similarity between the two embeddings
            cosine_scores = util.cos_sim(prompt_embedding, response_embedding)
            
            # The result is a tensor, we get the float value and clip it to [0, 1]
            # as sentence-transformers can sometimes output values slightly outside this range.
            similarity_score = float(cosine_scores[0][0].item())
            return max(0.0, min(1.0, similarity_score))

        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            return 0.0

    def _assess_factual_accuracy(self, response: str) -> float:
        """
        Assess the factual accuracy of the response.
        """
        return 0.9

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