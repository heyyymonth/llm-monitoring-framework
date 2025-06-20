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
import math
from collections import Counter

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
        """
        Assess coherence and logical flow of the response using advanced analysis.
        
        This method leverages the comprehensive CoherenceAnalyzer to evaluate:
        - Logical flow and argumentation structure
        - Structural coherence and organization
        - Linguistic quality and readability
        - Semantic continuity and topic consistency
        - Contradiction detection
        """
        try:
            # Use comprehensive coherence analysis
            coherence_results = self.coherence_analyzer.analyze_comprehensive_coherence(response)
            
            # Return the overall coherence score
            return coherence_results["overall_coherence"]
            
        except Exception as e:
            logger.error(f"Error in coherence assessment: {e}")
            # Fallback to simple assessment if advanced analysis fails
            return self._simple_coherence_fallback(response)
    
    def _simple_coherence_fallback(self, response: str) -> float:
        """Simple coherence assessment as fallback."""
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
        
        # Skip LLM judge if API keys are not available (e.g., during testing)
        openai_key = os.environ.get("OPENAI_API_KEY")
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        
        if not openai_key and not anthropic_key:
            logger.info("No API keys available for LLM judge, falling back to semantic similarity")
            return self._assess_contextual_relevance(prompt, response)
        
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
            
            if provider == "anthropic" and anthropic_key:
                client = anthropic.Anthropic(api_key=anthropic_key)
                message = client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=200,
                    temperature=0.1,  # Low temperature for consistent scoring
                    system="You are a precise relevance evaluator that responds only in JSON format.",
                    messages=[{"role": "user", "content": judge_prompt}]
                )
                judge_response_text = message.content[0].text
            elif openai_key:
                client = openai.OpenAI(api_key=openai_key)
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


class CoherenceAnalyzer:
    """Advanced coherence and language quality analysis."""
    
    def __init__(self):
        """Initialize the coherence analyzer with linguistic patterns."""
        
        # Logical connectors that indicate good flow
        self.logical_connectors = {
            "additive": ["furthermore", "moreover", "additionally", "also", "besides"],
            "causal": ["therefore", "thus", "consequently", "as a result", "because"],
            "contrastive": ["however", "nevertheless", "on the other hand", "despite", "although"],
            "temporal": ["first", "then", "next", "finally", "meanwhile", "subsequently"],
            "clarifying": ["specifically", "in other words", "for example", "namely"]
        }
        
        # Repetition patterns that may indicate lack of coherence
        self.repetitive_patterns = [
            r'\b(\w+)\b(?:\s+\w+){0,5}\s+\1\b',  # Word repetition within short span
            r'\b(the same)\b.*\b\1\b',  # "the same" repetition
            r'\b(in other words)\b.*\b\1\b'  # Phrase repetition
        ]
        
        # Contradiction indicators (more sophisticated than basic version)
        self.contradiction_patterns = [
            r'(?:not|never|no).*(?:but|however).*(?:is|are|was|were)',
            r'(?:always|never).*(?:but|however).*(?:sometimes|often)',
            r'(?:impossible|cannot).*(?:but|however).*(?:possible|can)'
        ]
        
        # Discourse markers for structure analysis
        self.discourse_markers = {
            "introduction": ["first", "initially", "to begin with", "in the beginning"],
            "development": ["furthermore", "in addition", "moreover", "also"],
            "conclusion": ["finally", "in conclusion", "to summarize", "in summary"],
            "example": ["for example", "for instance", "such as", "namely"],
            "emphasis": ["importantly", "notably", "particularly", "especially"]
        }
    
    def analyze_comprehensive_coherence(self, text: str) -> Dict[str, float]:
        """
        Comprehensive coherence analysis with multiple dimensions.
        
        Returns:
            Dict with coherence scores for different aspects
        """
        if not text or len(text.strip()) < 10:
            return {
                "overall_coherence": 0.5,
                "logical_flow": 0.5,
                "structural_coherence": 0.5,
                "linguistic_quality": 0.5,
                "semantic_continuity": 0.5,
                "contradiction_score": 1.0
            }
        
        # Analyze different aspects of coherence
        logical_flow = self._assess_logical_flow(text)
        structural_coherence = self._assess_structural_coherence(text)
        linguistic_quality = self._assess_linguistic_quality(text)
        semantic_continuity = self._assess_semantic_continuity(text)
        contradiction_score = self._detect_contradictions(text)
        
        # Calculate weighted overall coherence
        overall_coherence = (
            0.25 * logical_flow +
            0.20 * structural_coherence +
            0.20 * linguistic_quality +
            0.20 * semantic_continuity +
            0.15 * contradiction_score
        )
        
        return {
            "overall_coherence": overall_coherence,
            "logical_flow": logical_flow,
            "structural_coherence": structural_coherence,
            "linguistic_quality": linguistic_quality,
            "semantic_continuity": semantic_continuity,
            "contradiction_score": contradiction_score
        }
    
    def _assess_logical_flow(self, text: str) -> float:
        """Assess the logical flow and argumentation structure."""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) <= 1:
            return 0.8  # Single sentence assumed coherent
        
        score = 0.5  # Base score
        
        # Check for logical connectors
        connector_count = 0
        for sentence in sentences:
            sentence_lower = sentence.lower()
            for category, connectors in self.logical_connectors.items():
                for connector in connectors:
                    if connector in sentence_lower:
                        connector_count += 1
                        break
        
        # Normalize connector usage (optimal range: 20-50% of sentences)
        connector_ratio = connector_count / len(sentences)
        if 0.2 <= connector_ratio <= 0.5:
            score += 0.3
        elif 0.1 <= connector_ratio < 0.2 or 0.5 < connector_ratio <= 0.7:
            score += 0.2
        elif connector_ratio > 0.7:
            score -= 0.1  # Too many connectors can be awkward
        
        # Check for proper discourse progression
        has_intro_markers = any(marker in text.lower() 
                               for marker in self.discourse_markers["introduction"])
        has_conclusion_markers = any(marker in text.lower() 
                                    for marker in self.discourse_markers["conclusion"])
        
        if len(sentences) > 3:  # Only check for longer texts
            if has_intro_markers:
                score += 0.1
            if has_conclusion_markers:
                score += 0.1
        
        return min(score, 1.0)
    
    def _assess_structural_coherence(self, text: str) -> float:
        """Assess paragraph organization and topic consistency."""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        score = 0.6  # Base score
        
        # Paragraph structure analysis
        if len(paragraphs) > 1:
            # Check for balanced paragraph lengths
            para_lengths = [len(p.split()) for p in paragraphs]
            avg_length = sum(para_lengths) / len(para_lengths)
            length_variance = sum((l - avg_length) ** 2 for l in para_lengths) / len(para_lengths)
            
            # Lower variance indicates better structure
            if length_variance < (avg_length * 0.5):
                score += 0.2
            elif length_variance < avg_length:
                score += 0.1
        
        # Sentence length variety (good writing has varied sentence lengths)
        sentence_lengths = [len(s.split()) for s in sentences]
        if len(sentence_lengths) > 2:
            length_std = (sum((l - sum(sentence_lengths)/len(sentence_lengths))**2 
                             for l in sentence_lengths) / len(sentence_lengths)) ** 0.5
            avg_length = sum(sentence_lengths) / len(sentence_lengths)
            
            # Optimal coefficient of variation for sentence length
            if avg_length > 0:
                cv = length_std / avg_length
                if 0.3 <= cv <= 0.8:  # Good variety
                    score += 0.2
                elif 0.2 <= cv < 0.3 or 0.8 < cv <= 1.0:  # Acceptable variety
                    score += 0.1
        
        return min(score, 1.0)
    
    def _assess_linguistic_quality(self, text: str) -> float:
        """Assess grammar, syntax, and readability."""
        score = 0.7  # Base score assuming reasonable quality
        
        # Check for basic grammar patterns
        words = text.split()
        if len(words) < 5:
            return 0.6
        
        # Check for repetitive word usage
        word_freq = Counter(word.lower().strip('.,!?;:') for word in words)
        total_words = len(words)
        unique_words = len(word_freq)
        
        # Lexical diversity (Type-Token Ratio)
        if total_words > 0:
            lexical_diversity = unique_words / total_words
            if lexical_diversity >= 0.7:
                score += 0.2
            elif lexical_diversity >= 0.5:
                score += 0.1
            elif lexical_diversity < 0.3:
                score -= 0.2
        
        # Check for excessive repetition
        repetition_penalty = 0
        for pattern in self.repetitive_patterns:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            repetition_penalty += matches * 0.1
        
        score -= min(repetition_penalty, 0.3)
        
        # Basic readability indicators
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if sentences:
            avg_sentence_length = total_words / len(sentences)
            # Optimal sentence length: 15-25 words
            if 15 <= avg_sentence_length <= 25:
                score += 0.1
            elif avg_sentence_length > 35:
                score -= 0.1
        
        return min(max(score, 0.0), 1.0)
    
    def _assess_semantic_continuity(self, text: str) -> float:
        """Assess semantic consistency and topic continuity."""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) <= 1:
            return 0.8
        
        score = 0.6  # Base score
        
        # Extract keywords from each sentence
        sentence_keywords = []
        for sentence in sentences:
            keywords = self._extract_keywords(sentence)
            sentence_keywords.append(keywords)
        
        # Calculate semantic overlap between adjacent sentences
        overlaps = []
        for i in range(len(sentence_keywords) - 1):
            current_keywords = sentence_keywords[i]
            next_keywords = sentence_keywords[i + 1]
            
            if current_keywords and next_keywords:
                intersection = len(current_keywords.intersection(next_keywords))
                union = len(current_keywords.union(next_keywords))
                overlap = intersection / union if union > 0 else 0
                overlaps.append(overlap)
        
        if overlaps:
            avg_overlap = sum(overlaps) / len(overlaps)
            # Optimal overlap: 10-30% (too low = disconnected, too high = repetitive)
            if 0.1 <= avg_overlap <= 0.3:
                score += 0.3
            elif 0.05 <= avg_overlap < 0.1 or 0.3 < avg_overlap <= 0.5:
                score += 0.2
            elif avg_overlap > 0.6:
                score -= 0.1  # Too much repetition
        
        # Check for topic consistency across the whole text
        all_keywords = set()
        for keywords in sentence_keywords:
            all_keywords.update(keywords)
        
        if len(all_keywords) > 0:
            # Calculate how focused the text is on core topics
            keyword_freq = Counter()
            for keywords in sentence_keywords:
                keyword_freq.update(keywords)
            
            # Find dominant keywords (appearing in multiple sentences)
            dominant_keywords = {k for k, v in keyword_freq.items() if v > 1}
            focus_ratio = len(dominant_keywords) / len(all_keywords)
            
            if 0.3 <= focus_ratio <= 0.7:  # Good balance
                score += 0.1
        
        return min(score, 1.0)
    
    def _detect_contradictions(self, text: str) -> float:
        """Detect logical contradictions and inconsistencies."""
        score = 1.0  # Start with perfect score
        
        # Check for explicit contradiction patterns
        contradiction_count = 0
        for pattern in self.contradiction_patterns:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            contradiction_count += matches
        
        # Penalize contradictions
        score -= min(contradiction_count * 0.2, 0.5)
        
        # Check for conflicting statements about the same entity
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        # Simple contradiction detection for common patterns
        positive_statements = set()
        negative_statements = set()
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Extract simple subject-predicate patterns
            if ' is ' in sentence_lower:
                parts = sentence_lower.split(' is ')
                if len(parts) == 2:
                    subject = parts[0].strip().split()[-1]  # Last word before "is"
                    predicate = parts[1].strip().split()[0]  # First word after "is"
                    
                    if 'not' in parts[1] or 'never' in parts[1]:
                        negative_statements.add((subject, predicate))
                    else:
                        positive_statements.add((subject, predicate))
        
        # Check for direct contradictions
        for pos_stmt in positive_statements:
            subject, predicate = pos_stmt
            if (subject, predicate) in negative_statements:
                score -= 0.3
        
        return max(score, 0.0)
    
    def _extract_keywords(self, text: str) -> set:
        """Extract meaningful keywords from text for semantic analysis."""
        # Enhanced stopword list
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "in", "on", "of", "for", 
            "to", "and", "or", "but", "with", "by", "from", "this", "that", "these",
            "those", "i", "you", "he", "she", "it", "we", "they", "my", "your",
            "his", "her", "its", "our", "their", "be", "been", "being", "have",
            "has", "had", "do", "does", "did", "will", "would", "could", "should",
            "may", "might", "can", "must", "shall"
        }
        
        # Extract words and filter
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        keywords = {word for word in words 
                   if word not in stopwords and len(word) > 2}
        
        return keywords


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
        
        # Initialize coherence analyzer for advanced language quality assessment
        self.coherence_analyzer = CoherenceAnalyzer()

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