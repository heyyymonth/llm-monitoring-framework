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
from typing import Dict, List, Optional, Tuple, Set, Union
from datetime import datetime, timezone
import logging
import json
import os
import math
import numpy as np
from collections import Counter, defaultdict, deque

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
        self.factual_accuracy_monitor = FactualAccuracyMonitor()
        
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
    
    def evaluate_factual_accuracy(self, prompt: str, response: str) -> Dict[str, Union[float, Dict, List]]:
        """
        Detailed factual accuracy evaluation for content verification and consistency.
        
        Args:
            prompt: User's input prompt
            response: LLM's response to evaluate
            
        Returns:
            Dict with comprehensive factual accuracy metrics and analysis
        """
        return self.factual_accuracy_monitor.assess_factual_accuracy(response, prompt)


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


class AdvancedRelevanceAnalyzer:
    """
    Advanced relevance analysis with enhanced topic classification, 
    context understanding, and multi-modal assessment capabilities.
    """
    
    def __init__(self, similarity_model):
        """Initialize with enhanced models and configurations."""
        self.similarity_model = similarity_model
        
        # Enhanced topic classification with domain-specific patterns
        self.enhanced_topic_patterns = {
            # Technical domains
            "software_engineering": [
                r"implement|code|programming|algorithm|debug|API|framework|library",
                r"backend|frontend|database|deployment|architecture|design pattern"
            ],
            "data_science": [
                r"machine learning|data analysis|statistics|modeling|prediction|ML|AI",
                r"visualization|dataset|feature|training|regression|classification"
            ],
            "cybersecurity": [
                r"security|vulnerability|encryption|authentication|firewall|threat",
                r"penetration testing|malware|phishing|compliance|privacy"
            ],
            
            # Academic domains  
            "mathematics": [
                r"equation|formula|theorem|proof|calculate|algebra|geometry|calculus",
                r"probability|statistics|number theory|linear algebra|differential"
            ],
            "science": [
                r"experiment|hypothesis|research|theory|evidence|scientific method",
                r"physics|chemistry|biology|astronomy|geology|laboratory"
            ],
            "literature": [
                r"novel|poem|author|character|plot|theme|literary analysis|symbolism",
                r"narrative|metaphor|genre|style|interpretation|criticism"
            ],
            
            # Business domains
            "business_strategy": [
                r"strategy|market|competition|revenue|profit|growth|expansion|ROI",
                r"stakeholder|investor|business model|value proposition|partnership"
            ],
            "marketing": [
                r"brand|campaign|advertising|customer|target audience|conversion",
                r"SEO|social media|content marketing|lead generation|analytics"
            ],
            "finance": [
                r"investment|portfolio|risk|return|capital|asset|liability|valuation",
                r"stock|bond|derivative|interest rate|financial planning|budget"
            ],
            
            # General interaction types
            "instructional": [
                r"how to|step by step|tutorial|guide|instruction|procedure|method",
                r"teach me|show me|explain how|walk through|demonstrate"
            ],
            "exploratory": [
                r"what if|suppose|imagine|consider|explore|investigate|possibility",
                r"alternative|option|scenario|hypothetical|brainstorm"
            ],
            "comparative": [
                r"compare|contrast|difference|similarity|versus|vs|better|worse",
                r"advantage|disadvantage|pros and cons|trade-off|evaluation"
            ],
            "analytical": [
                r"analyze|evaluate|assess|examine|investigate|review|critique",
                r"breakdown|dissect|study|scrutinize|interpret|understand"
            ]
        }
        
        # Context patterns for conversation flow analysis
        self.context_patterns = {
            "follow_up": [
                r"also|additionally|furthermore|moreover|what about|and|plus",
                r"can you also|tell me more|expand on|elaborate|continue"
            ],
            "clarification": [
                r"what do you mean|clarify|explain|I don't understand|unclear",
                r"can you explain|what is|define|meaning|interpretation"
            ],
            "correction": [
                r"actually|no|that's wrong|incorrect|mistake|error|fix|correct",
                r"I meant|let me clarify|what I actually want|correction"
            ],
            "refinement": [
                r"be more specific|details|precisely|exactly|particular|focus on",
                r"narrow down|zoom in|get specific|be precise|elaborate on"
            ]
        }
        
        # Domain-specific relevance weights
        self.domain_weights = {
            "technical": {"accuracy": 0.4, "completeness": 0.3, "practicality": 0.3},
            "academic": {"accuracy": 0.5, "depth": 0.3, "clarity": 0.2},
            "business": {"actionability": 0.4, "relevance": 0.3, "clarity": 0.3},
            "creative": {"originality": 0.4, "relevance": 0.3, "engagement": 0.3},
            "general": {"relevance": 0.4, "clarity": 0.3, "completeness": 0.3}
        }
        
        # Conversation context tracking
        self.conversation_history = deque(maxlen=10)  # Keep last 10 interactions
        self.context_embedding_cache = {}
        
    def analyze_enhanced_relevance(
        self, 
        prompt: str, 
        response: str, 
        model_name: str,
        conversation_context: Optional[List[Dict]] = None
    ) -> Dict[str, Union[float, str, Dict]]:
        """
        Comprehensive relevance analysis with enhanced topic classification and context understanding.
        
        Args:
            prompt: User's input prompt
            response: LLM's response
            model_name: Name of the model used
            conversation_context: Optional conversation history
            
        Returns:
            Dict with detailed relevance metrics and analysis
        """
        # Enhanced topic classification
        topic_analysis = self._analyze_topic_comprehensive(prompt)
        
        # Context understanding and flow analysis
        context_analysis = self._analyze_context_flow(prompt, conversation_context)
        
        # Multi-dimensional relevance assessment
        relevance_scores = self._assess_multi_dimensional_relevance(
            prompt, response, topic_analysis, context_analysis, model_name
        )
        
        # Domain-specific scoring adjustments
        domain_adjusted_scores = self._apply_domain_specific_scoring(
            relevance_scores, topic_analysis["primary_domain"]
        )
        
        # Calculate overall enhanced relevance
        overall_relevance = self._calculate_weighted_relevance(domain_adjusted_scores)
        
        return {
            "overall_relevance": overall_relevance,
            "topic_analysis": topic_analysis,
            "context_analysis": context_analysis,
            "relevance_dimensions": domain_adjusted_scores,
            "confidence_score": self._calculate_confidence_score(domain_adjusted_scores),
            "recommendations": self._generate_improvement_recommendations(domain_adjusted_scores)
        }
    
    def _analyze_topic_comprehensive(self, prompt: str) -> Dict[str, Union[str, float, List]]:
        """Enhanced topic classification with confidence scoring and multi-label support."""
        # Handle empty or very short prompts
        if not prompt or len(prompt.strip()) < 3:
            return {
                "primary_domain": "general",
                "secondary_domains": [],
                "topic_scores": {},
                "complexity_score": 0.0,
                "specificity_score": 0.0,
                "is_multi_domain": False
            }
        
        prompt_lower = prompt.lower()
        topic_scores = defaultdict(float)
        
        # Multi-pattern matching with confidence weighting
        for domain, pattern_groups in self.enhanced_topic_patterns.items():
            domain_score = 0
            matched_patterns = []
            
            for patterns in pattern_groups:
                pattern_matches = len(re.findall(patterns, prompt_lower))
                if pattern_matches > 0:
                    domain_score += pattern_matches * 0.3
                    matched_patterns.append(patterns)
            
            if domain_score > 0:
                topic_scores[domain] = min(domain_score, 1.0)
        
        # Determine primary and secondary topics
        sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
        
        primary_domain = sorted_topics[0][0] if sorted_topics else "general"
        secondary_domains = [topic for topic, score in sorted_topics[1:3] if score >= 0.3]
        
        # Calculate topic complexity and specificity with safe division
        prompt_words = prompt.split()
        word_count = len(prompt_words)
        
        complexity_score = min(word_count / 20.0, 1.0) if word_count > 0 else 0.0
        long_words = [word for word in prompt_words if len(word) > 6]
        specificity_score = len(long_words) / word_count if word_count > 0 else 0.0
        
        return {
            "primary_domain": primary_domain,
            "secondary_domains": secondary_domains,
            "topic_scores": dict(topic_scores),
            "complexity_score": complexity_score,
            "specificity_score": specificity_score,
            "is_multi_domain": len(secondary_domains) > 0
        }
    
    def _analyze_context_flow(self, prompt: str, conversation_context: Optional[List[Dict]]) -> Dict:
        """Analyze conversation context and flow patterns."""
        context_analysis = {
            "conversation_type": "initial",
            "context_continuity": 0.0,
            "reference_clarity": 1.0,
            "context_shift_detected": False,
            "requires_history": False
        }
        
        if not conversation_context:
            return context_analysis
        
        prompt_lower = prompt.lower()
        
        # Detect conversation patterns
        for pattern_type, patterns in self.context_patterns.items():
            for pattern_group in patterns:
                if re.search(pattern_group, prompt_lower):
                    context_analysis["conversation_type"] = pattern_type
                    context_analysis["requires_history"] = pattern_type in ["follow_up", "clarification", "refinement"]
                    break
        
        # Analyze context continuity with recent conversation
        if len(conversation_context) > 0:
            context_analysis["context_continuity"] = self._calculate_context_continuity(
                prompt, conversation_context
            )
            
            # Detect pronouns and references requiring context
            pronouns = ["it", "this", "that", "they", "them", "these", "those"]
            reference_count = sum(1 for pronoun in pronouns if pronoun in prompt_lower.split())
            context_analysis["reference_clarity"] = max(0.0, 1.0 - (reference_count * 0.2))
        
        return context_analysis
    
    def _calculate_context_continuity(self, current_prompt: str, conversation_context: List[Dict]) -> float:
        """Calculate semantic continuity with conversation history."""
        if not conversation_context:
            return 0.0
        
        try:
            # Get embeddings for current prompt
            current_embedding = self.similarity_model.encode(current_prompt, convert_to_tensor=True)
            
            # Calculate similarity with recent context
            similarities = []
            for i, context_item in enumerate(conversation_context[-3:]):  # Last 3 interactions
                context_text = context_item.get("prompt", "") + " " + context_item.get("response", "")
                if context_text.strip():
                    context_embedding = self.similarity_model.encode(context_text, convert_to_tensor=True)
                    similarity = util.cos_sim(current_embedding, context_embedding)[0][0].item()
                    
                    # Weight recent interactions more heavily
                    weight = 1.0 - (i * 0.2)
                    similarities.append(similarity * weight)
            
            return sum(similarities) / len(similarities) if similarities else 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating context continuity: {e}")
            return 0.0
    
    def _assess_multi_dimensional_relevance(
        self, 
        prompt: str, 
        response: str, 
        topic_analysis: Dict, 
        context_analysis: Dict,
        model_name: str
    ) -> Dict[str, float]:
        """Multi-dimensional relevance assessment with enhanced scoring."""
        
        # Base semantic relevance
        semantic_relevance = self._calculate_enhanced_semantic_similarity(prompt, response)
        
        # Topic-specific relevance
        topic_relevance = self._assess_topic_specific_relevance(
            prompt, response, topic_analysis["primary_domain"], model_name
        )
        
        # Context-aware relevance
        context_relevance = self._assess_context_aware_relevance(
            prompt, response, context_analysis
        )
        
        # Intent fulfillment assessment
        intent_fulfillment = self._assess_intent_fulfillment(
            prompt, response, topic_analysis, model_name
        )
        
        # Completeness and depth assessment
        completeness_score = self._assess_response_completeness(prompt, response, topic_analysis)
        
        # Practical utility assessment
        utility_score = self._assess_practical_utility(prompt, response, topic_analysis["primary_domain"])
        
        return {
            "semantic_relevance": semantic_relevance,
            "topic_relevance": topic_relevance,
            "context_relevance": context_relevance,
            "intent_fulfillment": intent_fulfillment,
            "completeness": completeness_score,
            "practical_utility": utility_score
        }
    
    def _calculate_enhanced_semantic_similarity(self, prompt: str, response: str) -> float:
        """Enhanced semantic similarity with multiple strategies."""
        if not prompt or not response:
            return 0.0
        
        try:
            # Strategy 1: Direct semantic similarity
            prompt_embedding = self.similarity_model.encode(prompt, convert_to_tensor=True)
            response_embedding = self.similarity_model.encode(response, convert_to_tensor=True)
            direct_similarity = util.cos_sim(prompt_embedding, response_embedding)[0][0].item()
            
            # Strategy 2: Keyword overlap enhancement
            prompt_keywords = self._extract_enhanced_keywords(prompt)
            response_keywords = self._extract_enhanced_keywords(response)
            
            if prompt_keywords and response_keywords:
                keyword_overlap = len(prompt_keywords.intersection(response_keywords)) / len(prompt_keywords.union(response_keywords))
            else:
                keyword_overlap = 0.0
            
            # Strategy 3: Sentence-level alignment
            sentence_alignment = self._calculate_sentence_alignment(prompt, response)
            
            # Weighted combination
            enhanced_similarity = (
                0.6 * direct_similarity +
                0.2 * keyword_overlap +
                0.2 * sentence_alignment
            )
            
            return max(0.0, min(1.0, enhanced_similarity))
            
        except Exception as e:
            logger.error(f"Error in enhanced semantic similarity calculation: {e}")
            return 0.0
    
    def _extract_enhanced_keywords(self, text: str) -> Set[str]:
        """Extract meaningful keywords with enhanced filtering."""
        # Enhanced stopwords
        stopwords = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "up", "about", "into", "through", "during",
            "before", "after", "above", "below", "between", "among", "i", "you", 
            "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
            "my", "your", "his", "her", "its", "our", "their", "this", "that",
            "these", "those", "am", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "must", "can", "shall"
        }
        
        # Extract and filter words
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        keywords = {
            word for word in words 
            if len(word) > 2 and word not in stopwords and not word.isdigit()
        }
        
        return keywords
    
    def _calculate_sentence_alignment(self, prompt: str, response: str) -> float:
        """Calculate alignment between prompt and response at sentence level."""
        try:
            prompt_sentences = [s.strip() for s in prompt.split('.') if s.strip()]
            response_sentences = [s.strip() for s in response.split('.') if s.strip()]
            
            if not prompt_sentences or not response_sentences:
                return 0.0
            
            # Calculate cross-sentence similarities
            max_similarities = []
            for p_sent in prompt_sentences:
                if len(p_sent) > 10:  # Only consider substantial sentences
                    p_embedding = self.similarity_model.encode(p_sent, convert_to_tensor=True)
                    
                    sent_similarities = []
                    for r_sent in response_sentences:
                        if len(r_sent) > 10:
                            r_embedding = self.similarity_model.encode(r_sent, convert_to_tensor=True)
                            similarity = util.cos_sim(p_embedding, r_embedding)[0][0].item()
                            sent_similarities.append(similarity)
                    
                    if sent_similarities:
                        max_similarities.append(max(sent_similarities))
            
            return sum(max_similarities) / len(max_similarities) if max_similarities else 0.0
            
        except Exception as e:
            logger.warning(f"Error in sentence alignment calculation: {e}")
            return 0.0
    
    def _assess_topic_specific_relevance(self, prompt: str, response: str, domain: str, model_name: str) -> float:
        """Assess relevance specific to the identified domain."""
        
        domain_evaluation_criteria = {
            "software_engineering": "Does the response provide technically accurate, practical programming guidance that directly addresses the software engineering question?",
            "data_science": "Does the response demonstrate proper understanding of data science concepts and provide actionable analytical insights?",
            "cybersecurity": "Does the response address security concerns with appropriate technical depth and practical recommendations?",
            "mathematics": "Does the response provide mathematically rigorous and correct explanations that properly address the mathematical question?",
            "science": "Does the response demonstrate scientific accuracy and provide evidence-based explanations relevant to the scientific inquiry?",
            "business_strategy": "Does the response provide strategic insights and actionable business recommendations relevant to the question?",
            "instructional": "Does the response provide clear, step-by-step guidance that effectively teaches the requested skill or knowledge?",
            "analytical": "Does the response provide thorough analysis with logical reasoning that addresses all aspects of the analytical request?",
            "general": "Does the response directly and comprehensively address the main topics and intent of the question?"
        }
        
        evaluation_criteria = domain_evaluation_criteria.get(domain, domain_evaluation_criteria["general"])
        
        try:
            return self._llm_judge_relevance(prompt, response, model_name, evaluation_criteria)
        except Exception as e:
            logger.warning(f"Domain-specific relevance assessment failed, using fallback: {e}")
            return self._calculate_enhanced_semantic_similarity(prompt, response)
    
    def _assess_context_aware_relevance(self, prompt: str, response: str, context_analysis: Dict) -> float:
        """Assess relevance considering conversation context and flow."""
        base_score = 0.7
        
        # Adjust based on context continuity
        if context_analysis["requires_history"]:
            continuity_bonus = context_analysis["context_continuity"] * 0.2
            reference_penalty = (1.0 - context_analysis["reference_clarity"]) * 0.15
            base_score += continuity_bonus - reference_penalty
        
        # Adjust for conversation type
        conversation_type = context_analysis["conversation_type"]
        if conversation_type == "clarification" and len(response) > 200:
            base_score += 0.1  # Bonus for detailed clarifications
        elif conversation_type == "follow_up" and "also" in response.lower():
            base_score += 0.05  # Bonus for acknowledging follow-up nature
        
        return max(0.0, min(1.0, base_score))
    
    def _assess_intent_fulfillment(self, prompt: str, response: str, topic_analysis: Dict, model_name: str) -> float:
        """Enhanced intent fulfillment assessment."""
        
        intent_criteria = f"""
        Evaluate how well the response fulfills the user's underlying intent and goals.
        
        Domain: {topic_analysis["primary_domain"]}
        Complexity Level: {topic_analysis["complexity_score"]:.2f}
        Multi-domain: {topic_analysis["is_multi_domain"]}
        
        Consider:
        - Does the response satisfy the user's explicit and implicit needs?
        - Is the response appropriately detailed for the complexity level?
        - Does it address multi-domain aspects if present?
        - Is the response actionable and useful for the user's goals?
        """
        
        try:
            return self._llm_judge_relevance(prompt, response, model_name, intent_criteria)
        except Exception as e:
            logger.warning(f"Intent fulfillment assessment failed, using fallback: {e}")
            return self._calculate_enhanced_semantic_similarity(prompt, response)
    
    def _assess_response_completeness(self, prompt: str, response: str, topic_analysis: Dict) -> float:
        """Assess how complete and comprehensive the response is."""
        
        # Base completeness factors
        prompt_length = len(prompt.split())
        response_length = len(response.split())
        
        # Expected response length based on prompt complexity
        expected_length = max(50, prompt_length * 2)
        if topic_analysis["complexity_score"] > 0.7:
            expected_length *= 1.5
        
        # Length appropriateness score
        length_ratio = response_length / expected_length
        length_score = min(1.0, length_ratio) if length_ratio <= 2.0 else max(0.5, 2.0 / length_ratio)
        
        # Content structure score
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        structure_score = min(1.0, len(sentences) / 5.0)  # Optimal around 5+ sentences
        
        # Multi-domain coverage if applicable
        coverage_score = 1.0
        if topic_analysis["is_multi_domain"]:
            secondary_coverage = sum(
                1 for domain in topic_analysis["secondary_domains"]
                if any(pattern in response.lower() for pattern_group in self.enhanced_topic_patterns.get(domain, [])
                      for pattern in pattern_group.split('|'))
            )
            coverage_score = min(1.0, secondary_coverage / len(topic_analysis["secondary_domains"]))
        
        # Weighted completeness score
        completeness = (
            0.4 * length_score +
            0.3 * structure_score +
            0.3 * coverage_score
        )
        
        return max(0.0, min(1.0, completeness))
    
    def _assess_practical_utility(self, prompt: str, response: str, domain: str) -> float:
        """Assess the practical utility and actionability of the response."""
        
        utility_indicators = {
            "software_engineering": ["example", "code", "implementation", "step", "function", "method"],
            "data_science": ["dataset", "model", "algorithm", "example", "analysis", "visualization"],
            "business_strategy": ["action", "strategy", "implement", "plan", "step", "approach"],
            "instructional": ["step", "first", "then", "next", "example", "practice"],
            "general": ["example", "step", "how", "approach", "method", "way"]
        }
        
        indicators = utility_indicators.get(domain, utility_indicators["general"])
        response_lower = response.lower()
        
        # Count utility indicators
        utility_count = sum(1 for indicator in indicators if indicator in response_lower)
        utility_score = min(1.0, utility_count / len(indicators))
        
        # Bonus for specific formats
        if "step" in response_lower and ("1." in response or "first" in response_lower):
            utility_score += 0.2
        if "example" in response_lower and len(response) > 100:
            utility_score += 0.1
        
        return max(0.0, min(1.0, utility_score))
    
    def _apply_domain_specific_scoring(self, relevance_scores: Dict[str, float], domain: str) -> Dict[str, float]:
        """Apply domain-specific weighting to relevance scores."""
        
        domain_category = "general"
        if domain in ["software_engineering", "data_science", "cybersecurity"]:
            domain_category = "technical"
        elif domain in ["mathematics", "science", "literature"]:
            domain_category = "academic"
        elif domain in ["business_strategy", "marketing", "finance"]:
            domain_category = "business"
        elif domain in ["instructional", "exploratory", "creative"]:
            domain_category = "creative"
        
        weights = self.domain_weights[domain_category]
        
        # Map relevance dimensions to domain weights
        dimension_mapping = {
            "accuracy": ["semantic_relevance", "topic_relevance"],
            "completeness": ["completeness", "intent_fulfillment"],
            "practicality": ["practical_utility", "context_relevance"],
            "depth": ["completeness", "topic_relevance"],
            "clarity": ["context_relevance", "semantic_relevance"],
            "actionability": ["practical_utility", "intent_fulfillment"],
            "relevance": ["semantic_relevance", "topic_relevance", "context_relevance"],
            "originality": ["intent_fulfillment", "practical_utility"],
            "engagement": ["context_relevance", "practical_utility"]
        }
        
        adjusted_scores = relevance_scores.copy()
        
        # Apply domain-specific adjustments
        for weight_category, weight_value in weights.items():
            if weight_category in dimension_mapping:
                for dimension in dimension_mapping[weight_category]:
                    if dimension in adjusted_scores:
                        # Apply weight multiplier
                        adjusted_scores[dimension] *= (1.0 + (weight_value - 0.33) * 0.3)
        
        # Normalize scores
        for key in adjusted_scores:
            adjusted_scores[key] = max(0.0, min(1.0, adjusted_scores[key]))
        
        return adjusted_scores
    
    def _calculate_weighted_relevance(self, relevance_scores: Dict[str, float]) -> float:
        """Calculate overall weighted relevance score."""
        
        weights = {
            "semantic_relevance": 0.25,
            "topic_relevance": 0.25,
            "context_relevance": 0.15,
            "intent_fulfillment": 0.20,
            "completeness": 0.10,
            "practical_utility": 0.05
        }
        
        weighted_score = sum(
            relevance_scores.get(dimension, 0.5) * weight
            for dimension, weight in weights.items()
        )
        
        return max(0.0, min(1.0, weighted_score))
    
    def _calculate_confidence_score(self, relevance_scores: Dict[str, float]) -> float:
        """Calculate confidence in the relevance assessment."""
        
        # Higher confidence when scores are consistent
        score_variance = np.var(list(relevance_scores.values()))
        consistency_score = max(0.0, 1.0 - (score_variance * 2))
        
        # Higher confidence for extreme scores (very high or very low)
        avg_score = np.mean(list(relevance_scores.values()))
        extremeness = abs(avg_score - 0.5) * 2
        
        # Combine factors
        confidence = (0.7 * consistency_score + 0.3 * extremeness)
        
        return max(0.0, min(1.0, confidence))
    
    def _generate_improvement_recommendations(self, relevance_scores: Dict[str, float]) -> List[str]:
        """Generate recommendations for improving relevance."""
        
        recommendations = []
        threshold = 0.6
        
        if relevance_scores.get("semantic_relevance", 1.0) < threshold:
            recommendations.append("Improve semantic alignment with the user's question")
        
        if relevance_scores.get("topic_relevance", 1.0) < threshold:
            recommendations.append("Address the specific domain and technical requirements more directly")
        
        if relevance_scores.get("context_relevance", 1.0) < threshold:
            recommendations.append("Better incorporate conversation context and maintain continuity")
        
        if relevance_scores.get("intent_fulfillment", 1.0) < threshold:
            recommendations.append("Focus more on fulfilling the user's underlying goals and intent")
        
        if relevance_scores.get("completeness", 1.0) < threshold:
            recommendations.append("Provide more comprehensive coverage of the topic")
        
        if relevance_scores.get("practical_utility", 1.0) < threshold:
            recommendations.append("Include more actionable steps, examples, or practical guidance")
        
        return recommendations
    
    def _llm_judge_relevance(self, prompt: str, response: str, model_name: str, evaluation_criteria: str) -> float:
        """Enhanced LLM-as-a-judge with better error handling and fallbacks."""
        
        # Skip LLM judge if API keys are not available (e.g., during testing)
        openai_key = os.environ.get("OPENAI_API_KEY")
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        
        if not openai_key and not anthropic_key:
            logger.info("No API keys available for LLM judge, falling back to semantic similarity")
            return self._calculate_enhanced_semantic_similarity(prompt, response)
        
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

        # Fallback to enhanced semantic similarity
        try:
            return self._calculate_enhanced_semantic_similarity(prompt, response)
        except:
            return 0.5


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


class FactualAccuracyMonitor:
    """
    Comprehensive factual accuracy monitoring for content verification and consistency.
    
    Evaluates multiple dimensions of factual accuracy:
    - Internal consistency and contradiction detection
    - Citation and reference verification
    - Quantitative data accuracy (numbers, dates, statistics)
    - Knowledge consistency assessment
    - Temporal accuracy evaluation
    """
    
    def __init__(self):
        """Initialize factual accuracy monitoring components."""
        # Patterns for detecting specific claim types
        self.quantitative_patterns = {
            'numbers': r'\b\d+(?:,\d{3})*(?:\.\d+)?\b',
            'percentages': r'\b\d+(?:\.\d+)?%\b',
            'dates': r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b',
            'years': r'\b(?:19|20)\d{2}\b',
            'currencies': r'\$\d+(?:,\d{3})*(?:\.\d{2})?|\b\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars?|USD|euros?|EUR|pounds?|GBP)\b'
        }
        
        # Citation and reference patterns
        self.citation_patterns = {
            'academic': r'\([^)]*\d{4}[^)]*\)',  # (Author, 2023)
            'urls': r'https?://[^\s<>"\']+',
            'doi': r'doi:\s*10\.\d+/[^\s]+',
            'isbn': r'ISBN[-\s]*(?:\d{1,5}[-\s]*){4}\d{1,7}',
            'journal_ref': r'\b[A-Z][a-z]+\s+(?:Journal|Review|Proceedings|Conference)\b'
        }
        
        # Common factual claim indicators
        self.factual_claim_indicators = [
            'according to', 'research shows', 'studies indicate', 'data reveals',
            'statistics show', 'evidence suggests', 'findings demonstrate',
            'reports indicate', 'surveys show', 'analysis reveals'
        ]
        
        # Uncertainty and hedge words that may indicate lower confidence
        self.uncertainty_indicators = [
            'approximately', 'roughly', 'about', 'around', 'nearly', 'almost',
            'likely', 'probably', 'possibly', 'potentially', 'might', 'could',
            'appears to', 'seems to', 'suggests that', 'indicates that'
        ]
        
        # Contradiction detection patterns
        self.contradiction_patterns = [
            (r'\bnot\s+(\w+)', r'\bis\s+\1'),  # "not X" vs "is X"
            (r'\bnever\s+(\w+)', r'\balways\s+\1'),  # "never X" vs "always X"
            (r'\bimpossible\s+to\s+(\w+)', r'\bcan\s+\1'),  # "impossible to X" vs "can X"
            (r'\bno\s+(\w+)', r'\bhas\s+\1'),  # "no X" vs "has X"
        ]
        
        # Known factual domains for targeted verification
        self.verification_domains = {
            'scientific': ['temperature', 'speed of light', 'gravity', 'DNA', 'evolution'],
            'historical': ['world war', 'independence', 'revolution', 'empire', 'treaty'],
            'geographical': ['capital', 'population', 'area', 'mountain', 'river', 'country'],
            'mathematical': ['pi', 'fibonacci', 'prime number', 'theorem', 'equation'],
            'technological': ['invention', 'patent', 'algorithm', 'protocol', 'standard']
        }
    
    def assess_factual_accuracy(self, response: str, prompt: str = "", context: Dict = None) -> Dict[str, Union[float, Dict, List]]:
        """
        Comprehensive factual accuracy assessment.
        
        Args:
            response: The LLM response to evaluate
            prompt: Original prompt for context
            context: Additional context information
            
        Returns:
            Dict containing accuracy scores and detailed analysis
        """
        if not response or len(response.strip()) < 10:
            return {
                'overall_accuracy': 0.5,
                'consistency_score': 0.5,
                'citation_score': 1.0,  # No citations to verify
                'quantitative_accuracy': 1.0,  # No quantitative claims
                'knowledge_consistency': 0.5,
                'temporal_accuracy': 1.0,
                'confidence_indicators': {},
                'detected_claims': [],
                'verification_results': {},
                'recommendations': []
            }
        
        # Analyze different dimensions of factual accuracy
        consistency_analysis = self._assess_internal_consistency(response)
        citation_analysis = self._assess_citations_and_references(response)
        quantitative_analysis = self._assess_quantitative_accuracy(response)
        knowledge_analysis = self._assess_knowledge_consistency(response, prompt)
        temporal_analysis = self._assess_temporal_accuracy(response)
        confidence_analysis = self._analyze_confidence_indicators(response)
        
        # Calculate overall factual accuracy score
        overall_accuracy = self._calculate_overall_accuracy(
            consistency_analysis['score'],
            citation_analysis['score'],
            quantitative_analysis['score'],
            knowledge_analysis['score'],
            temporal_analysis['score']
        )
        
        return {
            'overall_accuracy': overall_accuracy,
            'consistency_score': consistency_analysis['score'],
            'citation_score': citation_analysis['score'],
            'quantitative_accuracy': quantitative_analysis['score'],
            'knowledge_consistency': knowledge_analysis['score'],
            'temporal_accuracy': temporal_analysis['score'],
            'confidence_indicators': confidence_analysis,
            'detected_claims': self._extract_factual_claims(response),
            'verification_results': {
                'consistency': consistency_analysis['details'],
                'citations': citation_analysis['details'],
                'quantitative': quantitative_analysis['details'],
                'knowledge': knowledge_analysis['details'],
                'temporal': temporal_analysis['details']
            },
            'recommendations': self._generate_accuracy_recommendations(
                consistency_analysis, citation_analysis, quantitative_analysis,
                knowledge_analysis, temporal_analysis
            )
        }
    
    def _assess_internal_consistency(self, response: str) -> Dict:
        """Detect internal contradictions and inconsistencies."""
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        contradictions = []
        consistency_issues = []
        
        # Check for direct contradictions using patterns
        for i, sentence1 in enumerate(sentences):
            for j, sentence2 in enumerate(sentences[i+1:], i+1):
                contradiction_found = self._detect_sentence_contradiction(sentence1, sentence2)
                if contradiction_found:
                    contradictions.append({
                        'sentence1': sentence1,
                        'sentence2': sentence2,
                        'type': contradiction_found
                    })
        
        # Check for inconsistent entity descriptions
        entities_mentioned = self._extract_entities(response)
        for entity, descriptions in entities_mentioned.items():
            if len(descriptions) > 1:
                # Check if descriptions are consistent
                inconsistency = self._check_entity_consistency(descriptions)
                if inconsistency:
                    consistency_issues.append({
                        'entity': entity,
                        'descriptions': descriptions,
                        'issue': inconsistency
                    })
        
        # Calculate consistency score
        total_statements = len(sentences)
        contradiction_penalty = len(contradictions) * 0.3
        consistency_penalty = len(consistency_issues) * 0.2
        
        consistency_score = max(0.0, 1.0 - (contradiction_penalty + consistency_penalty) / max(total_statements, 1))
        
        return {
            'score': consistency_score,
            'details': {
                'contradictions': contradictions,
                'consistency_issues': consistency_issues,
                'total_statements': total_statements
            }
        }
    
    def _assess_citations_and_references(self, response: str) -> Dict:
        """Evaluate citation quality and reference verification."""
        citations_found = {}
        citation_issues = []
        
        # Detect different types of citations
        for citation_type, pattern in self.citation_patterns.items():
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                citations_found[citation_type] = matches
        
        # Analyze citation quality
        if not citations_found:
            # Check if response makes factual claims that should be cited
            claim_indicators = sum(1 for indicator in self.factual_claim_indicators 
                                 if indicator in response.lower())
            if claim_indicators > 0:
                citation_issues.append("Contains factual claims without citations")
                citation_score = 0.6  # Penalty for uncited claims
            else:
                citation_score = 1.0  # No citations needed
        else:
            citation_score = 0.8  # Base score for having citations
            
            # Verify citation formats and accessibility
            for citation_type, citations in citations_found.items():
                for citation in citations:
                    issue = self._verify_citation_format(citation, citation_type)
                    if issue:
                        citation_issues.append(f"{citation_type}: {issue}")
                        citation_score -= 0.1
        
        citation_score = max(0.0, min(1.0, citation_score))
        
        return {
            'score': citation_score,
            'details': {
                'citations_found': citations_found,
                'citation_issues': citation_issues,
                'citation_density': len(sum(citations_found.values(), [])) / max(len(response.split()), 1)
            }
        }
    
    def _assess_quantitative_accuracy(self, response: str) -> Dict:
        """Evaluate accuracy of quantitative data (numbers, dates, statistics)."""
        quantitative_claims = {}
        accuracy_issues = []
        
        # Extract quantitative information
        for data_type, pattern in self.quantitative_patterns.items():
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                quantitative_claims[data_type] = matches
        
        # Verify quantitative claims
        verification_results = {}
        total_claims = 0
        accurate_claims = 0
        
        for data_type, claims in quantitative_claims.items():
            verification_results[data_type] = []
            for claim in claims:
                total_claims += 1
                verification = self._verify_quantitative_claim(claim, data_type, response)
                verification_results[data_type].append(verification)
                if verification['likely_accurate']:
                    accurate_claims += 1
                else:
                    accuracy_issues.append(f"{data_type}: {claim} - {verification['issue']}")
        
        # Calculate accuracy score
        if total_claims == 0:
            accuracy_score = 1.0  # No quantitative claims to verify
        else:
            accuracy_score = accurate_claims / total_claims
        
        return {
            'score': accuracy_score,
            'details': {
                'quantitative_claims': quantitative_claims,
                'verification_results': verification_results,
                'accuracy_issues': accuracy_issues,
                'total_claims': total_claims,
                'accurate_claims': accurate_claims
            }
        }
    
    def _assess_knowledge_consistency(self, response: str, prompt: str = "") -> Dict:
        """Assess consistency with established knowledge."""
        knowledge_issues = []
        domain_scores = {}
        
        # Identify knowledge domains mentioned
        mentioned_domains = []
        response_lower = response.lower()
        
        for domain, keywords in self.verification_domains.items():
            domain_mentions = sum(1 for keyword in keywords if keyword in response_lower)
            if domain_mentions > 0:
                mentioned_domains.append(domain)
                # Simplified domain verification (in practice, would use knowledge bases)
                domain_score = self._verify_domain_knowledge(response, domain, keywords)
                domain_scores[domain] = domain_score
        
        # Calculate overall knowledge consistency
        if not domain_scores:
            knowledge_score = 0.8  # Neutral score for general knowledge
        else:
            knowledge_score = sum(domain_scores.values()) / len(domain_scores)
        
        return {
            'score': knowledge_score,
            'details': {
                'mentioned_domains': mentioned_domains,
                'domain_scores': domain_scores,
                'knowledge_issues': knowledge_issues
            }
        }
    
    def _assess_temporal_accuracy(self, response: str) -> Dict:
        """Assess temporal accuracy and currency of information."""
        temporal_claims = []
        temporal_issues = []
        
        # Extract temporal references
        years = re.findall(self.quantitative_patterns['years'], response)
        dates = re.findall(self.quantitative_patterns['dates'], response)
        
        # Check for outdated information indicators
        outdated_indicators = [
            'recent studies', 'latest research', 'current trends', 'nowadays',
            'today', 'this year', 'recently', 'modern'
        ]
        
        temporal_context_score = 1.0
        
        for indicator in outdated_indicators:
            if indicator in response.lower():
                temporal_claims.append(indicator)
                # In practice, would check if information is actually current
                # For now, apply slight penalty for potentially outdated claims
                temporal_context_score -= 0.05
        
        # Verify year references for reasonableness
        current_year = datetime.now().year
        for year in years:
            year_int = int(year)
            if year_int > current_year:
                temporal_issues.append(f"Future year mentioned: {year}")
                temporal_context_score -= 0.2
            elif year_int < 1900 and year_int > current_year - 100:
                # Potentially suspicious very old dates in modern context
                temporal_issues.append(f"Potentially anachronistic year: {year}")
                temporal_context_score -= 0.1
        
        temporal_score = max(0.0, temporal_context_score)
        
        return {
            'score': temporal_score,
            'details': {
                'temporal_claims': temporal_claims,
                'temporal_issues': temporal_issues,
                'years_mentioned': years,
                'dates_mentioned': dates
            }
        }
    
    def _analyze_confidence_indicators(self, response: str) -> Dict:
        """Analyze confidence and uncertainty indicators in the response."""
        uncertainty_count = 0
        certainty_indicators = []
        uncertainty_phrases = []
        
        # Count uncertainty indicators
        for indicator in self.uncertainty_indicators:
            if indicator in response.lower():
                uncertainty_count += 1
                uncertainty_phrases.append(indicator)
        
        # Look for certainty claims
        certainty_words = ['definitely', 'certainly', 'absolutely', 'undoubtedly', 'proven', 'fact']
        for word in certainty_words:
            if word in response.lower():
                certainty_indicators.append(word)
        
        # Calculate confidence calibration
        total_indicators = uncertainty_count + len(certainty_indicators)
        uncertainty_ratio = uncertainty_count / max(total_indicators, 1)
        
        return {
            'uncertainty_count': uncertainty_count,
            'uncertainty_phrases': uncertainty_phrases,
            'certainty_indicators': certainty_indicators,
            'uncertainty_ratio': uncertainty_ratio,
            'confidence_calibration': self._assess_confidence_calibration(uncertainty_ratio, len(certainty_indicators))
        }
    
    def _calculate_overall_accuracy(self, consistency: float, citation: float, 
                                  quantitative: float, knowledge: float, temporal: float) -> float:
        """Calculate weighted overall factual accuracy score."""
        weights = {
            'consistency': 0.25,
            'citation': 0.20,
            'quantitative': 0.20,
            'knowledge': 0.25,
            'temporal': 0.10
        }
        
        overall_score = (
            weights['consistency'] * consistency +
            weights['citation'] * citation +
            weights['quantitative'] * quantitative +
            weights['knowledge'] * knowledge +
            weights['temporal'] * temporal
        )
        
        return overall_score
    
    def _detect_sentence_contradiction(self, sentence1: str, sentence2: str) -> Optional[str]:
        """Detect contradictions between two sentences."""
        sentence1_lower = sentence1.lower()
        sentence2_lower = sentence2.lower()
        
        # Check contradiction patterns
        for neg_pattern, pos_pattern in self.contradiction_patterns:
            try:
                neg_matches = re.findall(neg_pattern, sentence1_lower)
                pos_matches = re.findall(pos_pattern, sentence2_lower)
                
                # Check if same entity is mentioned with contradictory attributes
                for neg_match in neg_matches:
                    for pos_match in pos_matches:
                        if neg_match == pos_match:
                            return f"Contradiction found: negation vs affirmation"
            except re.error:
                # Skip invalid regex patterns
                continue
        
        # Simple word-based contradiction detection as fallback
        contradiction_words = [
            ('not', 'is'), ('never', 'always'), ('impossible', 'possible'),
            ('cannot', 'can'), ('no', 'yes'), ('false', 'true')
        ]
        
        for neg_word, pos_word in contradiction_words:
            if neg_word in sentence1_lower and pos_word in sentence2_lower:
                return f"Simple contradiction: {neg_word} vs {pos_word}"
        
        return None
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities and their descriptions for consistency checking."""
        # Simplified entity extraction (in practice, would use NER)
        entities = {}
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        # Look for patterns like "X is Y" or "X has Y"
        for sentence in sentences:
            patterns = [
                r'(\w+(?:\s+\w+)?)\s+is\s+([^.]+)',
                r'(\w+(?:\s+\w+)?)\s+has\s+([^.]+)',
                r'(\w+(?:\s+\w+)?)\s+can\s+([^.]+)'
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, sentence, re.IGNORECASE)
                for entity, description in matches:
                    entity = entity.strip().lower()
                    if len(entity) > 2:  # Filter out very short entities
                        if entity not in entities:
                            entities[entity] = []
                        entities[entity].append(description.strip())
        
        return entities
    
    def _check_entity_consistency(self, descriptions: List[str]) -> Optional[str]:
        """Check if entity descriptions are consistent."""
        # Simplified consistency check
        if len(descriptions) < 2:
            return None
        
        # Look for obvious contradictions
        contradiction_pairs = [
            ('big', 'small'), ('large', 'small'), ('fast', 'slow'),
            ('hot', 'cold'), ('high', 'low'), ('good', 'bad')
        ]
        
        desc_text = ' '.join(descriptions).lower()
        for word1, word2 in contradiction_pairs:
            if word1 in desc_text and word2 in desc_text:
                return f"Contradictory descriptions: {word1} vs {word2}"
        
        return None
    
    def _verify_citation_format(self, citation: str, citation_type: str) -> Optional[str]:
        """Verify citation format and basic validity."""
        if citation_type == 'urls':
            # Basic URL validation
            if not citation.startswith(('http://', 'https://')):
                return "Invalid URL format"
            if 'example.com' in citation or 'test.' in citation:
                return "Suspicious test/example URL"
        
        elif citation_type == 'academic':
            # Check for proper academic citation format
            if not re.search(r'\d{4}', citation):
                return "Missing year in citation"
        
        elif citation_type == 'doi':
            # Basic DOI format check
            if not citation.startswith('doi:10.'):
                return "Invalid DOI format"
        
        return None
    
    def _verify_quantitative_claim(self, claim: str, data_type: str, context: str) -> Dict:
        """Verify quantitative claims for reasonableness."""
        # Simplified verification logic
        verification = {'likely_accurate': True, 'issue': None}
        
        if data_type == 'percentages':
            try:
                value = float(claim.rstrip('%'))
                if value > 100:
                    verification = {'likely_accurate': False, 'issue': 'Percentage over 100%'}
                elif value < 0:
                    verification = {'likely_accurate': False, 'issue': 'Negative percentage'}
            except ValueError:
                verification = {'likely_accurate': False, 'issue': 'Invalid percentage format'}
        
        elif data_type == 'years':
            try:
                year = int(claim)
                current_year = datetime.now().year
                if year > current_year:
                    verification = {'likely_accurate': False, 'issue': 'Future year'}
                elif year < 1000:
                    verification = {'likely_accurate': False, 'issue': 'Implausibly old year'}
            except ValueError:
                verification = {'likely_accurate': False, 'issue': 'Invalid year format'}
        
        elif data_type == 'numbers':
            # Check for implausibly large numbers without context
            try:
                num_str = claim.replace(',', '')
                if '.' in num_str:
                    value = float(num_str)
                else:
                    value = int(num_str)
                
                # Flag very large numbers that might be errors
                if value > 1e12:  # Trillion+
                    verification = {'likely_accurate': False, 'issue': 'Implausibly large number'}
            except ValueError:
                verification = {'likely_accurate': False, 'issue': 'Invalid number format'}
        
        return verification
    
    def _verify_domain_knowledge(self, response: str, domain: str, keywords: List[str]) -> float:
        """Verify knowledge claims in specific domains."""
        # Simplified domain verification (in practice, would use knowledge graphs)
        score = 0.8  # Base score
        
        response_lower = response.lower()
        
        # Domain-specific checks
        if domain == 'scientific':
            # Check for common scientific misconceptions
            misconceptions = ['evolution is just a theory', 'vaccines cause autism']
            for misconception in misconceptions:
                if misconception in response_lower:
                    score -= 0.3
        
        elif domain == 'mathematical':
            # Check for mathematical errors
            if 'pi equals' in response_lower:
                pi_claims = re.findall(r'pi equals (\d+\.?\d*)', response_lower)
                for claim in pi_claims:
                    try:
                        pi_value = float(claim)
                        if abs(pi_value - 3.14159) > 0.01:  # Allow some rounding
                            score -= 0.2
                    except ValueError:
                        score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def _extract_factual_claims(self, response: str) -> List[Dict]:
        """Extract factual claims from the response."""
        claims = []
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        
        for sentence in sentences:
            # Check if sentence contains factual claim indicators
            for indicator in self.factual_claim_indicators:
                if indicator in sentence.lower():
                    claims.append({
                        'sentence': sentence,
                        'indicator': indicator,
                        'confidence': 'medium'  # Could be enhanced with ML classification
                    })
                    break
        
        return claims
    
    def _assess_confidence_calibration(self, uncertainty_ratio: float, certainty_count: int) -> str:
        """Assess how well confidence is calibrated in the response."""
        if uncertainty_ratio > 0.5 and certainty_count == 0:
            return "well_calibrated_uncertain"
        elif uncertainty_ratio < 0.2 and certainty_count > 2:
            return "overconfident"
        elif 0.2 <= uncertainty_ratio <= 0.5:
            return "well_calibrated_balanced"
        else:
            return "poorly_calibrated"
    
    def _generate_accuracy_recommendations(self, consistency_analysis: Dict, citation_analysis: Dict,
                                        quantitative_analysis: Dict, knowledge_analysis: Dict,
                                        temporal_analysis: Dict) -> List[str]:
        """Generate recommendations for improving factual accuracy."""
        recommendations = []
        
        if consistency_analysis['score'] < 0.8:
            recommendations.append("Review response for internal contradictions and inconsistencies")
        
        if citation_analysis['score'] < 0.7:
            recommendations.append("Add proper citations for factual claims")
        
        if quantitative_analysis['score'] < 0.8:
            recommendations.append("Verify numerical data and quantitative claims")
        
        if knowledge_analysis['score'] < 0.7:
            recommendations.append("Cross-check domain-specific knowledge claims")
        
        if temporal_analysis['score'] < 0.8:
            recommendations.append("Verify temporal claims and update outdated information")
        
        return recommendations


class QualityAssessor:
    """Assesses the overall quality of LLM responses."""
    
    def __init__(self):
        """Initialize the quality assessor and load required models."""
        # Load the sentence transformer model for semantic similarity
        # This model is optimized for semantic similarity tasks.
        # It's loaded once to be reused across assessments.
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize advanced relevance analyzer for enhanced topic and context analysis
        self.advanced_relevance_analyzer = AdvancedRelevanceAnalyzer(self.similarity_model)
        
        # Initialize legacy relevance assessor for backward compatibility
        self.relevance_assessor = RelevanceAssessor(self.similarity_model)
        
        # Initialize coherence analyzer for advanced language quality assessment
        self.coherence_analyzer = CoherenceAnalyzer()
        
        # Initialize factual accuracy monitor for comprehensive content verification
        self.factual_accuracy_monitor = FactualAccuracyMonitor()

    def assess_quality(self, prompt: str, response: str, model_name: str) -> QualityMetrics:
        """
        Assess overall quality of the response based on multiple metrics with enhanced relevance analysis.
        """
        similarity = self._calculate_semantic_similarity(prompt, response)
        accuracy = self._assess_factual_accuracy(response, prompt)
        
        # Enhanced relevance assessment with advanced topic and context analysis
        enhanced_relevance_results = self.advanced_relevance_analyzer.analyze_enhanced_relevance(
            prompt, response, model_name
        )
        relevance = enhanced_relevance_results["overall_relevance"]
        
        # Extract enhanced relevance metrics
        topic_analysis = enhanced_relevance_results["topic_analysis"]
        
        coherence = self._assess_coherence(response)
        response_length = len(response)
        
        # Aggregate quality score with enhanced weighting based on domain
        # Adjust weights based on detected domain for better accuracy
        domain_category = self._categorize_domain(topic_analysis["primary_domain"])
        weights = self._get_domain_quality_weights(domain_category)
        
        quality_score = (
            weights["similarity"] * similarity +
            weights["accuracy"] * accuracy +
            weights["relevance"] * relevance +
            weights["coherence"] * coherence
        )
        
        return QualityMetrics(
            overall_quality=quality_score,
            semantic_similarity=similarity,
            factual_accuracy=accuracy,
            response_relevance=relevance,
            topical_relevance=enhanced_relevance_results["relevance_dimensions"].get("topic_relevance"),
            contextual_relevance=enhanced_relevance_results["relevance_dimensions"].get("context_relevance"), 
            intent_relevance=enhanced_relevance_results["relevance_dimensions"].get("intent_fulfillment"),
            topic_category=topic_analysis["primary_domain"],
            coherence_score=coherence,
            response_length=response_length
        )
    
    def assess_enhanced_quality(self, prompt: str, response: str, model_name: str, conversation_context: Optional[List[Dict]] = None) -> Dict:
        """
        Enhanced quality assessment with detailed relevance analysis and recommendations.
        
        Args:
            prompt: User's input prompt
            response: LLM's response
            model_name: Name of the model used
            conversation_context: Optional conversation history
            
        Returns:
            Dict with comprehensive quality metrics and analysis
        """
        # Standard quality metrics
        quality_metrics = self.assess_quality(prompt, response, model_name)
        
        # Enhanced relevance analysis with conversation context
        enhanced_relevance_results = self.advanced_relevance_analyzer.analyze_enhanced_relevance(
            prompt, response, model_name, conversation_context
        )
        
        return {
            "standard_quality": quality_metrics.model_dump(),
            "enhanced_relevance": enhanced_relevance_results,
            "domain_insights": {
                "primary_domain": enhanced_relevance_results["topic_analysis"]["primary_domain"],
                "complexity_score": enhanced_relevance_results["topic_analysis"]["complexity_score"],
                "is_multi_domain": enhanced_relevance_results["topic_analysis"]["is_multi_domain"],
                "secondary_domains": enhanced_relevance_results["topic_analysis"]["secondary_domains"]
            },
            "context_insights": enhanced_relevance_results["context_analysis"],
            "improvement_recommendations": enhanced_relevance_results["recommendations"],
            "confidence_score": enhanced_relevance_results["confidence_score"]
        }
    
    def _categorize_domain(self, domain: str) -> str:
        """Categorize domain into broader categories for quality weighting."""
        technical_domains = ["software_engineering", "data_science", "cybersecurity", "mathematics"]
        academic_domains = ["science", "literature", "mathematics"]
        business_domains = ["business_strategy", "marketing", "finance"]
        creative_domains = ["instructional", "exploratory", "comparative", "analytical"]
        
        if domain in technical_domains:
            return "technical"
        elif domain in academic_domains:
            return "academic"
        elif domain in business_domains:
            return "business"
        elif domain in creative_domains:
            return "creative"
        else:
            return "general"
    
    def _get_domain_quality_weights(self, domain_category: str) -> Dict[str, float]:
        """Get domain-specific weights for quality scoring."""
        domain_weights = {
            "technical": {"similarity": 0.15, "accuracy": 0.40, "relevance": 0.35, "coherence": 0.10},
            "academic": {"similarity": 0.20, "accuracy": 0.35, "relevance": 0.30, "coherence": 0.15},
            "business": {"similarity": 0.20, "accuracy": 0.25, "relevance": 0.40, "coherence": 0.15},
            "creative": {"similarity": 0.25, "accuracy": 0.20, "relevance": 0.35, "coherence": 0.20},
            "general": {"similarity": 0.20, "accuracy": 0.30, "relevance": 0.30, "coherence": 0.20}
        }
        
        return domain_weights.get(domain_category, domain_weights["general"])

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

    def _assess_factual_accuracy(self, response: str, prompt: str = "") -> float:
        """
        Assess the factual accuracy of the response using comprehensive content verification.
        """
        try:
            # Use the comprehensive factual accuracy monitor
            accuracy_analysis = self.factual_accuracy_monitor.assess_factual_accuracy(response, prompt)
            return accuracy_analysis['overall_accuracy']
        except Exception as e:
            logger.error(f"Error in factual accuracy assessment: {e}")
            # Fallback to conservative score
            return 0.7

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