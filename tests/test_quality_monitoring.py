"""
Test suite for LLM Quality & Safety Monitoring

Tests the core functionality of quality assessment, safety evaluation,
and cost tracking for LLM monitoring applications.
"""

import pytest
import os
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

from monitoring.quality import QualityMonitor, HallucinationDetector, SafetyEvaluator, QualityAssessor
from monitoring.cost import CostTracker
from monitoring.models import SafetyFlag, QualityMetrics, SafetyAssessment, CostMetrics


class TestQualityMonitor:
    """Test the main QualityMonitor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.monitor = QualityMonitor()
    
    def test_evaluate_response_basic(self):
        """Test basic response evaluation."""
        prompt = "What is the capital of France?"
        response = "The capital of France is Paris."
        
        trace = self.monitor.evaluate_response(prompt, response)
        
        assert trace.prompt == prompt
        assert trace.response == response
        assert trace.trace_id is not None
        assert len(trace.trace_id) == 12  # MD5 hash truncated to 12 chars
        assert trace.quality_metrics.overall_quality > 0
        assert trace.safety_assessment.is_safe is not None
        assert trace.cost_metrics.total_tokens > 0
    
    def test_evaluate_response_with_quality_checks(self):
        """Test response evaluation with all quality checks enabled."""
        prompt = "Explain quantum computing"
        response = "Quantum computing uses quantum bits (qubits) to process information in fundamentally different ways than classical computers."
        
        trace = self.monitor.evaluate_response(
            prompt=prompt,
            response=response,
            check_hallucination=True,
            check_toxicity=True,
            check_bias=True,
            check_pii=True
        )
        
        # Check quality metrics
        assert 0 <= trace.quality_metrics.semantic_similarity <= 1
        assert 0 <= trace.quality_metrics.factual_accuracy <= 1
        assert 0 <= trace.quality_metrics.response_relevance <= 1
        assert 0 <= trace.quality_metrics.coherence_score <= 1
        assert 0 <= trace.quality_metrics.overall_quality <= 1
        
        # Check safety assessment
        assert isinstance(trace.safety_assessment.is_safe, bool)
        assert 0 <= trace.safety_assessment.safety_score <= 1
        assert isinstance(trace.safety_assessment.flags, list)
    
    def test_evaluate_response_with_safety_violations(self):
        """Test response evaluation that should trigger safety flags."""
        prompt = "Tell me about violence"
        response = "Violence is harmful and dangerous behavior that causes physical harm to others."
        
        trace = self.monitor.evaluate_response(prompt, response, model_name="test-model")
        
        # Should detect some toxicity due to keywords
        if SafetyFlag.TOXICITY in trace.safety_assessment.flags:
            assert trace.safety_assessment.safety_score < 1.0


class TestHallucinationDetector:
    """Test hallucination detection functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = HallucinationDetector()
    
    def test_detect_hallucination_explicit_patterns(self):
        """Test detection of explicit hallucination patterns."""
        response = "I apologize, but I don't actually have access to real-time data."
        
        is_hallucination, score, details = self.detector.detect_hallucination(response)
        
        assert is_hallucination == True
        assert score > 0.5
        assert "pattern_matches" in details
        assert len(details["pattern_matches"]) > 0
    
    def test_detect_hallucination_confidence_markers(self):
        """Test detection of unsupported confidence claims."""
        response = "I am absolutely certain that this is 100% correct without a doubt."
        
        is_hallucination, score, details = self.detector.detect_hallucination(response)
        
        assert score > 0
        assert "confidence_markers" in details
        assert len(details["confidence_markers"]) > 0
    
    def test_detect_hallucination_clean_response(self):
        """Test that clean responses don't trigger hallucination detection."""
        response = "The capital of France is Paris, which is located in the northern part of the country."
        
        is_hallucination, score, details = self.detector.detect_hallucination(response)
        
        assert is_hallucination == False
        assert score < 0.5


class TestSafetyEvaluator:
    """Test safety evaluation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.evaluator = SafetyEvaluator()
    
    def test_evaluate_safety_clean_response(self):
        """Test safety evaluation of clean response."""
        response = "The weather today is sunny and pleasant."
        
        assessment = self.evaluator.evaluate_safety(response)
        
        assert assessment.is_safe == True
        assert assessment.safety_score > 0.7
        assert len(assessment.flags) == 0
    
    def test_evaluate_safety_toxic_content(self):
        """Test safety evaluation of potentially toxic content."""
        response = "This content contains harmful and offensive language."
        
        assessment = self.evaluator.evaluate_safety(response)
        
        # Should detect toxicity
        assert assessment.safety_score < 1.0
        if SafetyFlag.TOXICITY in assessment.flags:
            assert "toxicity_score" in assessment.details
    
    def test_evaluate_safety_pii_detection(self):
        """Test PII detection functionality."""
        response = "My email is john.doe@example.com and my SSN is 123-45-6789."
        
        assessment = self.evaluator.evaluate_safety(response)
        
        # Should detect PII
        assert SafetyFlag.PII_LEAK in assessment.flags
        assert assessment.details["pii_detected"] == True
    
    def test_evaluate_safety_selective_checks(self):
        """Test safety evaluation with selective checks."""
        response = "Some potentially biased content about groups."
        
        # Only check toxicity, not bias
        assessment = self.evaluator.evaluate_safety(
            response, 
            check_toxicity=True,
            check_bias=False,
            check_hallucination=False,
            check_pii=False
        )
        
        # Should only have toxicity-related details
        assert "toxicity_score" in assessment.details
        assert "bias_score" not in assessment.details


class TestQualityAssessor:
    """Test quality assessment functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.assessor = QualityAssessor()
    
    @patch('monitoring.quality.anthropic.Anthropic')
    @patch('monitoring.quality.openai.OpenAI')
    def test_assess_quality_relevant_response(self, mock_openai_client, mock_anthropic_client):
        """Test quality assessment of relevant response."""
        # Mock the API clients
        mock_openai_client.return_value.chat.completions.create.return_value = MagicMock(choices=[MagicMock(message=MagicMock(content='{"score": 0.9, "justification": "Mocked response"}'))])
        mock_anthropic_client.return_value.messages.create.return_value = MagicMock(content=[MagicMock(text='{"score": 0.9, "justification": "Mocked response"}')])

        prompt = "What is machine learning?"
        response = "Machine learning is a subset of artificial intelligence that enables computers to learn from data."
        
        quality = self.assessor.assess_quality(prompt, response, model_name="test-model")
        
        assert 0 <= quality.semantic_similarity <= 1
        assert 0 <= quality.factual_accuracy <= 1
        assert 0 <= quality.response_relevance <= 1
        assert 0 <= quality.coherence_score <= 1
        assert 0 <= quality.overall_quality <= 1
        
        # Should have good relevance due to keyword overlap
        assert quality.response_relevance > 0.3
    
    @patch('monitoring.quality.anthropic.Anthropic')
    @patch('monitoring.quality.openai.OpenAI')
    def test_assess_quality_irrelevant_response(self, mock_openai_client, mock_anthropic_client):
        """Test quality assessment of irrelevant response."""
        mock_openai_client.return_value.chat.completions.create.return_value = MagicMock(choices=[MagicMock(message=MagicMock(content='{"score": 0.1, "justification": "Mocked irrelevant"}'))])
        mock_anthropic_client.return_value.messages.create.return_value = MagicMock(content=[MagicMock(text='{"score": 0.1, "justification": "Mocked irrelevant"}')])

        prompt = "What is machine learning?"
        response = "I like pizza and ice cream on sunny days."
        
        quality = self.assessor.assess_quality(prompt, response, model_name="test-model")
        
        # Should have low relevance
        assert quality.response_relevance < 0.5
        assert quality.overall_quality < 0.7
    
    @patch('monitoring.quality.anthropic.Anthropic')
    @patch('monitoring.quality.openai.OpenAI')
    def test_assess_quality_response_length(self, mock_openai_client, mock_anthropic_client):
        """Test that response_length is correctly calculated."""
        mock_openai_client.return_value.chat.completions.create.return_value = MagicMock(choices=[MagicMock(message=MagicMock(content='{"score": 0.8, "justification": "Mocked response"}'))])
        mock_anthropic_client.return_value.messages.create.return_value = MagicMock(content=[MagicMock(text='{"score": 0.8, "justification": "Mocked response"}')])

        prompt = "Hello"
        response = "This is a test response."
        
        quality = self.assessor.assess_quality(prompt, response, model_name="test-model")
        
        assert quality.response_length == len(response)
    
    @patch('monitoring.quality.anthropic.Anthropic')
    @patch('monitoring.quality.openai.OpenAI')
    def test_assess_quality_short_response(self, mock_openai_client, mock_anthropic_client):
        """Test quality assessment of very short response."""
        mock_openai_client.return_value.chat.completions.create.return_value = MagicMock(choices=[MagicMock(message=MagicMock(content='{"score": 0.2, "justification": "Mocked short"}'))])
        mock_anthropic_client.return_value.messages.create.return_value = MagicMock(content=[MagicMock(text='{"score": 0.2, "justification": "Mocked short"}')])

        prompt = "Explain quantum physics in detail."
        response = "Yes."
        
        quality = self.assessor.assess_quality(prompt, response, model_name="test-model")
        
        # Should have low relevance due to short length
        assert quality.response_relevance < 0.5

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key', 'ANTHROPIC_API_KEY': 'test-key'})
    @patch('monitoring.quality.anthropic.Anthropic')
    @patch('monitoring.quality.openai.OpenAI')
    def test_assess_relevance_provider_selection(self, mock_openai_client, mock_anthropic_client):
        """Test that the correct LLM provider is selected based on the model name."""
        mock_openai_instance = mock_openai_client.return_value
        mock_openai_instance.chat.completions.create.return_value = MagicMock(choices=[MagicMock(message=MagicMock(content='{"score": 0.9}'))])

        mock_anthropic_instance = mock_anthropic_client.return_value
        mock_anthropic_instance.messages.create.return_value = MagicMock(content=[MagicMock(text='{"score": 0.9}')])

        # Test with an OpenAI model - using the new enhanced relevance assessment
        self.assessor.relevance_assessor._llm_judge_relevance(
            "A prompt", "A response", "gpt-4-turbo", "Test evaluation criteria"
        )
        mock_openai_instance.chat.completions.create.assert_called_once()
        mock_anthropic_instance.messages.create.assert_not_called()

        # Reset mocks
        mock_openai_instance.chat.completions.create.reset_mock()
        mock_anthropic_instance.messages.create.reset_mock()

        # Test with an Anthropic (Claude) model - using the new enhanced relevance assessment
        self.assessor.relevance_assessor._llm_judge_relevance(
            "A prompt", "A response", "claude-3-haiku-20240307", "Test evaluation criteria"
        )
        mock_openai_instance.chat.completions.create.assert_not_called()
        mock_anthropic_instance.messages.create.assert_called_once()


class TestCostTracker:
    """Test cost tracking functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tracker = CostTracker()
    
    def test_log_inference_basic(self):
        """Test basic inference logging."""
        cost_metrics = self.tracker.log_inference(
            model="gpt-4",
            prompt_tokens=100,
            completion_tokens=50
        )
        
        assert cost_metrics.model_name == "gpt-4"
        assert cost_metrics.prompt_tokens == 100
        assert cost_metrics.completion_tokens == 50
        assert cost_metrics.total_tokens == 150
        assert cost_metrics.cost_usd > 0
        assert isinstance(cost_metrics.timestamp, datetime)
    
    def test_log_inference_custom_cost(self):
        """Test inference logging with custom cost."""
        cost_metrics = self.tracker.log_inference(
            model="custom-model",
            prompt_tokens=200,
            completion_tokens=100,
            cost_per_token=0.00001
        )
        
        expected_cost = 300 * 0.00001  # (200 + 100) * cost_per_token
        assert cost_metrics.cost_usd == expected_cost
    
    def test_get_cost_analysis_empty(self):
        """Test cost analysis with no data."""
        analysis = self.tracker.get_cost_analysis("24h")
        
        assert analysis.time_period == "24h"
        assert analysis.total_cost_usd == 0.0
        assert analysis.avg_cost_per_request == 0.0
        assert len(analysis.most_expensive_operations) == 0
        assert len(analysis.optimization_suggestions) == 0
        assert analysis.projected_monthly_cost == 0.0
    
    def test_get_cost_analysis_with_data(self):
        """Test cost analysis with logged data."""
        # Log several inferences
        for i in range(5):
            self.tracker.log_inference(
                model="gpt-3.5-turbo",
                prompt_tokens=100 + i * 10,
                completion_tokens=50 + i * 5
            )
        
        analysis = self.tracker.get_cost_analysis("24h")
        
        assert analysis.total_cost_usd > 0
        assert analysis.avg_cost_per_request > 0
        assert analysis.projected_monthly_cost > 0
    
    def test_calculate_cost_known_model(self):
        """Test cost calculation for known model."""
        cost = self.tracker._calculate_cost("gpt-4", 100, 50)
        
        # GPT-4 pricing: input=0.00003, output=0.00006
        expected_cost = (100 * 0.00003) + (50 * 0.00006)
        assert cost == expected_cost
    
    def test_calculate_cost_unknown_model(self):
        """Test cost calculation for unknown model."""
        cost = self.tracker._calculate_cost("unknown-model", 100, 50)
        
        # Default pricing: input=0.00001, output=0.00002
        expected_cost = (100 * 0.00001) + (50 * 0.00002)
        assert cost == expected_cost


def test_integration_quality_and_cost():
    """Integration test combining quality monitoring and cost tracking."""
    monitor = QualityMonitor()
    tracker = CostTracker()
    
    prompt = "What are the benefits of renewable energy?"
    response = "Renewable energy sources like solar and wind power provide clean electricity, reduce carbon emissions, and help combat climate change."
    
    # Monitor quality
    trace = monitor.evaluate_response(prompt, response, model_name="gpt-4")
    
    # Track cost separately (would normally be integrated)
    cost_metrics = tracker.log_inference(
        model="gpt-4",
        prompt_tokens=trace.cost_metrics.prompt_tokens,
        completion_tokens=trace.cost_metrics.completion_tokens
    )
    
    assert trace.quality_metrics.overall_quality > 0
    assert trace.safety_assessment.is_safe == True
    assert cost_metrics.cost_usd > 0
    assert cost_metrics.model_name == "gpt-4"


if __name__ == "__main__":
    pytest.main([__file__]) 