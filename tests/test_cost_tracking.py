"""
Comprehensive Cost Tracking Tests

This module provides extensive testing for LLM cost tracking and optimization
capabilities, including real-time cost calculation, usage analytics, budget
alerts, and optimization suggestions.
"""

import pytest
from datetime import datetime, timezone, timedelta
from typing import List, Dict
import statistics

from monitoring.cost import CostTracker
from monitoring.models import CostMetrics, CostAnalysis


class TestCostCalculation:
    """Test cost calculation accuracy for different models."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tracker = CostTracker()
    
    def test_gpt4_cost_calculation(self):
        """Test GPT-4 cost calculation accuracy."""
        # GPT-4 pricing: input=0.00003, output=0.00006
        prompt_tokens = 1000
        completion_tokens = 500
        
        cost_metrics = self.tracker.log_inference(
            model="gpt-4",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens
        )
        
        expected_cost = (1000 * 0.00003) + (500 * 0.00006)
        assert abs(cost_metrics.cost_usd - expected_cost) < 0.0001, \
            f"GPT-4 cost calculation error: got {cost_metrics.cost_usd}, expected {expected_cost}"
    
    def test_gpt35_turbo_cost_calculation(self):
        """Test GPT-3.5 Turbo cost calculation accuracy."""
        # GPT-3.5 Turbo pricing: input=0.000001, output=0.000002
        prompt_tokens = 2000
        completion_tokens = 1000
        
        cost_metrics = self.tracker.log_inference(
            model="gpt-3.5-turbo",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens
        )
        
        expected_cost = (2000 * 0.000001) + (1000 * 0.000002)
        assert abs(cost_metrics.cost_usd - expected_cost) < 0.0001, \
            f"GPT-3.5 Turbo cost calculation error: got {cost_metrics.cost_usd}, expected {expected_cost}"
    
    def test_claude3_cost_calculation(self):
        """Test Claude-3 cost calculation accuracy."""
        # Claude-3 pricing: input=0.000015, output=0.000075
        prompt_tokens = 1500
        completion_tokens = 800
        
        cost_metrics = self.tracker.log_inference(
            model="claude-3",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens
        )
        
        expected_cost = (1500 * 0.000015) + (800 * 0.000075)
        assert abs(cost_metrics.cost_usd - expected_cost) < 0.0001, \
            f"Claude-3 cost calculation error: got {cost_metrics.cost_usd}, expected {expected_cost}"
    
    def test_unknown_model_default_pricing(self):
        """Test that unknown models use default pricing."""
        prompt_tokens = 100
        completion_tokens = 50
        
        cost_metrics = self.tracker.log_inference(
            model="unknown-model-xyz",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens
        )
        
        # Default pricing: input=0.00001, output=0.00002
        expected_cost = (100 * 0.00001) + (50 * 0.00002)
        assert abs(cost_metrics.cost_usd - expected_cost) < 0.0001, \
            f"Default pricing error: got {cost_metrics.cost_usd}, expected {expected_cost}"


class TestCostTracking:
    """Test cost tracking and analytics functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tracker = CostTracker()
    
    def test_multiple_inference_logging(self):
        """Test logging multiple inferences and cost accumulation."""
        inferences = [
            {"model": "gpt-4", "prompt_tokens": 100, "completion_tokens": 50},
            {"model": "gpt-3.5-turbo", "prompt_tokens": 200, "completion_tokens": 100},
            {"model": "claude-3", "prompt_tokens": 150, "completion_tokens": 75},
        ]
        
        total_expected_cost = 0
        for inference in inferences:
            cost_metrics = self.tracker.log_inference(**inference)
            total_expected_cost += cost_metrics.cost_usd
            
            # Verify individual cost metrics
            assert cost_metrics.model_name == inference["model"]
            assert cost_metrics.prompt_tokens == inference["prompt_tokens"]
            assert cost_metrics.completion_tokens == inference["completion_tokens"]
            assert cost_metrics.total_tokens == inference["prompt_tokens"] + inference["completion_tokens"]
            assert cost_metrics.cost_usd > 0
    
    def test_cost_analysis_time_periods(self):
        """Test cost analysis for different time periods."""
        # Log some inferences
        for i in range(5):
            self.tracker.log_inference(
                model="gpt-4",
                prompt_tokens=100,
                completion_tokens=50
            )
        
        # Test different time periods
        time_periods = ["1h", "24h", "7d", "30d"]
        for period in time_periods:
            analysis = self.tracker.get_cost_analysis(period)
            assert isinstance(analysis, CostAnalysis)
            assert analysis.time_period == period
            assert analysis.total_cost_usd >= 0
            assert analysis.avg_cost_per_request >= 0


class TestCostOptimization:
    """Test cost optimization suggestions and analysis."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tracker = CostTracker()
    
    def test_optimization_suggestions_generation(self):
        """Test generation of optimization suggestions."""
        # Log multiple expensive GPT-4 operations
        for i in range(10):
            self.tracker.log_inference(
                model="gpt-4",
                prompt_tokens=2000,
                completion_tokens=1000
            )
        
        analysis = self.tracker.get_cost_analysis("24h")
        
        # Should generate optimization suggestions for expensive usage
        assert len(analysis.optimization_suggestions) > 0
        
        # Check for common optimization suggestions
        suggestions_text = " ".join(analysis.optimization_suggestions).lower()
        assert any(keyword in suggestions_text for keyword in [
            "gpt-3.5", "cheaper", "cost", "optimize", "efficient"
        ])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
