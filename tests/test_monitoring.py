#!/usr/bin/env python3
"""
Minimalist LLM monitoring framework core tests.
"""

import pytest
import time
from datetime import datetime, timezone
from monitoring.metrics import MetricsCollector
from monitoring.models import InferenceMetrics, SystemMetrics


class TestMetricsCollector:
    """Test MetricsCollector functionality."""
    
    def test_init(self):
        """Test MetricsCollector initialization."""
        collector = MetricsCollector()
        assert collector._running is False
        assert len(collector._system_metrics) == 0
        assert len(collector._inference_metrics) == 0
    
    def test_system_metrics_collection(self):
        """Test system metrics collection."""
        collector = MetricsCollector()
        metrics = collector._collect_system_metrics()
        
        assert metrics is not None
        assert isinstance(metrics, SystemMetrics)
        assert metrics.cpu_percent >= 0
        assert metrics.memory_percent >= 0
        assert metrics.memory_used_gb >= 0
        assert metrics.memory_available_gb >= 0
        assert metrics.memory_total_gb >= 0
    
    def test_llm_process_metrics(self):
        """Test LLM process metrics collection."""
        collector = MetricsCollector()
        process_metrics = collector._collect_llm_process_metrics()
        
        assert process_metrics is not None
        assert process_metrics.pid > 0
        assert process_metrics.memory_rss_mb > 0

    
    def test_inference_logging(self):
        """Test inference metrics logging."""
        collector = MetricsCollector()
        
        inference = InferenceMetrics(
            request_id="test-123",
            model_name="test-model",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            response_time_ms=500.0,
            tokens_per_second=300.0,
            success=True
        )
        
        collector.log_inference(inference)
        
        recent = collector.get_recent_inference_metrics(10)
        assert len(recent) == 1
        assert recent[0].request_id == "test-123"
        assert recent[0].success is True
    
    def test_queue_management(self):
        """Test queue management functionality."""
        collector = MetricsCollector()
        
        # Test incrementing
        collector.increment_queue_pending()
        collector.increment_queue_processing()
        
        stats = collector.get_stats()
        assert stats["pending_requests"] == 1
        assert stats["processing_requests"] == 1
        
        # Test decrementing
        collector.decrement_queue_pending()
        collector.decrement_queue_processing()
        
        stats = collector.get_stats()
        assert stats["pending_requests"] == 0
        assert stats["processing_requests"] == 0
    
    def test_performance_summary(self):
        """Test performance summary generation."""
        collector = MetricsCollector()
        
        # Add test inferences
        for i in range(3):
            inference = InferenceMetrics(
                request_id=f"test-{i}",
                model_name="test-model",
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
                response_time_ms=400.0 + i * 100,
                tokens_per_second=250.0,
                success=True
            )
            collector.log_inference(inference)
        
        summary = collector.get_performance_summary("1h")
        assert summary.total_requests == 3
        assert summary.successful_requests == 3
        assert summary.failed_requests == 0
        assert summary.error_rate == 0.0
        assert summary.avg_response_time_ms > 0
    
    def test_stats_collection(self):
        """Test stats collection."""
        collector = MetricsCollector()
        
        inference = InferenceMetrics(
            request_id="stats-test",
            model_name="test-model",
            prompt_tokens=50,
            completion_tokens=25,
            total_tokens=75,
            response_time_ms=300.0,
            tokens_per_second=250.0,
            success=True
        )
        collector.log_inference(inference)
        
        stats = collector.get_stats()
        assert "uptime_seconds" in stats
        assert "total_inference_requests" in stats
        assert stats["total_inference_requests"] == 1
        assert stats["completed_requests"] == 1
        assert stats["failed_requests"] == 0


class TestInferenceMetrics:
    """Test InferenceMetrics model."""
    
    def test_inference_metrics_creation(self):
        """Test creating InferenceMetrics."""
        inference = InferenceMetrics(
            request_id="test-123",
            model_name="test-model",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            response_time_ms=500.0,
            tokens_per_second=300.0,
            success=True
        )
        
        assert inference.request_id == "test-123"
        assert inference.model_name == "test-model"
        assert inference.total_tokens == 150
        assert inference.success is True
    
    def test_inference_metrics_defaults(self):
        """Test InferenceMetrics default values."""
        inference = InferenceMetrics(
            request_id="test-123",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            response_time_ms=500.0
        )
        
        assert inference.success is True
        assert inference.tokens_per_second == 0.0



class TestSystemMetrics:
    """Test SystemMetrics model."""
    
    def test_system_metrics_creation(self):
        """Test creating SystemMetrics."""
        metrics = SystemMetrics(
            cpu_percent=25.5,
            memory_percent=60.0,
            memory_used_gb=4.0,
            memory_available_gb=2.0,
            memory_total_gb=8.0
        )
        
        assert metrics.cpu_percent == 25.5
        assert metrics.memory_percent == 60.0
        assert metrics.memory_used_gb == 4.0
        assert metrics.memory_available_gb == 2.0
        assert isinstance(metrics.timestamp, datetime)


if __name__ == "__main__":
    # Run tests without pytest
    import sys
    
    print("🚀 Running minimalist LLM monitoring tests\n")
    
    test_classes = [TestMetricsCollector, TestInferenceMetrics, TestSystemMetrics]
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"📋 Testing {test_class.__name__}")
        instance = test_class()
        
        for attr_name in dir(instance):
            if attr_name.startswith('test_'):
                total_tests += 1
                try:
                    getattr(instance, attr_name)()
                    print(f"   ✅ {attr_name}")
                    passed_tests += 1
                except Exception as e:
                    print(f"   ❌ {attr_name}: {e}")
        print()
    
    print(f"📊 Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("⚠️  Some tests failed")
        sys.exit(1) 