#!/usr/bin/env python3
"""
Minimalist LLM monitoring framework tests.
Tests essential functionality only.
"""

import time
import psutil
from datetime import datetime, timezone
from monitoring.metrics import MetricsCollector
from monitoring.models import InferenceMetrics, SystemMetrics


def test_metrics_collector():
    """Test basic metrics collection functionality."""
    print("🧪 Testing MetricsCollector...")
    
    collector = MetricsCollector()
    
    # Test basic collection
    system_metrics = collector._collect_system_metrics()
    assert system_metrics is not None
    assert system_metrics.cpu_percent >= 0
    assert system_metrics.memory_percent >= 0
    
    print(f"   ✅ System metrics: CPU {system_metrics.cpu_percent}%, Memory {system_metrics.memory_percent}%")
    

def test_inference_tracking():
    """Test inference metrics tracking."""
    print("🧪 Testing inference tracking...")
    
    collector = MetricsCollector()
    
    # Create test inference metric
    inference = InferenceMetrics(
        request_id="test-123",
        model_name="test-model",
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        response_time_ms=500.0,
        tokens_per_second=300.0,
        memory_peak_mb=256.0,
        success=True
    )
    
    # Log the inference
    collector.log_inference(inference)
    
    # Verify it was logged
    recent = collector.get_recent_inference_metrics(10)
    assert len(recent) == 1
    assert recent[0].request_id == "test-123"
    assert recent[0].model_name == "test-model"
    
    print(f"   ✅ Inference tracked: {inference.model_name}, {inference.tokens_per_second} tok/s")


def test_performance_summary():
    """Test performance summary generation."""
    print("🧪 Testing performance summary...")
    
    collector = MetricsCollector()
    
    # Add multiple test inferences
    for i in range(5):
        inference = InferenceMetrics(
            request_id=f"test-{i}",
            model_name="test-model",
            prompt_tokens=100 + i * 10,
            completion_tokens=50 + i * 5,
            total_tokens=150 + i * 15,
            response_time_ms=400.0 + i * 50,
            tokens_per_second=250.0 + i * 25,
            memory_peak_mb=200.0 + i * 20,
            success=True
        )
        collector.log_inference(inference)
        
        # Get performance summary
        summary = collector.get_performance_summary("1h")
    assert summary.total_requests == 5
    assert summary.successful_requests == 5
    assert summary.failed_requests == 0
    assert summary.error_rate == 0.0
    assert summary.avg_response_time_ms > 0
    
    print(f"   ✅ Performance summary: {summary.total_requests} requests, {summary.avg_response_time_ms:.1f}ms avg")


def test_queue_management():
    """Test queue management."""
    print("🧪 Testing queue management...")
    
    collector = MetricsCollector()
    
    # Test queue operations
    collector.increment_queue_pending()
    collector.increment_queue_pending()
    collector.increment_queue_processing()
    
    stats = collector.get_stats()
    assert stats["pending_requests"] == 2
    assert stats["processing_requests"] == 1
    
    collector.decrement_queue_pending()
    collector.decrement_queue_processing()
    
    stats = collector.get_stats()
    assert stats["pending_requests"] == 1
    assert stats["processing_requests"] == 0
    
    print(f"   ✅ Queue management working correctly")


def test_llm_process_metrics():
    """Test LLM process metrics collection."""
    print("🧪 Testing LLM process metrics...")
    
    collector = MetricsCollector()
    process_metrics = collector._collect_llm_process_metrics()
    
    assert process_metrics is not None
    assert process_metrics.pid > 0
    assert process_metrics.memory_rss_mb > 0
    assert process_metrics.inference_threads > 0
    
    print(f"   ✅ LLM process: PID {process_metrics.pid}, {process_metrics.memory_rss_mb:.1f}MB, {process_metrics.inference_threads} threads")


def test_stats_collection():
    """Test basic stats collection."""
    print("🧪 Testing stats collection...")
    
    collector = MetricsCollector()
    
    # Add some test data
    for i in range(3):
        inference = InferenceMetrics(
            request_id=f"stats-test-{i}",
            model_name="stats-model",
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
    assert stats["total_inference_requests"] == 3
    assert stats["completed_requests"] == 3
    
    print(f"   ✅ Stats: {stats['total_inference_requests']} inferences, {stats['uptime_seconds']:.1f}s uptime")


def run_all_tests():
    """Run all tests."""
    print("🚀 Starting minimalist LLM monitoring framework tests\n")
    
    tests = [
        test_metrics_collector,
        test_inference_tracking,
        test_performance_summary,
        test_queue_management,
        test_llm_process_metrics,
        test_stats_collection
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
            failed += 1
        print()
    
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Framework is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the output above.")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1) 