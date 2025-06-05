#!/usr/bin/env python3
"""
LLM Inference Metrics Test

Tests the minimal LLM inference-focused metrics collection.
"""

import time
import random
import uuid
from datetime import datetime
from monitoring.metrics import MetricsCollector
from monitoring.models import InferenceMetrics, ErrorMetrics


def test_llm_inference_metrics():
    """Test LLM inference-specific metrics collection."""
    print("🚀 Testing LLM Inference Metrics")
    print("=" * 50)
    
    # Initialize metrics collector
    collector = MetricsCollector()
    collector.start()
    
    try:
        # Wait for initial metrics collection
        time.sleep(2)
        
        # Test system metrics collection
        print("\n📊 Current System Metrics:")
        system_metrics = collector.get_current_system_metrics()
        
        if system_metrics:
            print(f"  CPU Usage: {system_metrics.cpu_percent:.1f}%")
            print(f"  Memory Usage: {system_metrics.memory_percent:.1f}%")
            print(f"  Memory Used: {system_metrics.memory_used_gb:.1f}GB")
            print(f"  Available Memory: {system_metrics.available_memory_gb:.1f}GB")
            print(f"  Memory Pressure: {system_metrics.memory_pressure}")
            print(f"  GPU Count: {system_metrics.gpu_count}")
            print(f"  CPU Temperature: {system_metrics.cpu_temp_celsius:.1f}°C")
            print(f"  Thermal Throttling: {system_metrics.thermal_throttling}")
            
            if system_metrics.llm_process_metrics:
                llm_proc = system_metrics.llm_process_metrics
                print(f"\n🧠 LLM Process Metrics:")
                print(f"  Process ID: {llm_proc.pid}")
                print(f"  CPU Usage: {llm_proc.cpu_percent:.1f}%")
                print(f"  RSS Memory: {llm_proc.memory_rss_mb:.1f}MB")
                print(f"  Model Memory: {llm_proc.model_memory_mb:.1f}MB")
                print(f"  Inference Threads: {llm_proc.inference_threads}")
            
            if system_metrics.gpu_count > 0 and system_metrics.gpu_metrics:
                print(f"\n🎮 GPU Metrics:")
                for gpu in system_metrics.gpu_metrics:
                    print(f"  GPU {gpu.get('gpu_id', 0)}: {gpu.get('name', 'Unknown')}")
                    print(f"    Utilization: {gpu.get('utilization_percent', 0):.1f}%")
                    print(f"    Memory: {gpu.get('memory_percent', 0):.1f}%")
                    print(f"    Temperature: {gpu.get('temperature', 0):.1f}°C")
        else:
            print("  No system metrics available")
        
        # Simulate LLM inference requests
        print(f"\n🔄 Simulating LLM Inference Requests...")
        
        inference_results = []
        for i in range(5):
            request_id = str(uuid.uuid4())
            
            # Simulate inference queue
            collector.increment_queue_pending()
            queue_wait_time = random.uniform(10, 100)  # ms
            collector.log_wait_time(queue_wait_time)
            
            # Simulate processing
            collector.decrement_queue_pending()
            collector.increment_queue_processing()
            
            # Simulate inference completion
            prompt_tokens = random.randint(50, 200)
            completion_tokens = random.randint(20, 150)
            response_time = random.uniform(500, 2000)  # ms
            processing_time = response_time - queue_wait_time
            tokens_per_second = (prompt_tokens + completion_tokens) / (response_time / 1000)
            
            # Create inference metrics
            inference_metrics = InferenceMetrics(
                request_id=request_id,
                model_name="test-llm-model",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                response_time_ms=response_time,
                queue_time_ms=queue_wait_time,
                processing_time_ms=processing_time,
                tokens_per_second=tokens_per_second,
                prompt_length=prompt_tokens * 4,  # Rough estimate
                response_length=completion_tokens * 4,
                success=True,
                model_version="v1.0",
                temperature=0.7,
                max_tokens=256,
                # LLM-specific metrics
                memory_peak_mb=random.uniform(1000, 3000),
                gpu_utilization_percent=random.uniform(60, 95),
                cache_hit=random.choice([True, False]),
                batch_size=1,
                sequence_length=prompt_tokens + completion_tokens
            )
            
            collector.log_inference(inference_metrics)
            collector.decrement_queue_processing()
            
            inference_results.append(inference_metrics)
            print(f"  Request {i+1}: {response_time:.0f}ms, {tokens_per_second:.1f} tokens/sec")
        
        # Simulate an error
        error_metrics = ErrorMetrics(
            request_id=str(uuid.uuid4()),
            error_type="ModelLoadError",
            error_message="Insufficient GPU memory for model loading",
            model_name="test-llm-model",
            endpoint="/v1/chat/completions",
            memory_usage_at_error_mb=7800,
            gpu_memory_usage_mb=10240,
            tokens_processed_before_error=0
        )
        collector.log_error(error_metrics)
        
        # Get performance summary
        print(f"\n📈 Performance Summary (1h):")
        summary = collector.get_performance_summary("1h")
        print(f"  Total Requests: {summary.total_requests}")
        print(f"  Success Rate: {(summary.successful_requests/summary.total_requests)*100:.1f}%" if summary.total_requests > 0 else "  Success Rate: N/A")
        print(f"  Avg Response Time: {summary.avg_response_time_ms:.0f}ms")
        print(f"  P95 Response Time: {summary.p95_response_time_ms:.0f}ms")
        print(f"  Avg Tokens/sec: {summary.avg_tokens_per_second:.1f}")
        print(f"  Total Tokens: {summary.total_tokens_processed}")
        print(f"  Avg Memory Usage: {summary.avg_memory_usage_mb:.1f}MB")
        print(f"  Peak Memory Usage: {summary.peak_memory_usage_mb:.1f}MB")
        print(f"  Avg GPU Utilization: {summary.avg_gpu_utilization:.1f}%")
        print(f"  Cache Hit Rate: {summary.cache_hit_rate:.1f}%")
        print(f"  Avg Queue Time: {summary.avg_queue_time_ms:.1f}ms")
        
        # Health scoring based on LLM performance
        health_score = calculate_llm_health_score(summary, system_metrics)
        print(f"\n🏥 LLM Health Score: {health_score}/100")
        
        if health_score >= 80:
            print("   Status: EXCELLENT - LLM performing optimally")
        elif health_score >= 60:
            print("   Status: GOOD - LLM performing well")
        elif health_score >= 40:
            print("   Status: FAIR - Some performance issues")
        else:
            print("   Status: POOR - Significant performance issues")
        
        # Performance recommendations
        print(f"\n💡 LLM Performance Recommendations:")
        recommendations = generate_llm_recommendations(summary, system_metrics)
        for rec in recommendations:
            print(f"  • {rec}")
        
        print(f"\n✅ LLM Inference Metrics Test Complete!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        collector.stop()


def calculate_llm_health_score(summary, system_metrics):
    """Calculate health score specific to LLM inference performance."""
    score = 100
    
    # Response time penalty
    if summary.avg_response_time_ms > 3000:
        score -= 20
    elif summary.avg_response_time_ms > 1500:
        score -= 10
    
    # Error rate penalty
    if summary.error_rate > 5:
        score -= 25
    elif summary.error_rate > 1:
        score -= 10
    
    # Memory pressure penalty
    if system_metrics and system_metrics.memory_pressure:
        score -= 15
    elif system_metrics and system_metrics.memory_percent > 85:
        score -= 10
    
    # Thermal throttling penalty
    if system_metrics and system_metrics.thermal_throttling:
        score -= 20
    
    # GPU utilization (too low or too high can be bad)
    if summary.avg_gpu_utilization > 95:
        score -= 15  # Potential bottleneck
    elif summary.avg_gpu_utilization < 30:
        score -= 10  # Underutilization
    
    # Queue time penalty
    if summary.avg_queue_time_ms > 500:
        score -= 15
    elif summary.avg_queue_time_ms > 200:
        score -= 5
    
    return max(0, min(100, score))


def generate_llm_recommendations(summary, system_metrics):
    """Generate LLM-specific performance recommendations."""
    recommendations = []
    
    if summary.avg_response_time_ms > 2000:
        recommendations.append("High response time detected - consider model optimization or GPU upgrade")
    
    if summary.error_rate > 1:
        recommendations.append("Error rate above threshold - check model loading and memory allocation")
    
    if system_metrics and system_metrics.memory_pressure:
        recommendations.append("Memory pressure detected - consider reducing model size or adding RAM")
    
    if system_metrics and system_metrics.thermal_throttling:
        recommendations.append("CPU thermal throttling - improve cooling or reduce CPU load")
    
    if summary.avg_gpu_utilization > 90:
        recommendations.append("GPU running at high utilization - consider adding more GPUs or optimizing batching")
    
    if summary.avg_gpu_utilization < 40:
        recommendations.append("Low GPU utilization - check for CPU bottlenecks or increase batch size")
    
    if summary.avg_queue_time_ms > 300:
        recommendations.append("High queue times - consider scaling inference workers or optimizing model")
    
    if summary.cache_hit_rate < 30:
        recommendations.append("Low cache hit rate - consider improving caching strategy for repeated requests")
    
    if not recommendations:
        recommendations.append("LLM performance is optimal - no immediate improvements needed")
    
    return recommendations


if __name__ == "__main__":
    test_llm_inference_metrics() 