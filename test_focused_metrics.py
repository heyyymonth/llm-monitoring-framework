#!/usr/bin/env python3
"""
Focused On-Premise LLM Monitoring Test

This script demonstrates focused monitoring for on-premise LLM deployments,
only tracking metrics that directly impact local model performance:

RELEVANT METRICS FOR ON-PREM LLMs:
✅ Memory pressure and fragmentation (affects large model loading)
✅ GPU utilization and memory (critical for inference)  
✅ LLM process performance (CPU, memory, threads)
✅ Model loading performance (disk I/O during model loading)
✅ Thermal throttling (affects inference speed)
✅ Inference performance (tokens/sec, response times)

REMOVED IRRELEVANT METRICS:
❌ Network interface monitoring (not relevant for local models)
❌ Container metrics (not common in on-prem deployments)
❌ Generic system metrics not related to LLM performance
❌ Database metrics (unless storing conversations)
"""

import time
import random
import uuid
import psutil
from datetime import datetime
from monitoring.metrics import MetricsCollector


def analyze_focused_llm_metrics():
    """Analyze system metrics with focus on on-premise LLM performance."""
    print("🚀 FOCUSED ON-PREMISE LLM MONITORING ANALYSIS")
    print("=" * 60)
    
    # Collect current metrics
    collector = MetricsCollector()
    collector.start()
    time.sleep(2)  # Let it collect some data
    
    system_metrics = collector.get_current_system_metrics()
    if not system_metrics:
        print("❌ No metrics available")
        return
    
    # Analyze memory pressure (CRITICAL for LLM)
    print(f"\n💾 MEMORY ANALYSIS (Critical for Model Loading):")
    print(f"  Available Memory: {system_metrics.available_memory_gb:.1f}GB")
    print(f"  Memory Pressure: {'🔴 YES' if system_metrics.memory_pressure else '✅ NO'}")
    
    # Check memory fragmentation impact
    if system_metrics.memory_fragmentation:
        frag = system_metrics.memory_fragmentation
        print(f"  Memory Fragmentation: {frag.fragmentation_percent:.1f}%")
        print(f"  Swap Usage: {frag.swap_usage_mb/1024:.1f}GB")
        print(f"  Swap Pressure: {'🔴 YES' if frag.swap_pressure else '✅ NO'}")
        
        if frag.swap_pressure:
            print("  ⚠️  WARNING: Swap pressure will severely degrade LLM performance!")
        if frag.fragmentation_percent > 60:
            print("  ⚠️  WARNING: High memory fragmentation may slow model loading")
    
    # GPU analysis (CRITICAL for inference)
    print(f"\n🖥️  GPU ANALYSIS (Critical for Inference):")
    if system_metrics.gpu_count > 0:
        for i, gpu in enumerate(system_metrics.gpu_metrics):
            print(f"  GPU {i}: {gpu.get('name', 'Unknown')}")
            print(f"    Utilization: {gpu.get('utilization_percent', 0):.1f}%")
            print(f"    Memory: {gpu.get('memory_percent', 0):.1f}% ({gpu.get('memory_used_mb', 0)/1024:.1f}GB)")
            print(f"    Temperature: {gpu.get('temperature', 0):.1f}°C")
            
            # Performance warnings
            if gpu.get('utilization_percent', 0) > 95:
                print("    🔴 GPU overloaded - may cause inference delays")
            elif gpu.get('utilization_percent', 0) < 20:
                print("    ⚠️  Low GPU utilization - check model configuration")
    else:
        print("  No GPU detected - CPU-only inference")
    
    # LLM Process analysis
    print(f"\n🧠 LLM PROCESS ANALYSIS:")
    if system_metrics.llm_process_metrics:
        llm = system_metrics.llm_process_metrics
        print(f"  Process ID: {llm.pid}")
        print(f"  CPU Usage: {llm.cpu_percent:.1f}%")
        print(f"  Memory Usage: {llm.memory_rss_mb:.1f}MB")
        print(f"  Model Memory: {llm.model_memory_mb:.1f}MB")
        print(f"  Inference Threads: {llm.inference_threads}")
        print(f"  Context Switches: {llm.context_switches}")
        
        if llm.cpu_affinity:
            print(f"  CPU Affinity: {llm.cpu_affinity}")
        
        # Performance insights
        if llm.context_switches > 10000:
            print("  ⚠️  High context switches may indicate CPU contention")
        if llm.memory_rss_mb > 8000:  # >8GB
            print("  ℹ️  Large model detected - ensure sufficient RAM")
    
    # Thermal analysis (affects performance)
    print(f"\n🌡️  THERMAL ANALYSIS (Performance Impact):")
    print(f"  CPU Temperature: {system_metrics.cpu_temp_celsius:.1f}°C")
    print(f"  Thermal Throttling: {'🔴 ACTIVE' if system_metrics.thermal_throttling else '✅ NONE'}")
    
    if system_metrics.thermal_throttling:
        print("  🚨 CRITICAL: Thermal throttling will reduce inference speed!")
    elif system_metrics.cpu_temp_celsius > 80:
        print("  ⚠️  WARNING: High CPU temperature may lead to throttling")
    
    # System load analysis
    print(f"\n⚡ SYSTEM LOAD (Inference Capacity):")
    load_1m, load_5m, load_15m = system_metrics.system_load_1m, system_metrics.system_load_5m, system_metrics.system_load_15m
    cpu_cores = psutil.cpu_count()
    print(f"  Load Average: {load_1m:.2f}, {load_5m:.2f}, {load_15m:.2f}")
    print(f"  CPU Cores: {cpu_cores}")
    print(f"  Load per Core: {load_1m/cpu_cores:.2f}")
    
    if load_1m > cpu_cores * 0.8:
        print("  🔴 High system load - may impact inference latency")
    elif load_1m > cpu_cores * 0.5:
        print("  ⚠️  Moderate system load - monitor inference performance")
    else:
        print("  ✅ System load healthy for inference")
    
    collector.stop()
    
    # Calculate focused performance score
    score = calculate_focused_performance_score(system_metrics)
    print(f"\n🏥 ON-PREMISE LLM PERFORMANCE SCORE")
    print("=" * 45)
    print(f"Overall Score: {score}/100")
    
    if score >= 80:
        status = "🟢 EXCELLENT"
        advice = "System optimized for LLM inference"
    elif score >= 60:
        status = "🟡 GOOD"  
        advice = "Minor optimizations recommended"
    elif score >= 40:
        status = "🟠 FAIR"
        advice = "Performance issues may impact inference"
    else:
        status = "🔴 POOR"
        advice = "Significant performance problems detected"
    
    print(f"Status: {status}")
    print(f"Assessment: {advice}")


def calculate_focused_performance_score(metrics):
    """Calculate performance score focusing only on LLM-relevant metrics."""
    score = 100
    
    # Memory pressure (30 points - critical for LLM)
    if metrics.memory_pressure:
        score -= 30
    elif metrics.available_memory_gb < 4:
        score -= 20
    elif metrics.available_memory_gb < 8:
        score -= 10
    
    # Memory fragmentation (20 points)
    if metrics.memory_fragmentation:
        if metrics.memory_fragmentation.swap_pressure:
            score -= 20
        elif metrics.memory_fragmentation.fragmentation_percent > 70:
            score -= 15
        elif metrics.memory_fragmentation.fragmentation_percent > 50:
            score -= 10
    
    # Thermal throttling (25 points - directly affects performance)
    if metrics.thermal_throttling:
        score -= 25
    elif metrics.cpu_temp_celsius > 85:
        score -= 15
    elif metrics.cpu_temp_celsius > 75:
        score -= 10
    
    # System load (15 points)
    cpu_cores = psutil.cpu_count()
    if metrics.system_load_1m > cpu_cores * 1.2:
        score -= 15
    elif metrics.system_load_1m > cpu_cores * 0.8:
        score -= 10
    elif metrics.system_load_1m > cpu_cores * 0.6:
        score -= 5
    
    # GPU analysis (10 points)
    if metrics.gpu_count > 0:
        avg_gpu_util = sum(gpu.get('utilization_percent', 0) for gpu in metrics.gpu_metrics) / len(metrics.gpu_metrics)
        if avg_gpu_util > 95:
            score -= 10  # Overloaded
        elif avg_gpu_util < 10:
            score -= 5   # Underutilized
    
    return max(0, score)


def simulate_focused_inference_load():
    """Simulate inference requests to demonstrate focused monitoring."""
    print("\n🔄 SIMULATING LLM INFERENCE REQUESTS")
    print("=" * 50)
    
    collector = MetricsCollector()
    collector.start()
    
    # Simulate inference requests
    for i in range(3):
        request_id = str(uuid.uuid4())
        model_name = "stable-code:latest"
        prompt_tokens = random.randint(50, 200)
        completion_tokens = random.randint(20, 100)
        response_time = random.randint(800, 1500)
        
        # Track the inference
        collector.track_inference(
            request_id=request_id,
            model_name=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            response_time_ms=response_time,
            memory_usage_mb=random.randint(2000, 5000),
            gpu_utilization_percent=random.randint(60, 95),
            cache_hit=random.choice([True, False])
        )
        
        print(f"  Request {i+1}: {response_time}ms, {completion_tokens} tokens")
        time.sleep(1)
    
    # Get performance summary
    summary = collector.get_performance_summary("5m")
    
    print(f"\n📊 FOCUSED PERFORMANCE SUMMARY")
    print(f"  Total Requests: {summary.total_requests}")
    print(f"  Avg Response Time: {summary.avg_response_time_ms:.0f}ms")
    print(f"  Avg Tokens/sec: {summary.avg_tokens_per_second:.1f}")
    print(f"  Peak Memory: {summary.peak_memory_usage_mb:.0f}MB")
    print(f"  Avg GPU Utilization: {summary.avg_gpu_utilization:.1f}%")
    print(f"  Cache Hit Rate: {summary.cache_hit_rate:.1f}%")
    
    collector.stop()


def main():
    """Run focused on-premise LLM monitoring analysis."""
    print("🎯 FOCUSED ON-PREMISE LLM MONITORING")
    print("Only tracking metrics that matter for local model performance")
    print("Excluding network monitoring, containers, and generic system metrics")
    print("=" * 70)
    
    try:
        # Analyze current system for LLM readiness
        analyze_focused_llm_metrics()
        
        # Simulate inference load
        simulate_focused_inference_load()
        
        print(f"\n✅ Focused LLM monitoring analysis complete!")
        print(f"📈 Focus: Memory pressure, GPU utilization, thermal throttling")
        print(f"🚫 Excluded: Network interfaces, containers, generic metrics")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 