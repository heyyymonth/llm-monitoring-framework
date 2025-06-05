#!/usr/bin/env python3
"""
Enhanced System-Level Metrics Test for LLM Performance Monitoring

This script demonstrates comprehensive system-level metrics collection
that can identify root causes of LLM inference performance degradation.
"""

import time
import random
import uuid
from datetime import datetime
from monitoring.metrics import MetricsCollector
from monitoring.models import InferenceMetrics, ErrorMetrics


def display_system_health_analysis(system_metrics):
    """Analyze and display system health with LLM performance context."""
    print("\n🔍 SYSTEM HEALTH ANALYSIS FOR LLM PERFORMANCE")
    print("=" * 60)
    
    # Core system metrics
    print(f"📊 Core System Metrics:")
    print(f"  CPU Usage: {system_metrics.cpu_percent:.1f}%")
    print(f"  Memory Usage: {system_metrics.memory_percent:.1f}%")
    print(f"  Memory Used: {system_metrics.memory_used_gb:.1f}GB")
    print(f"  Available Memory: {system_metrics.available_memory_gb:.1f}GB")
    print(f"  Memory Pressure: {'⚠️  YES' if system_metrics.memory_pressure else '✅ NO'}")
    
    # System load and performance
    print(f"\n⚡ System Performance:")
    print(f"  Load Average (1m): {system_metrics.system_load_1m:.2f}")
    print(f"  Load Average (5m): {system_metrics.system_load_5m:.2f}")
    print(f"  Load Average (15m): {system_metrics.system_load_15m:.2f}")
    print(f"  Uptime: {system_metrics.uptime_seconds/3600:.1f} hours")
    
    # Thermal analysis
    print(f"\n🌡️  Thermal Status:")
    print(f"  CPU Temperature: {system_metrics.cpu_temp_celsius:.1f}°C")
    print(f"  Thermal Throttling: {'⚠️  ACTIVE' if system_metrics.thermal_throttling else '✅ NONE'}")
    
    if system_metrics.thermal_zones:
        print(f"  Thermal Zones:")
        for zone, temp in system_metrics.thermal_zones.items():
            status = "🔥" if temp > 80 else "⚠️ " if temp > 70 else "✅"
            print(f"    {zone}: {temp:.1f}°C {status}")
    
    # Disk I/O analysis (critical for model loading)
    if system_metrics.disk_io_metrics:
        print(f"\n💾 Disk I/O Metrics (Model Loading Performance):")
        total_read_rate = 0
        total_write_rate = 0
        max_latency = 0
        
        for disk in system_metrics.disk_io_metrics:
            read_mb_s = disk.read_bytes_per_sec / (1024 * 1024)
            write_mb_s = disk.write_bytes_per_sec / (1024 * 1024)
            total_read_rate += read_mb_s
            total_write_rate += write_mb_s
            max_latency = max(max_latency, disk.read_latency_ms, disk.write_latency_ms)
            
            print(f"  {disk.device}:")
            print(f"    Read: {read_mb_s:.1f} MB/s, Write: {write_mb_s:.1f} MB/s")
            print(f"    Read IOPS: {disk.read_iops:.0f}, Write IOPS: {disk.write_iops:.0f}")
            if disk.read_latency_ms > 0 or disk.write_latency_ms > 0:
                print(f"    Latency: R={disk.read_latency_ms:.1f}ms, W={disk.write_latency_ms:.1f}ms")
        
        print(f"  📈 Total Read Rate: {total_read_rate:.1f} MB/s")
        print(f"  📈 Total Write Rate: {total_write_rate:.1f} MB/s")
        
        # Analyze disk performance for model loading
        if total_read_rate < 50:  # Less than 50 MB/s
            print(f"  ⚠️  WARNING: Low disk read rate may slow model loading")
        if max_latency > 10:  # More than 10ms latency
            print(f"  ⚠️  WARNING: High disk latency ({max_latency:.1f}ms) may impact performance")
    
    # Network I/O analysis (distributed inference)
    if system_metrics.network_metrics:
        print(f"\n🌐 Network I/O Metrics (Distributed Inference):")
        total_rx_rate = 0
        total_tx_rate = 0
        total_errors = 0
        
        for net in system_metrics.network_metrics:
            rx_mb_s = net.bytes_recv_per_sec / (1024 * 1024)
            tx_mb_s = net.bytes_sent_per_sec / (1024 * 1024)
            total_rx_rate += rx_mb_s
            total_tx_rate += tx_mb_s
            total_errors += net.errors_per_sec + net.drops_per_sec
            
            if rx_mb_s > 0.1 or tx_mb_s > 0.1:  # Only show active interfaces
                print(f"  {net.interface}:")
                print(f"    RX: {rx_mb_s:.2f} MB/s, TX: {tx_mb_s:.2f} MB/s")
                print(f"    Packets: RX={net.packets_recv_per_sec:.0f}/s, TX={net.packets_sent_per_sec:.0f}/s")
                if net.errors_per_sec > 0 or net.drops_per_sec > 0:
                    print(f"    ⚠️  Errors: {net.errors_per_sec:.1f}/s, Drops: {net.drops_per_sec:.1f}/s")
        
        if total_errors > 1:
            print(f"  ⚠️  WARNING: Network errors/drops detected ({total_errors:.1f}/s)")
    
    # Memory fragmentation analysis
    if system_metrics.memory_fragmentation:
        frag = system_metrics.memory_fragmentation
        print(f"\n🧩 Memory Fragmentation (Model Loading Impact):")
        print(f"  Largest Free Block: {frag.largest_free_block_mb:.1f} MB")
        print(f"  Fragmentation: {frag.fragmentation_percent:.1f}%")
        print(f"  Swap Usage: {frag.swap_usage_mb:.1f} MB")
        print(f"  Swap Pressure: {'⚠️  YES' if frag.swap_pressure else '✅ NO'}")
        
        if frag.fragmentation_percent > 50:
            print(f"  ⚠️  WARNING: High memory fragmentation may impact large model loading")
        if frag.swap_pressure:
            print(f"  ⚠️  WARNING: Swap pressure detected - consider increasing RAM")
    
    # Process scheduler analysis
    if system_metrics.scheduler_metrics:
        sched = system_metrics.scheduler_metrics
        print(f"\n⏰ Process Scheduler (Inference Latency Impact):")
        print(f"  Context Switches: {sched.context_switches_per_sec:.0f}/sec")
        print(f"  Load Averages: {sched.load_average_1m:.2f}, {sched.load_average_5m:.2f}, {sched.load_average_15m:.2f}")
        
        # Analyze scheduler efficiency
        cpu_cores = 8  # Assume 8 cores, would normally detect this
        if sched.load_average_1m > cpu_cores * 1.5:
            print(f"  ⚠️  WARNING: High load average may cause scheduling delays")
        if sched.context_switches_per_sec > 10000:
            print(f"  ⚠️  WARNING: High context switching may impact inference latency")
    
    # LLM process analysis
    if system_metrics.llm_process_metrics:
        proc = system_metrics.llm_process_metrics
        print(f"\n🧠 LLM Process Metrics:")
        print(f"  Process ID: {proc.pid}")
        print(f"  CPU Usage: {proc.cpu_percent:.1f}%")
        print(f"  RSS Memory: {proc.memory_rss_mb:.1f} MB")
        print(f"  VMS Memory: {proc.memory_vms_mb:.1f} MB")
        print(f"  Model Memory: {proc.model_memory_mb:.1f} MB")
        print(f"  Inference Threads: {proc.inference_threads}")
        print(f"  Open Files: {proc.open_files}")
        print(f"  Context Switches: {proc.context_switches}")
        print(f"  CPU Affinity: {proc.cpu_affinity}")
        print(f"  Nice Value: {proc.nice_value}")
        print(f"  I/O: Read={proc.io_read_bytes//1024//1024}MB, Write={proc.io_write_bytes//1024//1024}MB")
        
        # Process health analysis
        if proc.open_files > 1000:
            print(f"  ⚠️  WARNING: High file descriptor usage ({proc.open_files})")
        if proc.memory_rss_mb > 8000:  # More than 8GB
            print(f"  📈 INFO: Large model detected ({proc.model_memory_mb:.0f}MB)")
    
    # Container metrics (if applicable)
    if system_metrics.container_memory_limit_gb or system_metrics.container_cpu_limit:
        print(f"\n🐳 Container Metrics:")
        if system_metrics.container_memory_limit_gb:
            usage_pct = (system_metrics.memory_used_gb / system_metrics.container_memory_limit_gb) * 100
            print(f"  Memory Limit: {system_metrics.container_memory_limit_gb:.1f}GB")
            print(f"  Memory Usage: {usage_pct:.1f}% of container limit")
            if usage_pct > 85:
                print(f"  ⚠️  WARNING: Near container memory limit")
        
        if system_metrics.container_cpu_limit:
            print(f"  CPU Limit: {system_metrics.container_cpu_limit:.1f} cores")
        
        if system_metrics.container_throttled_time_ms > 0:
            print(f"  ⚠️  WARNING: CPU throttling detected ({system_metrics.container_throttled_time_ms:.1f}ms)")
    
    # GPU metrics
    if system_metrics.gpu_count > 0 and system_metrics.gpu_metrics:
        print(f"\n🎮 GPU Metrics:")
        for gpu in system_metrics.gpu_metrics:
            print(f"  GPU {gpu.get('gpu_id', 0)}: {gpu.get('name', 'Unknown')}")
            print(f"    Utilization: {gpu.get('utilization_percent', 0):.1f}%")
            print(f"    Memory: {gpu.get('memory_percent', 0):.1f}% ({gpu.get('memory_used_mb', 0):.0f}/{gpu.get('memory_total_mb', 0):.0f}MB)")
            print(f"    Temperature: {gpu.get('temperature', 0):.1f}°C")
            print(f"    Power: {gpu.get('power_draw_watts', 0):.1f}W / {gpu.get('power_limit_watts', 0):.1f}W")


def calculate_llm_performance_score(system_metrics):
    """Calculate a comprehensive LLM performance score based on system metrics."""
    score = 100
    issues = []
    
    # Memory analysis (critical for LLM)
    if system_metrics.memory_pressure:
        score -= 25
        issues.append("Memory pressure detected")
    elif system_metrics.memory_percent > 85:
        score -= 15
        issues.append("High memory usage")
    
    # Thermal analysis
    if system_metrics.thermal_throttling:
        score -= 30
        issues.append("Thermal throttling active")
    elif system_metrics.cpu_temp_celsius > 80:
        score -= 15
        issues.append("High CPU temperature")
    
    # System load analysis
    if system_metrics.system_load_1m > 8:  # Assuming 8-core system
        score -= 20
        issues.append("High system load")
    elif system_metrics.system_load_1m > 4:
        score -= 10
        issues.append("Moderate system load")
    
    # Disk I/O analysis
    if system_metrics.disk_io_metrics:
        total_read_rate = sum(d.read_bytes_per_sec for d in system_metrics.disk_io_metrics) / (1024*1024)
        avg_latency = sum(d.read_latency_ms for d in system_metrics.disk_io_metrics) / len(system_metrics.disk_io_metrics)
        
        if total_read_rate < 50:  # Less than 50 MB/s
            score -= 10
            issues.append("Low disk read rate")
        if avg_latency > 10:  # More than 10ms
            score -= 15
            issues.append("High disk latency")
    
    # Memory fragmentation
    if system_metrics.memory_fragmentation:
        if system_metrics.memory_fragmentation.swap_pressure:
            score -= 20
            issues.append("Swap pressure")
        if system_metrics.memory_fragmentation.fragmentation_percent > 60:
            score -= 10
            issues.append("High memory fragmentation")
    
    # Process scheduler
    if system_metrics.scheduler_metrics:
        if system_metrics.scheduler_metrics.context_switches_per_sec > 15000:
            score -= 10
            issues.append("High context switching")
    
    # Container constraints
    if system_metrics.container_throttled_time_ms > 100:  # More than 100ms throttling
        score -= 15
        issues.append("Container CPU throttling")
    
    return max(0, score), issues


def generate_performance_recommendations(system_metrics, score, issues):
    """Generate specific recommendations for LLM performance improvement."""
    recommendations = []
    
    if "Memory pressure detected" in issues:
        recommendations.append("🚨 CRITICAL: Reduce model size or increase system RAM")
        recommendations.append("Consider model quantization (INT8/FP16) to reduce memory usage")
        recommendations.append("Enable model sharding across multiple GPUs if available")
    
    if "Thermal throttling active" in issues:
        recommendations.append("🚨 CRITICAL: Improve system cooling to prevent performance throttling")
        recommendations.append("Reduce inference batch size to lower thermal load")
        recommendations.append("Consider underclocking CPU/GPU to reduce heat generation")
    
    if "High system load" in issues:
        recommendations.append("⚠️  Reduce concurrent inference requests")
        recommendations.append("Consider CPU affinity for LLM process")
        recommendations.append("Scale to multiple instances or machines")
    
    if "Low disk read rate" in issues:
        recommendations.append("⚠️  Use SSD storage for model files")
        recommendations.append("Consider model caching in memory")
        recommendations.append("Implement model file prefetching")
    
    if "High disk latency" in issues:
        recommendations.append("⚠️  Check for disk fragmentation")
        recommendations.append("Consider RAID configuration for better I/O")
        recommendations.append("Move model files to faster storage")
    
    if "Swap pressure" in issues:
        recommendations.append("🚨 Add more RAM or reduce memory usage")
        recommendations.append("Disable swap for LLM processes if possible")
        recommendations.append("Use memory-mapped model loading")
    
    if "High memory fragmentation" in issues:
        recommendations.append("⚠️  Restart LLM service to defragment memory")
        recommendations.append("Use huge pages for large memory allocations")
        recommendations.append("Consider memory pooling strategies")
    
    if "Container CPU throttling" in issues:
        recommendations.append("⚠️  Increase container CPU limits")
        recommendations.append("Optimize container resource allocation")
        recommendations.append("Consider dedicated nodes for LLM workloads")
    
    # General recommendations based on score
    if score < 60:
        recommendations.append("🔧 Consider hardware upgrade for better LLM performance")
        recommendations.append("🔧 Implement comprehensive monitoring and alerting")
    
    if not recommendations:
        recommendations.append("✅ System appears optimized for LLM inference")
        recommendations.append("✅ Continue monitoring for performance regressions")
    
    return recommendations


def test_enhanced_system_metrics():
    """Test the enhanced system-level metrics for LLM performance monitoring."""
    print("🚀 ENHANCED SYSTEM-LEVEL METRICS TEST FOR LLM PERFORMANCE")
    print("=" * 70)
    
    # Initialize metrics collector
    collector = MetricsCollector()
    collector.start()
    
    try:
        # Wait for initial metrics collection
        print("⏳ Collecting baseline system metrics...")
        time.sleep(3)
        
        # Get comprehensive system metrics
        system_metrics = collector.get_current_system_metrics()
        
        if not system_metrics:
            print("❌ No system metrics available")
            return
        
        # Display comprehensive system analysis
        display_system_health_analysis(system_metrics)
        
        # Calculate performance score
        score, issues = calculate_llm_performance_score(system_metrics)
        
        print(f"\n🏥 LLM PERFORMANCE HEALTH SCORE")
        print("=" * 40)
        print(f"Overall Score: {score}/100")
        
        if score >= 90:
            status = "🟢 EXCELLENT"
            description = "System optimized for LLM inference"
        elif score >= 75:
            status = "🟡 GOOD"
            description = "Minor performance issues detected"
        elif score >= 60:
            status = "🟠 FAIR"
            description = "Performance issues may impact inference"
        else:
            status = "🔴 POOR"
            description = "Significant performance issues detected"
        
        print(f"Status: {status}")
        print(f"Assessment: {description}")
        
        if issues:
            print(f"\n⚠️  Issues Detected:")
            for issue in issues:
                print(f"  • {issue}")
        
        # Generate recommendations
        recommendations = generate_performance_recommendations(system_metrics, score, issues)
        print(f"\n💡 PERFORMANCE RECOMMENDATIONS")
        print("=" * 40)
        for rec in recommendations:
            print(f"  {rec}")
        
        # Simulate some inference load and test performance impact
        print(f"\n🔄 SIMULATING INFERENCE LOAD...")
        print("=" * 40)
        
        # Simulate inference requests
        for i in range(3):
            request_id = str(uuid.uuid4())
            
            # Simulate inference metrics
            inference_metrics = InferenceMetrics(
                request_id=request_id,
                model_name="enhanced-llm-model",
                prompt_tokens=random.randint(100, 500),
                completion_tokens=random.randint(50, 300),
                total_tokens=random.randint(150, 800),
                response_time_ms=random.uniform(800, 2500),
                queue_time_ms=random.uniform(50, 200),
                processing_time_ms=random.uniform(750, 2300),
                tokens_per_second=random.uniform(15, 45),
                success=True,
                memory_peak_mb=random.uniform(2000, 6000),
                gpu_utilization_percent=random.uniform(70, 95),
                cache_hit=random.choice([True, False]),
                batch_size=1,
                sequence_length=random.randint(150, 800)
            )
            
            collector.log_inference(inference_metrics)
            print(f"  Request {i+1}: {inference_metrics.response_time_ms:.0f}ms, "
                  f"{inference_metrics.tokens_per_second:.1f} tokens/sec, "
                  f"Peak Mem: {inference_metrics.memory_peak_mb:.0f}MB")
        
        # Get updated performance summary
        summary = collector.get_performance_summary("1h")
        print(f"\n📊 PERFORMANCE SUMMARY")
        print("=" * 30)
        print(f"  Total Requests: {summary.total_requests}")
        print(f"  Success Rate: {(summary.successful_requests/summary.total_requests)*100:.1f}%")
        print(f"  Avg Response Time: {summary.avg_response_time_ms:.0f}ms")
        print(f"  P95 Response Time: {summary.p95_response_time_ms:.0f}ms")
        print(f"  Avg Tokens/sec: {summary.avg_tokens_per_second:.1f}")
        print(f"  Peak Memory: {summary.peak_memory_usage_mb:.0f}MB")
        print(f"  Avg GPU Utilization: {summary.avg_gpu_utilization:.1f}%")
        print(f"  Cache Hit Rate: {summary.cache_hit_rate:.1f}%")
        
        print(f"\n✅ Enhanced System Metrics Test Complete!")
        print(f"📈 {len(system_metrics.disk_io_metrics)} disk devices monitored")
        print(f"📈 {len(system_metrics.network_metrics)} network interfaces monitored")
        print(f"📈 {len(system_metrics.thermal_zones)} thermal zones monitored")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        collector.stop()


if __name__ == "__main__":
    test_enhanced_system_metrics() 