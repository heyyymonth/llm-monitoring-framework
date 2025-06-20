#!/usr/bin/env python3
"""
Factual Accuracy Monitoring Demonstration

This script demonstrates the new comprehensive factual accuracy monitoring feature
that evaluates content verification and consistency in LLM responses.

Features demonstrated:
- Internal consistency and contradiction detection
- Citation and reference verification
- Quantitative data accuracy (numbers, dates, statistics)
- Knowledge consistency assessment
- Temporal accuracy evaluation
"""

import sys
sys.path.append('..')

from monitoring.quality import QualityMonitor

def demonstrate_factual_accuracy():
    """Demonstrate various aspects of factual accuracy monitoring."""
    
    monitor = QualityMonitor()
    
    print("🔍 Factual Accuracy Monitoring Demonstration")
    print("=" * 60)
    
    # Test Case 1: Well-cited response with quantitative data
    print("\n📊 Test Case 1: Response with Citations and Data")
    print("-" * 50)
    
    prompt1 = "What is the current global smartphone adoption rate?"
    response1 = """According to research from Statista (2023), approximately 6.8 billion people worldwide use smartphones, representing about 85% of the global population. The adoption rate has grown significantly from 49% in 2016 to the current 85% in 2023."""
    
    result1 = monitor.evaluate_factual_accuracy(prompt1, response1)
    print(f"Prompt: {prompt1}")
    print(f"Response: {response1}")
    print(f"\n📈 Results:")
    print(f"  Overall Accuracy: {result1['overall_accuracy']:.3f}")
    print(f"  Consistency Score: {result1['consistency_score']:.3f}")
    print(f"  Citation Score: {result1['citation_score']:.3f}")
    print(f"  Quantitative Accuracy: {result1['quantitative_accuracy']:.3f}")
    print(f"  Temporal Accuracy: {result1['temporal_accuracy']:.3f}")
    print(f"  Claims Detected: {len(result1['detected_claims'])}")
    
    # Test Case 2: Response with contradictions
    print("\n⚠️  Test Case 2: Response with Internal Contradictions")
    print("-" * 50)
    
    prompt2 = "Is artificial intelligence safe?"
    response2 = """Artificial intelligence is completely safe and poses no risks to humanity. However, AI systems can be dangerous and unpredictable. Studies show that AI is 100% reliable, but many experts warn about potential AI safety concerns."""
    
    result2 = monitor.evaluate_factual_accuracy(prompt2, response2)
    print(f"Prompt: {prompt2}")
    print(f"Response: {response2}")
    print(f"\n📈 Results:")
    print(f"  Overall Accuracy: {result2['overall_accuracy']:.3f}")
    print(f"  Consistency Score: {result2['consistency_score']:.3f}")
    print(f"  Contradictions Found: {len(result2['verification_results']['consistency']['contradictions'])}")
    print(f"  Recommendations: {', '.join(result2['recommendations'])}")
    
    # Test Case 3: Response with questionable quantitative claims
    print("\n🔢 Test Case 3: Response with Questionable Quantitative Claims")
    print("-" * 50)
    
    prompt3 = "What is the speed of light?"
    response3 = """The speed of light in vacuum is approximately 300,000 km/s. This was first measured in 1887, and recent studies from 2025 have confirmed this value with 150% accuracy."""
    
    result3 = monitor.evaluate_factual_accuracy(prompt3, response3)
    print(f"Prompt: {prompt3}")
    print(f"Response: {response3}")
    print(f"\n📈 Results:")
    print(f"  Overall Accuracy: {result3['overall_accuracy']:.3f}")
    print(f"  Quantitative Accuracy: {result3['quantitative_accuracy']:.3f}")
    print(f"  Temporal Accuracy: {result3['temporal_accuracy']:.3f}")
    print(f"  Accuracy Issues: {result3['verification_results']['quantitative']['accuracy_issues']}")
    print(f"  Temporal Issues: {result3['verification_results']['temporal']['temporal_issues']}")
    
    # Test Case 4: Integration with full quality assessment
    print("\n🔄 Test Case 4: Integration with Full Quality Assessment")
    print("-" * 50)
    
    prompt4 = "Explain quantum computing"
    response4 = """Quantum computing is a revolutionary technology that uses quantum mechanical phenomena to process information. According to IBM research, quantum computers can potentially solve certain problems exponentially faster than classical computers. Current quantum computers have achieved quantum supremacy in specific tasks."""
    
    full_assessment = monitor.evaluate_response(prompt4, response4, "gpt-4")
    factual_details = monitor.evaluate_factual_accuracy(prompt4, response4)
    
    print(f"Prompt: {prompt4}")
    print(f"Response: {response4}")
    print(f"\n📈 Quality Assessment Results:")
    print(f"  Overall Quality: {full_assessment.quality_metrics.overall_quality:.3f}")
    print(f"  Factual Accuracy: {full_assessment.quality_metrics.factual_accuracy:.3f}")
    print(f"  Semantic Similarity: {full_assessment.quality_metrics.semantic_similarity:.3f}")
    print(f"  Response Relevance: {full_assessment.quality_metrics.response_relevance:.3f}")
    print(f"  Coherence Score: {full_assessment.quality_metrics.coherence_score:.3f}")
    
    print(f"\n📊 Detailed Factual Accuracy Analysis:")
    print(f"  Consistency: {factual_details['consistency_score']:.3f}")
    print(f"  Citations: {factual_details['citation_score']:.3f}")
    print(f"  Knowledge: {factual_details['knowledge_consistency']:.3f}")
    print(f"  Confidence Calibration: {factual_details['confidence_indicators']['confidence_calibration']}")
    
    print("\n✅ Demonstration completed!")
    print("\nKey Features Demonstrated:")
    print("• Comprehensive fact-checking across multiple dimensions")
    print("• Internal consistency and contradiction detection")  
    print("• Citation quality and reference verification")
    print("• Quantitative data accuracy validation")
    print("• Temporal accuracy and currency assessment")
    print("• Integration with existing quality monitoring")
    print("• Detailed analysis and actionable recommendations")

if __name__ == "__main__":
    demonstrate_factual_accuracy() 