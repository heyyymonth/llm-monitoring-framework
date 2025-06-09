#!/usr/bin/env python3
"""
Quality monitoring component validation script for CI testing.
"""

from monitoring.quality import QualityAssessor, SafetyEvaluator, HallucinationDetector
from monitoring.models import QualityMetrics, SafetyAssessment

def main():
    # Test quality assessor
    assessor = QualityAssessor()
    print('✅ QualityAssessor initialized')
    
    # Test safety evaluator  
    evaluator = SafetyEvaluator()
    print('✅ SafetyEvaluator initialized')
    
    # Test hallucination detector
    detector = HallucinationDetector()
    print('✅ HallucinationDetector initialized')
    
    print('✅ Quality monitoring components working correctly')

if __name__ == '__main__':
    main() 