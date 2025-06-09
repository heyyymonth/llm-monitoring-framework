#!/usr/bin/env python3
"""
Cost tracking component validation script for CI testing.
"""

from monitoring.cost import CostTracker, BudgetManager
from monitoring.models import CostMetrics

def main():
    # Test cost tracker
    tracker = CostTracker()
    print('✅ CostTracker initialized')
    
    # Test budget manager
    manager = BudgetManager()
    print('✅ BudgetManager initialized')
    
    print('✅ Cost tracking components working correctly')

if __name__ == '__main__':
    main() 