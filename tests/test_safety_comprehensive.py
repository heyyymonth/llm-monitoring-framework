"""
Comprehensive Safety Monitoring Tests

This module provides extensive testing for LLM safety assessment capabilities,
focusing on the core safety dimensions: Hallucination Detection, Toxicity 
Filtering, Bias Detection, and PII Protection.
"""

import pytest
from typing import List, Dict

from monitoring.quality import SafetyEvaluator, HallucinationDetector
from monitoring.models import SafetyFlag, SafetyAssessment


class TestHallucinationDetection:
    """Comprehensive tests for hallucination detection."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = HallucinationDetector()
    
    def test_explicit_hallucination_patterns(self):
        """Test detection of explicit hallucination admission patterns."""
        hallucination_responses = [
            "I apologize, but I don't actually have access to real-time data.",
            "I made that up and cannot verify this information.",
            "I cannot verify this claim without access to current databases.",
            "As an AI language model, I don't have access to live information.",
            "I should clarify that I don't actually know this specific detail."
        ]
        
        for response in hallucination_responses:
            is_hallucination, score, details = self.detector.detect_hallucination(response)
            assert is_hallucination == True, f"Failed to detect hallucination in: {response}"
            assert score > 0.5, f"Low hallucination score {score} for: {response}"
            assert len(details["pattern_matches"]) > 0, f"No pattern matches for: {response}"
    
    def test_confidence_marker_detection(self):
        """Test detection of unsupported confidence claims."""
        overconfident_responses = [
            "I am absolutely certain this is 100% correct without any doubt.",
            "This is definitely true and guaranteed to work perfectly.",
            "Without a doubt, this is a proven fact that's absolutely reliable.",
            "I'm completely sure this is certainly the right answer.",
            "This is guaranteed to be the definitive solution."
        ]
        
        for response in overconfident_responses:
            is_hallucination, score, details = self.detector.detect_hallucination(response)
            assert score > 0.0, f"No confidence markers detected in: {response}"
            assert len(details["confidence_markers"]) > 0, f"No confidence markers found: {response}"
    
    def test_clean_responses_no_hallucination(self):
        """Test that legitimate responses don't trigger hallucination detection."""
        clean_responses = [
            "Paris is the capital of France, located in the northern part of the country.",
            "Machine learning involves training algorithms on data to make predictions.",
            "Photosynthesis is the process by which plants convert sunlight into energy.",
            "The water cycle includes evaporation, condensation, and precipitation.",
            "Python is a programming language known for its simplicity and readability."
        ]
        
        for response in clean_responses:
            is_hallucination, score, details = self.detector.detect_hallucination(response)
            assert is_hallucination == False, f"False positive hallucination for: {response}"
            assert score < 0.5, f"High hallucination score {score} for clean response: {response}"


class TestToxicityDetection:
    """Comprehensive tests for toxicity detection."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.evaluator = SafetyEvaluator()
    
    def test_explicit_toxic_content(self):
        """Test detection of explicitly toxic content."""
        toxic_responses = [
            "This content contains harmful and dangerous language that promotes violence.",
            "The discussion involves toxic behavior and offensive material.",
            "This response includes discriminatory and inappropriate content.",
            "Such harmful rhetoric is dangerous to communities."
        ]
        
        for response in toxic_responses:
            toxicity_score = self.evaluator._check_toxicity(response)
            assert toxicity_score > 0.0, f"No toxicity detected in: {response}"
    
    def test_context_aware_toxicity(self):
        """Test that toxicity detection considers context appropriately."""
        contextual_responses = [
            {
                "text": "The violent storm caused significant damage to the coastline.",
                "expected_safe": True,  # "violent" in weather context
                "description": "Weather violence"
            },
            {
                "text": "This harmful chemical reaction requires proper safety protocols.",
                "expected_safe": True,  # "harmful" in scientific context
                "description": "Scientific harm"
            },
            {
                "text": "The dangerous mountain climbing expedition required expert guides.",
                "expected_safe": True,  # "dangerous" in adventure context
                "description": "Adventure danger"
            },
            {
                "text": "We need to address the toxic workplace culture through policy changes.",
                "expected_safe": True,  # "toxic" in professional context
                "description": "Workplace toxicity"
            }
        ]
        
        for case in contextual_responses:
            toxicity_score = self.evaluator._check_toxicity(case["text"])
            if case["expected_safe"]:
                # Should have low or zero toxicity score for contextual usage
                assert toxicity_score <= 0.4, \
                    f"High toxicity score {toxicity_score} for contextual use: {case['description']}"


class TestPIIDetection:
    """Comprehensive tests for PII (Personally Identifiable Information) detection."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.evaluator = SafetyEvaluator()
    
    def test_ssn_detection(self):
        """Test detection of Social Security Numbers."""
        ssn_responses = [
            "My SSN is 123-45-6789 for the application.",
            "Social Security Number: 987-65-4321",
            "The number 555-44-3333 is my identifier.",
            "Please use SSN 111-22-3333 for verification."
        ]
        
        for response in ssn_responses:
            pii_detected = self.evaluator._check_pii(response)
            assert pii_detected == True, f"Failed to detect SSN in: {response}"
    
    def test_email_detection(self):
        """Test detection of email addresses."""
        email_responses = [
            "Contact me at john.doe@example.com for more information.",
            "My email address is user123@company.org",
            "Send the report to analyst@research.edu please.",
            "You can reach me at firstname.lastname@domain.co.uk"
        ]
        
        for response in email_responses:
            pii_detected = self.evaluator._check_pii(response)
            assert pii_detected == True, f"Failed to detect email in: {response}"
    
    def test_no_pii_false_positives(self):
        """Test that non-PII content doesn't trigger false positives."""
        clean_responses = [
            "The weather forecast shows 75 degrees at 12:30 PM.",
            "Please call extension 1234 for customer service.",
            "The meeting is scheduled for room 567 at 2:00 PM.",
            "Project deadline is March 15th with 100 participants.",
            "Version 2.5.3 includes these new features."
        ]
        
        for response in clean_responses:
            pii_detected = self.evaluator._check_pii(response)
            assert pii_detected == False, f"False positive PII detection for: {response}"


class TestSafetyIntegration:
    """Integration tests for comprehensive safety evaluation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.evaluator = SafetyEvaluator()
    
    def test_comprehensive_safety_evaluation(self):
        """Test complete safety evaluation with multiple checks."""
        test_cases = [
            {
                "name": "Clean safe response",
                "response": "Machine learning is a subset of AI that helps computers learn from data.",
                "expected_safe": True,
                "expected_flags": []
            },
            {
                "name": "Response with PII",
                "response": "Contact me at john@example.com for the machine learning tutorial.",
                "expected_safe": False,
                "expected_flags": [SafetyFlag.PII_LEAK]
            },
            {
                "name": "Toxic content",
                "response": "This harmful and dangerous approach is offensive to everyone.",
                "expected_safe": False,
                "expected_flags": [SafetyFlag.TOXICITY]
            },
            {
                "name": "Hallucination admission",
                "response": "I apologize, but I don't actually have access to verify this information.",
                "expected_safe": False,
                "expected_flags": [SafetyFlag.HALLUCINATION]
            }
        ]
        
        for case in test_cases:
            assessment = self.evaluator.evaluate_safety(case["response"])
            
            assert assessment.is_safe == case["expected_safe"], \
                f"Safety assessment mismatch for {case['name']}: expected {case['expected_safe']}, got {assessment.is_safe}"
            
            for expected_flag in case["expected_flags"]:
                assert expected_flag in assessment.flags, \
                    f"Missing expected flag {expected_flag} for {case['name']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
