"""
LLM Cost Tracking Module

Monitors and analyzes LLM costs to prevent budget overruns and optimize usage:
- Token usage tracking
- Cost per request analysis
- Budget alerts and optimization suggestions
- Model cost comparison
"""

from typing import Dict, List, Optional
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque
import logging

from .models import CostMetrics, CostAnalysis

logger = logging.getLogger(__name__)


class CostTracker:
    """Tracks and analyzes LLM usage costs."""
    
    def __init__(self, max_history: int = 10000):
        self.cost_history = deque(maxlen=max_history)
        self.model_costs = {
            "gpt-4": {"input": 0.00003, "output": 0.00006},
            "gpt-3.5-turbo": {"input": 0.000001, "output": 0.000002},
            "claude-3": {"input": 0.000015, "output": 0.000075},
            # Ollama models (local inference - electricity costs)
            "stable-code": {"input": 0.00001, "output": 0.00002},  
            "stable-code:latest": {"input": 0.00001, "output": 0.00002},
            "mistral-small3.1": {"input": 0.00001, "output": 0.00002},
            "llama3.1": {"input": 0.00001, "output": 0.00002},
            "llama3": {"input": 0.00001, "output": 0.00002},
            # Generic local model pricing (electricity + compute)
            "ollama": {"input": 0.00001, "output": 0.00002},
            "local": {"input": 0.00001, "output": 0.00002},
            "unknown": {"input": 0.00001, "output": 0.00002}
        }
        
    def log_inference(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_per_token: Optional[float] = None
    ) -> CostMetrics:
        """Log an inference with cost calculation."""
        
        if cost_per_token:
            cost_usd = (prompt_tokens + completion_tokens) * cost_per_token
        else:
            cost_usd = self._calculate_cost(model, prompt_tokens, completion_tokens)
        
        cost_metrics = CostMetrics(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=cost_usd,
            model_name=model,
            timestamp=datetime.now(timezone.utc)
        )
        
        self.cost_history.append(cost_metrics.model_dump())
        return cost_metrics
    
    def get_cost_analysis(self, timeframe: str = "24h") -> CostAnalysis:
        """Generate cost analysis for specified timeframe."""
        
        hours = 24 if timeframe == "24h" else 1
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        recent_costs = [
            record for record in self.cost_history
            if record["timestamp"] >= cutoff
        ]
        
        if not recent_costs:
            return CostAnalysis(
                time_period=timeframe,
                total_cost_usd=0.0,
                avg_cost_per_request=0.0,
                most_expensive_operations=[],
                optimization_suggestions=[],
                projected_monthly_cost=0.0
            )
        
        total_cost = sum(record["cost_usd"] for record in recent_costs)
        avg_cost_per_request = total_cost / len(recent_costs)
        
        # Generate optimization suggestions
        optimization_suggestions = self._generate_optimization_suggestions(recent_costs)
        
        # Get most expensive operations
        sorted_costs = sorted(recent_costs, key=lambda x: x["cost_usd"], reverse=True)
        most_expensive_operations = [
            f"{record['model_name']} (${record['cost_usd']:.4f})"
            for record in sorted_costs[:5]
        ]
        
        return CostAnalysis(
            time_period=timeframe,
            total_cost_usd=total_cost,
            avg_cost_per_request=avg_cost_per_request,
            most_expensive_operations=most_expensive_operations,
            optimization_suggestions=optimization_suggestions,
            projected_monthly_cost=total_cost * 30
        )
    
    def get_cost_by_model(self, timeframe: str = "24h") -> Dict[str, float]:
        """Get cost breakdown by model."""
        
        if timeframe == "1h":
            hours = 1
        elif timeframe == "24h":
            hours = 24
        elif timeframe == "7d":
            hours = 168
        else:
            hours = 24
            
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        model_costs = defaultdict(float)
        for record in self.cost_history:
            if record["timestamp"] >= cutoff:
                model_costs[record["model_name"]] += record["cost_usd"]
        
        return dict(model_costs)
    
    def get_cost_by_user(self, timeframe: str = "24h") -> Dict[str, float]:
        """Get cost breakdown by user."""
        
        if timeframe == "1h":
            hours = 1
        elif timeframe == "24h":
            hours = 24
        elif timeframe == "7d":
            hours = 168
        else:
            hours = 24
            
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        user_costs = defaultdict(float)
        for record in self.cost_history:
            if record["timestamp"] >= cutoff and record["user_id"]:
                user_costs[record["user_id"]] += record["cost_usd"]
        
        return dict(user_costs)
    
    def check_budget_alert(self, daily_budget: float) -> Dict:
        """Check if daily budget is being exceeded."""
        
        today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        today_costs = [
            record for record in self.cost_history
            if record["timestamp"] >= today
        ]
        
        today_total = sum(record["cost_usd"] for record in today_costs)
        budget_used_pct = (today_total / daily_budget) * 100 if daily_budget > 0 else 0
        
        alert_level = "none"
        if budget_used_pct >= 100:
            alert_level = "critical"
        elif budget_used_pct >= 80:
            alert_level = "warning"
        elif budget_used_pct >= 60:
            alert_level = "info"
        
        return {
            "alert_level": alert_level,
            "budget_used_pct": budget_used_pct,
            "cost_today": today_total,
            "daily_budget": daily_budget,
            "remaining_budget": max(0, daily_budget - today_total)
        }
    
    def _calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost based on model pricing."""
        
        if model not in self.model_costs:
            input_cost = 0.00001
            output_cost = 0.00002
        else:
            input_cost = self.model_costs[model]["input"]
            output_cost = self.model_costs[model]["output"]
        
        return (prompt_tokens * input_cost) + (completion_tokens * output_cost)
    
    def _generate_optimization_suggestions(self, cost_records: List[Dict]) -> List[str]:
        """Generate cost optimization suggestions based on usage patterns."""
        
        suggestions = []
        
        if not cost_records:
            return suggestions
        
        # Analyze model usage
        model_usage = defaultdict(list)
        for record in cost_records:
            model_usage[record["model_name"]].append(record)
        
        # Check for expensive model overuse
        total_cost = sum(record["cost_usd"] for record in cost_records)
        for model, records in model_usage.items():
            model_cost = sum(record["cost_usd"] for record in records)
            model_pct = (model_cost / total_cost) * 100
            
            if model_pct > 70 and model in ["gpt-4", "claude-3"]:
                suggestions.append(
                    f"Consider using cheaper models for simple tasks. {model} accounts for "
                    f"{model_pct:.1f}% of costs."
                )
        
        # Check for long prompts
        avg_prompt_tokens = sum(record["prompt_tokens"] for record in cost_records) / len(cost_records)
        if avg_prompt_tokens > 1000:
            suggestions.append(
                f"Average prompt length is {avg_prompt_tokens:.0f} tokens. "
                "Consider prompt optimization to reduce input costs."
            )
        
        # Check for long responses
        avg_completion_tokens = sum(record["completion_tokens"] for record in cost_records) / len(cost_records)
        if avg_completion_tokens > 500:
            suggestions.append(
                f"Average response length is {avg_completion_tokens:.0f} tokens. "
                "Consider setting max_tokens limits to control output costs."
            )
        
        # Check token efficiency
        for record in cost_records:
            if record["prompt_tokens"] > record["completion_tokens"] * 3:
                suggestions.append(
                    "Some requests have very long prompts relative to responses. "
                    "Consider prompt compression techniques."
                )
                break
        
        return suggestions[:5]  # Limit to top 5 suggestions


class BudgetManager:
    """Manages budgets and alerts for LLM usage."""
    
    def __init__(self):
        self.budgets = {}  # user_id -> daily_budget
        self.alerts = []
        
    def set_user_budget(self, user_id: str, daily_budget: float):
        """Set daily budget for a user."""
        self.budgets[user_id] = daily_budget
        
    def check_all_budgets(self, cost_tracker: CostTracker) -> List[Dict]:
        """Check all user budgets and return alerts."""
        alerts = []
        
        user_costs = cost_tracker.get_cost_by_user("24h")
        
        for user_id, daily_budget in self.budgets.items():
            current_cost = user_costs.get(user_id, 0.0)
            usage_pct = (current_cost / daily_budget) * 100 if daily_budget > 0 else 0
            
            if usage_pct >= 100:
                alerts.append({
                    "user_id": user_id,
                    "alert_type": "budget_exceeded",
                    "usage_pct": usage_pct,
                    "current_cost": current_cost,
                    "daily_budget": daily_budget
                })
            elif usage_pct >= 80:
                alerts.append({
                    "user_id": user_id,
                    "alert_type": "budget_warning",
                    "usage_pct": usage_pct,
                    "current_cost": current_cost,
                    "daily_budget": daily_budget
                })
        
        return alerts 