"""
LangSmith Tracing Utility for InvestmentIQ ADK Agents

Provides decorators and context managers for tracing agent execution,
API calls, and data processing steps.
"""

import os
import functools
from typing import Any, Dict, Optional
from datetime import datetime
from langsmith import traceable, Client
from langsmith.run_helpers import get_current_run_tree

# Initialize LangSmith client
LANGSMITH_ENABLED = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "investmentiq-adk")

if LANGSMITH_ENABLED:
    client = Client()
    print(f"✅ LangSmith tracing enabled for project: {LANGSMITH_PROJECT}")
else:
    client = None
    print("ℹ️  LangSmith tracing disabled")


def trace_agent(agent_name: str):
    """
    Decorator to trace entire agent execution.
    
    Usage:
        @trace_agent("financial_analyst")
        async def analyze(self, ticker, company_name, sector):
            ...
    """
    def decorator(func):
        if not LANGSMITH_ENABLED:
            return func
            
        @functools.wraps(func)
        @traceable(
            run_type="chain",
            name=f"{agent_name}_agent",
            project_name=LANGSMITH_PROJECT
        )
        async def wrapper(*args, **kwargs):
            # Extract ticker from args/kwargs for better tracing
            ticker = kwargs.get('ticker') or (args[1] if len(args) > 1 else 'UNKNOWN')
            
            # Add metadata
            run = get_current_run_tree()
            if run:
                run.extra = {
                    "agent": agent_name,
                    "ticker": ticker,
                    "timestamp": datetime.now().isoformat()
                }
            
            result = await func(*args, **kwargs)
            return result
            
        return wrapper
    return decorator


def trace_step(step_name: str, step_type: str = "tool"):
    """
    Decorator to trace individual steps within an agent.
    
    Usage:
        @trace_step("fetch_fmp_data", step_type="tool")
        async def _fetch_financial_ratios(self, ticker):
            ...
    """
    def decorator(func):
        if not LANGSMITH_ENABLED:
            return func
            
        @functools.wraps(func)
        @traceable(
            run_type=step_type,
            name=step_name,
            project_name=LANGSMITH_PROJECT
        )
        async def async_wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            return result
            
        @functools.wraps(func)
        @traceable(
            run_type=step_type,
            name=step_name,
            project_name=LANGSMITH_PROJECT
        )
        def sync_wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return result
            
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
            
    return decorator


def trace_llm_call(model_name: str = "gemini-2.0-flash"):
    """
    Decorator to trace LLM/Gemini API calls.
    
    Usage:
        @trace_llm_call("gemini-2.0-flash")
        async def _analyze_with_gemini(self, ticker, metrics):
            ...
    """
    def decorator(func):
        if not LANGSMITH_ENABLED:
            return func
            
        @functools.wraps(func)
        @traceable(
            run_type="llm",
            name=f"gemini_analysis",
            project_name=LANGSMITH_PROJECT
        )
        async def wrapper(*args, **kwargs):
            run = get_current_run_tree()
            if run:
                run.extra = {
                    "model": model_name,
                    "provider": "google",
                    "timestamp": datetime.now().isoformat()
                }
            
            result = await func(*args, **kwargs)
            
            # Log token usage if available
            if run and hasattr(result, 'usage_metadata'):
                run.outputs = {
                    "usage": {
                        "prompt_tokens": getattr(result.usage_metadata, 'prompt_token_count', 0),
                        "completion_tokens": getattr(result.usage_metadata, 'candidates_token_count', 0)
                    }
                }
            
            return result
            
        return wrapper
    return decorator


def log_metrics(agent_name: str, metrics: Dict[str, Any]):
    """
    Log extracted metrics to LangSmith.
    
    Usage:
        log_metrics("financial_analyst", {"revenue_growth": 0.15, "pe_ratio": 28.5})
    """
    if not LANGSMITH_ENABLED:
        return
        
    run = get_current_run_tree()
    if run:
        run.outputs = run.outputs or {}
        run.outputs["metrics"] = metrics
        run.extra = run.extra or {}
        run.extra["metrics_count"] = len(metrics)


def log_api_call(api_name: str, endpoint: str, status_code: int, response_time: float):
    """
    Log API call details to LangSmith.
    
    Usage:
        log_api_call("FMP", "/ratios/AAPL", 200, 0.245)
    """
    if not LANGSMITH_ENABLED:
        return
        
    run = get_current_run_tree()
    if run:
        run.outputs = run.outputs or {}
        run.outputs["api_call"] = {
            "api": api_name,
            "endpoint": endpoint,
            "status_code": status_code,
            "response_time_seconds": response_time
        }


def log_error(error: Exception, context: Optional[Dict[str, Any]] = None):
    """
    Log error details to LangSmith.
    
    Usage:
        try:
            result = await api_call()
        except Exception as e:
            log_error(e, {"ticker": ticker, "api": "FMP"})
            raise
    """
    if not LANGSMITH_ENABLED:
        return
        
    run = get_current_run_tree()
    if run:
        run.error = str(error)
        run.extra = run.extra or {}
        run.extra["error_context"] = context or {}
        run.end(error=error)
