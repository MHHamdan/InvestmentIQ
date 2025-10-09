"""
Batch script to add LangSmith tracing to all agents.
Adds import statements and decorators programmatically.

Usage:
    python utils/add_tracing.py
"""

import re

# List of agent files to update
agents = [
    "agents/adk_market_intelligence.py",
    "agents/adk_qualitative_signal.py",
    "agents/adk_context_engine.py",
    "agents/adk_orchestrator.py"
]

# Import statement to add
import_line = "from utils.langsmith_tracer import trace_agent, trace_step, trace_llm_call, log_metrics, log_api_call, log_error\n"

for agent_file in agents:
    with open(agent_file, 'r') as f:
        content = f.read()
    
    # Skip if already has tracing
    if "langsmith_tracer" in content:
        print(f"✓ {agent_file} already has tracing")
        continue
    
    # Add import after datetime
    content = content.replace(
        "from datetime import datetime\n",
        f"from datetime import datetime\n{import_line}"
    )
    
    # Add @trace_agent decorator to main analyze() method
    agent_name = agent_file.split('_', 1)[1].replace('.py', '').replace('adk_', '')
    content = re.sub(
        r'(\s+)async def analyze\(',
        rf'\1@trace_agent("{agent_name}")\n\1async def analyze(',
        content
    )
    
    # Add @trace_llm_call to _analyze_with_gemini
    content = re.sub(
        r'(\s+)async def _analyze_with_gemini\(',
        r'\1@trace_llm_call("gemini-2.0-flash")\n\1async def _analyze_with_gemini(',
        content
    )
    
    # Write back
    with open(agent_file, 'w') as f:
        f.write(content)
    
    print(f"✅ Added tracing to {agent_file}")

print("\n✨ Tracing added to all agents!")
