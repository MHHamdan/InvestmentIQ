"""
InvestmentIQ MVAS - Launcher Script

This script launches the Streamlit dashboard for InvestmentIQ.
With LangGraph v3.0, all functionality is accessed through the web dashboard.
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Launch the Streamlit dashboard."""
    # Add the project root to Python path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))

    dashboard_path = project_root / "apps" / "dashboard.py"

    print("🚀 Launching InvestmentIQ MVAS Dashboard...")
    print(f"📊 Dashboard: {dashboard_path}")
    print("\nPress Ctrl+C to stop the dashboard\n")

    # Launch streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", str(dashboard_path)
    ])

if __name__ == "__main__":
    main()
