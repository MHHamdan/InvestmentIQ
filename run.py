"""
InvestmentIQ MVAS - Launcher Script

This script serves as the entry point for running the system.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and run main
from main import main

if __name__ == "__main__":
    main()
