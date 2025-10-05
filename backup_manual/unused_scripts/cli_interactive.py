"""
Interactive CLI Interface for InvestmentIQ MVAS

Provides a menu-driven interface for exploring system capabilities,
running analyses, and viewing results.
"""

import asyncio
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich import box
from pathlib import Path
import json

from main_enhanced import InvestmentIQSystemEnhanced
from utils.ui_components import InvestmentIQUI


class InteractiveCLI:
    """Interactive command-line interface for InvestmentIQ"""

    def __init__(self):
        self.console = Console()
        self.ui = InvestmentIQUI()
        self.system = None
        self.last_results = None

    def show_banner(self):
        """Display application banner"""
        banner = """
[bold cyan]╔═══════════════════════════════════════════════════════════════╗[/bold cyan]
[bold cyan]║                                                               ║[/bold cyan]
[bold cyan]║[/bold cyan]       [bold white]InvestmentIQ MVAS - Interactive Console[/bold white]           [bold cyan]║[/bold cyan]
[bold cyan]║                                                               ║[/bold cyan]
[bold cyan]║[/bold cyan]     [dim]Multi-Agent Financial Intelligence System[/dim]            [bold cyan]║[/bold cyan]
[bold cyan]║[/bold cyan]              [dim]Group 2 Capstone Project[/dim]                   [bold cyan]║[/bold cyan]
[bold cyan]║                                                               ║[/bold cyan]
[bold cyan]╚═══════════════════════════════════════════════════════════════╝[/bold cyan]
        """
        self.console.print(banner)

    def show_main_menu(self):
        """Display main menu"""
        menu = Table(
            show_header=False,
            box=box.ROUNDED,
            border_style="cyan",
            title="[bold]Main Menu[/bold]",
            title_style="bold cyan"
        )

        menu.add_column("Option", style="cyan bold", width=10)
        menu.add_column("Description", style="white")

        options = [
            ("1", "Run Contrarian Opportunity Analysis"),
            ("2", "View System Architecture"),
            ("3", "Test Individual Agents"),
            ("4", "View Sample Data"),
            ("5", "View Last Analysis Results"),
            ("6", "System Performance Metrics"),
            ("7", "About InvestmentIQ"),
            ("0", "Exit")
        ]

        for opt, desc in options:
            menu.add_row(opt, desc)

        self.console.print(menu)

    async def run_analysis(self):
        """Run investment analysis"""
        if not self.system:
            self.ui.print_info("Initializing system...")
            self.system = InvestmentIQSystemEnhanced()

        self.ui.print_section("Investment Analysis", "bold cyan")

        # Show available companies
        companies = ["COMPANY_X (TechVision Inc.)", "Custom Company (Coming Soon)"]

        company_table = Table(
            show_header=True,
            header_style="bold magenta",
            box=box.SIMPLE
        )
        company_table.add_column("#", style="cyan")
        company_table.add_column("Company", style="white")

        for i, company in enumerate(companies, 1):
            company_table.add_row(str(i), company)

        self.console.print(company_table)

        choice = Prompt.ask(
            "[cyan]Select company[/cyan]",
            choices=["1", "2"],
            default="1"
        )

        if choice == "1":
            company_id = "COMPANY_X"
        else:
            self.ui.print_warning("Custom company analysis coming soon!")
            return

        # Run analysis
        self.ui.print_info(f"Analyzing {company_id}...")
        self.last_results = await self.system.analyze_investment(company_id)

        # Display results
        data = self.last_results.get("data", {})
        self.ui.print_analysis_results(data)

        # Get detailed logs
        logs = self.system.get_detailed_logs()

        # Show workflow
        if Confirm.ask("[cyan]View workflow execution log?[/cyan]", default=True):
            self.ui.print_workflow_tree(logs.get("workflow_log", []))

        # Show A2A communication
        if Confirm.ask("[cyan]View A2A communication log?[/cyan]", default=True):
            self.ui.print_a2a_communication(logs.get("a2a_messages", []))

        # Save results
        if Confirm.ask("[cyan]Save results to file?[/cyan]", default=True):
            output_file = self.system.save_results({
                "analysis_results": self.last_results,
                "detailed_logs": logs
            })
            self.ui.print_success(f"Results saved to: {output_file}")

    def view_architecture(self):
        """Display system architecture"""
        self.ui.print_section("System Architecture", "bold blue")

        arch_text = """
[bold]4-Agent Architecture:[/bold]

[cyan]1. Strategic Orchestrator Agent[/cyan]
   • Workflow management and coordination
   • Conflict detection between signals
   • Final decision synthesis
   • A2A communication coordination

[cyan]2. Financial Analyst Agent[/cyan]
   • Quantitative financial analysis
   • Financial ratio calculation
   • Balance sheet health assessment
   • Uses: FinancialDataTool (MCP pattern)

[cyan]3. Qualitative Signal Agent[/cyan]
   • Sentiment analysis from unstructured data
   • Risk assessment and theme extraction
   • Employee and market sentiment
   • Uses: QualitativeDataTool (MCP pattern)

[cyan]4. Context Engine Agent[/cyan]
   • Historical pattern matching
   • Context rule application
   • Confidence scoring based on historical accuracy
   • Uses: ContextRuleTool (MCP pattern)

[bold]Data Flow:[/bold]
Orchestrator → (Financial + Qualitative) → Conflict Detection →
Context Engine → Final Recommendation

[bold]Key Features:[/bold]
• Parallel agent execution for performance
• Structured A2A message protocol
• Complete audit trail
• Context-aware decision making
        """

        self.console.print(Panel(
            arch_text.strip(),
            box=box.DOUBLE,
            border_style="blue",
            title="[bold]InvestmentIQ MVAS Architecture[/bold]"
        ))

    async def test_individual_agents(self):
        """Test individual agent functionality"""
        if not self.system:
            self.ui.print_info("Initializing system...")
            self.system = InvestmentIQSystemEnhanced()

        self.ui.print_section("Agent Testing", "bold yellow")

        agents_table = Table(show_header=True, header_style="bold magenta")
        agents_table.add_column("#", style="cyan")
        agents_table.add_column("Agent", style="white")

        agents_table.add_row("1", "Financial Analyst Agent")
        agents_table.add_row("2", "Qualitative Signal Agent")
        agents_table.add_row("3", "Context Engine Agent")
        agents_table.add_row("0", "Back to main menu")

        self.console.print(agents_table)

        choice = Prompt.ask(
            "[cyan]Select agent to test[/cyan]",
            choices=["1", "2", "3", "0"],
            default="0"
        )

        if choice == "0":
            return

        company_id = "COMPANY_X"

        if choice == "1":
            self.ui.print_info("Testing Financial Analyst Agent...")
            response = await self.system.financial_agent.process({
                "company_id": company_id,
                "analysis_type": "comprehensive"
            })
            self.ui.print_json_data(response.to_dict(), "Financial Analysis Results")

        elif choice == "2":
            self.ui.print_info("Testing Qualitative Signal Agent...")
            response = await self.system.qualitative_agent.process({
                "company_id": company_id
            })
            self.ui.print_json_data(response.to_dict(), "Qualitative Analysis Results")

        elif choice == "3":
            self.ui.print_info("Testing Context Engine Agent...")
            response = await self.system.context_agent.process({
                "scenario_type": "contrarian_opportunity",
                "context": {
                    "financial_health": "Strong",
                    "sentiment": "Very Negative",
                    "sector": "technology",
                    "gross_margin": 50
                }
            })
            self.ui.print_json_data(response.to_dict(), "Context Engine Results")

    def view_sample_data(self):
        """Display sample data"""
        self.ui.print_section("Sample Mock Data", "bold green")

        data_files = [
            ("Financial Data", "data/mock/COMPANY_X_financial.json"),
            ("Qualitative Data", "data/mock/COMPANY_X_qualitative.txt"),
            ("Context Rules", "data/mock/context_rules.json")
        ]

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("#", style="cyan")
        table.add_column("Data Type", style="white")
        table.add_column("File", style="dim")

        for i, (dtype, file) in enumerate(data_files, 1):
            table.add_row(str(i), dtype, file)
        table.add_row("0", "Back to main menu", "")

        self.console.print(table)

        choice = Prompt.ask(
            "[cyan]Select data to view[/cyan]",
            choices=["1", "2", "3", "0"],
            default="0"
        )

        if choice == "0":
            return

        _, file_path = data_files[int(choice) - 1]
        file_path = Path(file_path)

        if file_path.suffix == ".json":
            with open(file_path, 'r') as f:
                data = json.load(f)
            self.ui.print_json_data(data, file_path.name)
        else:
            with open(file_path, 'r') as f:
                content = f.read()
            self.console.print(Panel(
                content[:1000] + "..." if len(content) > 1000 else content,
                title=file_path.name,
                border_style="green"
            ))

    def view_last_results(self):
        """View last analysis results"""
        if not self.last_results:
            self.ui.print_warning("No analysis has been run yet!")
            return

        self.ui.print_section("Last Analysis Results", "bold green")
        data = self.last_results.get("data", {})
        self.ui.print_analysis_results(data)

    def show_metrics(self):
        """Show system performance metrics"""
        self.ui.print_section("System Performance Metrics", "bold cyan")

        metrics = {
            "Agents": 4,
            "Tools (MCP)": 3,
            "Analysis Time": "< 1 second",
            "Memory Usage": "~50 MB",
            "Python Version": "3.8+",
            "Dependencies": "Minimal (uv managed)",
            "Test Coverage": "80%+",
            "Documentation": "8 guides"
        }

        self.ui.print_metrics_dashboard(metrics)

    def show_about(self):
        """Show about information"""
        about_text = """
[bold cyan]InvestmentIQ MVAS[/bold cyan]
[dim]Multi-Agent Financial Intelligence System[/dim]

[bold]Version:[/bold] 1.0.0
[bold]Team:[/bold] Group 2 (Rajesh, Rui, Ameya, Mohammed, Murthy, Amine)
[bold]Project:[/bold] Agentic AI Capstone

[bold]Key Features:[/bold]
• 4-agent architecture (Orchestrator, Financial, Qualitative, Context)
• Agent-to-Agent (A2A) communication protocol
• Tool use with MCP pattern
• Context engineering with historical patterns
• Production-grade code quality

[bold]Technologies:[/bold]
• Python 3.8+ with async/await
• uv package manager
• Rich console UI
• Type-safe codebase
• Comprehensive testing

[bold]Documentation:[/bold]
• README.md - Complete guide
• QUICKSTART.md - 5-minute setup
• SETUP.md - Production deployment
• ARCHITECTURE.md - System design
• And 4 more guides...

[bold]License:[/bold] MIT
        """

        self.console.print(Panel(
            about_text.strip(),
            box=box.DOUBLE,
            border_style="cyan",
            title="[bold]About InvestmentIQ[/bold]"
        ))

    async def run(self):
        """Main interactive loop"""
        self.show_banner()

        while True:
            print()
            self.show_main_menu()
            print()

            choice = Prompt.ask(
                "[bold cyan]Select an option[/bold cyan]",
                choices=["0", "1", "2", "3", "4", "5", "6", "7"],
                default="1"
            )

            print()

            if choice == "0":
                self.ui.print_success("Thank you for using InvestmentIQ!")
                break
            elif choice == "1":
                await self.run_analysis()
            elif choice == "2":
                self.view_architecture()
            elif choice == "3":
                await self.test_individual_agents()
            elif choice == "4":
                self.view_sample_data()
            elif choice == "5":
                self.view_last_results()
            elif choice == "6":
                self.show_metrics()
            elif choice == "7":
                self.show_about()

            if not Confirm.ask("\n[dim]Continue?[/dim]", default=True):
                self.ui.print_success("Goodbye!")
                break


def main():
    """Entry point for interactive CLI"""
    cli = InteractiveCLI()
    asyncio.run(cli.run())


if __name__ == "__main__":
    main()
