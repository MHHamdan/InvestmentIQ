"""
Enhanced UI Components for InvestmentIQ MVAS

Provides rich console output, progress tracking, and interactive visualizations
for production-grade terminal interfaces.
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.tree import Tree
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich import box
from rich.layout import Layout
from rich.live import Live
from tabulate import tabulate
from typing import Dict, Any, List
import json
from datetime import datetime


class InvestmentIQUI:
    """Enhanced UI components for InvestmentIQ system"""

    def __init__(self):
        self.console = Console()

    def print_header(self, title: str, subtitle: str = ""):
        """Print styled header"""
        header_text = f"[bold cyan]{title}[/bold cyan]"
        if subtitle:
            header_text += f"\n[dim]{subtitle}[/dim]"

        self.console.print(Panel(
            header_text,
            box=box.DOUBLE,
            border_style="cyan",
            padding=(1, 2)
        ))

    def print_section(self, title: str, style: str = "bold yellow"):
        """Print section header"""
        self.console.print(f"\n[{style}]{'─' * 80}[/{style}]")
        self.console.print(f"[{style}]{title}[/{style}]")
        self.console.print(f"[{style}]{'─' * 80}[/{style}]\n")

    def print_agent_status(self, agents: List[str]):
        """Print agent initialization status"""
        table = Table(
            show_header=True,
            header_style="bold magenta",
            box=box.ROUNDED,
            title="[bold]Multi-Agent System Initialization[/bold]"
        )

        table.add_column("Agent", style="cyan", width=30)
        table.add_column("Status", style="green", width=15)
        table.add_column("Role", style="yellow")

        agent_roles = {
            "Financial Analyst Agent": "Quantitative Analysis",
            "Qualitative Signal Agent": "Sentiment & Risk Analysis",
            "Context Engine Agent": "Pattern Matching & Rules",
            "Strategic Orchestrator": "Workflow Coordination"
        }

        for agent in agents:
            role = agent_roles.get(agent, "Specialized Agent")
            table.add_row(agent, "✓ Ready", role)

        self.console.print(table)

    def print_workflow_step(self, step: int, total: int, description: str, status: str = "in_progress"):
        """Print workflow step with progress"""
        status_icons = {
            "in_progress": "⏳",
            "completed": "✅",
            "failed": "❌"
        }

        status_colors = {
            "in_progress": "yellow",
            "completed": "green",
            "failed": "red"
        }

        icon = status_icons.get(status, "•")
        color = status_colors.get(status, "white")

        self.console.print(
            f"[bold {color}]Step {step}/{total}:[/bold {color}] "
            f"{icon} {description}"
        )

    def print_analysis_results(self, results: Dict[str, Any]):
        """Print comprehensive analysis results"""
        recommendation = results.get("recommendation", {})

        # Main recommendation panel
        rec_text = f"""
[bold green]Investment Recommendation[/bold green]

[cyan]Company:[/cyan] {results.get('company_id', 'N/A')}
[cyan]Action:[/cyan] [bold yellow]{recommendation.get('action', 'N/A')}[/bold yellow]
[cyan]Position Size:[/cyan] {recommendation.get('position_size', 'N/A')}
[cyan]Confidence:[/cyan] [bold]{recommendation.get('confidence', 0) * 100:.0f}%[/bold]

[bold]Reasoning:[/bold]
{recommendation.get('reasoning', 'N/A')}
        """

        self.console.print(Panel(
            rec_text.strip(),
            box=box.DOUBLE,
            border_style="green",
            title="[bold]Final Recommendation[/bold]",
            title_align="left"
        ))

        # Supporting factors table
        supporting = recommendation.get('supporting_factors', {})
        if supporting:
            factors_table = Table(
                show_header=True,
                header_style="bold cyan",
                box=box.SIMPLE,
                title="Supporting Analysis"
            )

            factors_table.add_column("Factor", style="cyan")
            factors_table.add_column("Value", style="yellow")

            for key, value in supporting.items():
                factors_table.add_row(
                    key.replace('_', ' ').title(),
                    str(value)
                )

            self.console.print(factors_table)

        # Context rule information
        context_rule = recommendation.get('context_rule_applied')
        if context_rule:
            rule_text = f"""
[bold]Historical Pattern Applied:[/bold]

[cyan]Rule ID:[/cyan] {context_rule.get('rule_id', 'N/A')}
[cyan]Scenario:[/cyan] {context_rule.get('scenario_type', 'N/A').replace('_', ' ').title()}
[cyan]Historical Accuracy:[/cyan] {context_rule.get('historical_accuracy', 0) * 100:.0f}%
[cyan]Sample Size:[/cyan] {context_rule.get('historical_occurrences', 'N/A')} cases

[bold]Description:[/bold]
{context_rule.get('description', 'N/A')}
            """

            self.console.print(Panel(
                rule_text.strip(),
                box=box.ROUNDED,
                border_style="blue",
                title="Context Engine Analysis"
            ))

    def print_workflow_tree(self, workflow_log: List[Dict[str, Any]]):
        """Print workflow execution as a tree"""
        tree = Tree(
            "[bold cyan]📊 Workflow Execution Tree[/bold cyan]",
            guide_style="dim"
        )

        for step in workflow_log:
            step_name = step.get('step', 'Unknown')
            timestamp = step.get('timestamp', 'N/A')

            # Format step name
            formatted_name = step_name.replace('_', ' ').title()

            # Add to tree
            branch = tree.add(f"[yellow]{formatted_name}[/yellow] [dim]({timestamp})[/dim]")

            # Add data if available
            data = step.get('data', {})
            if isinstance(data, dict) and data:
                for key, value in list(data.items())[:3]:  # Show first 3 items
                    if isinstance(value, (str, int, float, bool)):
                        branch.add(f"[cyan]{key}:[/cyan] {value}")

        self.console.print(tree)

    def print_a2a_communication(self, messages: List[Dict[str, Any]]):
        """Print agent-to-agent communication log"""
        table = Table(
            show_header=True,
            header_style="bold magenta",
            box=box.ROUNDED,
            title="[bold]Agent-to-Agent Communication Log[/bold]"
        )

        table.add_column("#", style="dim", width=4)
        table.add_column("From", style="cyan", width=20)
        table.add_column("To", style="green", width=20)
        table.add_column("Type", style="yellow", width=12)
        table.add_column("Timestamp", style="dim", width=20)

        for i, msg in enumerate(messages, 1):
            table.add_row(
                str(i),
                msg.get('sender', 'Unknown'),
                msg.get('receiver', 'Unknown'),
                msg.get('message_type', 'Unknown'),
                msg.get('timestamp', 'N/A')[-8:]  # Show only time
            )

        self.console.print(table)

    def print_conflict_detection(self, conflict: Dict[str, Any]):
        """Print conflict detection results"""
        if not conflict.get('has_conflict'):
            self.console.print(
                Panel(
                    "[green]✓ No conflicts detected - signals are aligned[/green]",
                    box=box.ROUNDED,
                    border_style="green"
                )
            )
            return

        conflict_text = f"""
[bold red]⚠️  Conflict Detected[/bold red]

[cyan]Type:[/cyan] {conflict.get('conflict_type', 'Unknown').replace('_', ' ').title()}
[cyan]Severity:[/cyan] {conflict.get('severity', 'Unknown').upper()}

[bold]Signals:[/bold]
• Financial: [green]{conflict.get('financial_signal', 'N/A')}[/green]
• Qualitative: [red]{conflict.get('qualitative_signal', 'N/A')}[/red]

[bold]Description:[/bold]
{conflict.get('description', 'No description available')}
        """

        self.console.print(Panel(
            conflict_text.strip(),
            box=box.HEAVY,
            border_style="red",
            title="[bold]Signal Conflict Analysis[/bold]"
        ))

    def create_progress_bar(self, total: int, description: str = "Processing"):
        """Create a progress bar context manager"""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console
        )

    def print_json_data(self, data: Dict[str, Any], title: str = "Data"):
        """Print JSON data with syntax highlighting"""
        json_str = json.dumps(data, indent=2)
        syntax = Syntax(json_str, "json", theme="monokai", line_numbers=True)

        self.console.print(Panel(
            syntax,
            title=f"[bold]{title}[/bold]",
            border_style="blue"
        ))

    def print_metrics_dashboard(self, metrics: Dict[str, Any]):
        """Print metrics in a dashboard layout"""
        table = Table(
            show_header=True,
            header_style="bold cyan",
            box=box.DOUBLE_EDGE,
            title="[bold]System Performance Metrics[/bold]"
        )

        table.add_column("Metric", style="cyan", width=30)
        table.add_column("Value", style="yellow", justify="right")
        table.add_column("Status", style="green", width=15)

        for key, value in metrics.items():
            # Determine status based on metric
            status = "✓ Good"
            if isinstance(value, (int, float)):
                if value < 0:
                    status = "⚠ Warning"

            table.add_row(
                key.replace('_', ' ').title(),
                str(value),
                status
            )

        self.console.print(table)

    def print_summary_box(self, title: str, content: str, style: str = "cyan"):
        """Print a summary box"""
        self.console.print(Panel(
            content,
            title=f"[bold]{title}[/bold]",
            border_style=style,
            box=box.ROUNDED
        ))

    def print_success(self, message: str):
        """Print success message"""
        self.console.print(f"[bold green]✓[/bold green] {message}")

    def print_warning(self, message: str):
        """Print warning message"""
        self.console.print(f"[bold yellow]⚠[/bold yellow] {message}")

    def print_error(self, message: str):
        """Print error message"""
        self.console.print(f"[bold red]✗[/bold red] {message}")

    def print_info(self, message: str):
        """Print info message"""
        self.console.print(f"[bold blue]ℹ[/bold blue] {message}")
