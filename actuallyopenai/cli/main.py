"""
ActuallyOpenAI CLI - Command-line interface for managing workers and viewing stats.
"""

import asyncio
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
import httpx

app = typer.Typer(
    name="aoai",
    help="ActuallyOpenAI - Distributed AI Training Network CLI"
)

console = Console()


# =============================================================================
# Worker Commands
# =============================================================================

worker_app = typer.Typer(help="Worker node management")
app.add_typer(worker_app, name="worker")


@worker_app.command("start")
def start_worker(
    wallet: str = typer.Option(..., "--wallet", "-w", help="Your wallet address for AOAI rewards"),
    orchestrator: str = typer.Option("http://localhost:8000", "--orchestrator", "-o", help="Orchestrator URL"),
    name: str = typer.Option(None, "--name", "-n", help="Worker name"),
    region: str = typer.Option("unknown", "--region", "-r", help="Your region"),
    cpu_only: bool = typer.Option(False, "--cpu", help="Use CPU only (no GPU)")
):
    """Start a worker node to contribute compute and earn AOAI tokens."""
    from actuallyopenai.worker.node import WorkerNode
    
    console.print(Panel.fit(
        "[bold blue]ActuallyOpenAI Worker Node[/bold blue]\n"
        f"Wallet: {wallet}\n"
        f"Orchestrator: {orchestrator}",
        title="Starting Worker"
    ))
    
    worker = WorkerNode(
        orchestrator_url=orchestrator,
        wallet_address=wallet,
        name=name,
        region=region,
        use_gpu=not cpu_only
    )
    
    asyncio.run(worker.run())


@worker_app.command("status")
def worker_status(
    orchestrator: str = typer.Option("http://localhost:8000", "--orchestrator", "-o")
):
    """Check status of all workers in the network."""
    async def _status():
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{orchestrator}/api/v1/workers")
                response.raise_for_status()
                workers = response.json()
                
                table = Table(title="Network Workers")
                table.add_column("ID", style="dim")
                table.add_column("Name")
                table.add_column("Status")
                table.add_column("Region")
                table.add_column("Compute Score")
                table.add_column("Tasks")
                table.add_column("Tokens Earned")
                
                for w in workers:
                    status_icon = "🟢" if w["status"] in ["online", "idle", "training"] else "🔴"
                    table.add_row(
                        w["id"][:8] + "...",
                        w["name"],
                        f"{status_icon} {w['status']}",
                        w["region"],
                        f"{w.get('compute_score', 0):.1f}",
                        str(w.get("total_tasks_completed", 0)),
                        f"{float(w.get('total_tokens_earned', 0)):.2f}"
                    )
                
                console.print(table)
                
            except Exception as e:
                console.print(f"[red]Error connecting to orchestrator: {e}[/red]")
    
    asyncio.run(_status())


# =============================================================================
# Network Commands
# =============================================================================

@app.command("stats")
def network_stats(
    orchestrator: str = typer.Option("http://localhost:8000", "--orchestrator", "-o")
):
    """View network-wide statistics."""
    async def _stats():
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{orchestrator}/api/v1/stats")
                response.raise_for_status()
                stats = response.json()
                
                console.print(Panel.fit(
                    f"[cyan]Workers Online:[/cyan] {stats['network']['online_workers']} / {stats['network']['total_workers']}\n"
                    f"[cyan]Total Compute Score:[/cyan] {stats['network']['total_compute_score']:.1f}\n"
                    f"\n"
                    f"[green]Active Jobs:[/green] {stats['training']['active_jobs']}\n"
                    f"[green]Pending Tasks:[/green] {stats['training']['pending_tasks']}\n"
                    f"[green]Running Tasks:[/green] {stats['training']['running_tasks']}\n"
                    f"[green]Completed Tasks:[/green] {stats['training']['completed_tasks']}\n"
                    f"\n"
                    f"[yellow]Total AOAI Distributed:[/yellow] {stats['rewards']['total_tokens_distributed']}\n"
                    f"[yellow]Total Compute Hours:[/yellow] {stats['rewards']['total_compute_hours']}",
                    title="[bold]ActuallyOpenAI Network Stats[/bold]"
                ))
                
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
    
    asyncio.run(_stats())


@app.command("leaderboard")
def leaderboard(
    orchestrator: str = typer.Option("http://localhost:8000", "--orchestrator", "-o"),
    limit: int = typer.Option(10, "--limit", "-l", help="Number of top contributors")
):
    """View top compute contributors."""
    async def _leaderboard():
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{orchestrator}/api/v1/stats/leaderboard",
                    params={"limit": limit}
                )
                response.raise_for_status()
                leaders = response.json()
                
                table = Table(title="🏆 Top Contributors")
                table.add_column("Rank", style="bold")
                table.add_column("Name")
                table.add_column("Wallet")
                table.add_column("AOAI Earned", justify="right")
                table.add_column("Tasks", justify="right")
                table.add_column("Compute Hours", justify="right")
                
                for entry in leaders:
                    rank_style = "gold1" if entry["rank"] == 1 else "silver" if entry["rank"] == 2 else "orange3" if entry["rank"] == 3 else ""
                    table.add_row(
                        f"[{rank_style}]#{entry['rank']}[/{rank_style}]" if rank_style else f"#{entry['rank']}",
                        entry["name"],
                        f"{entry['wallet'][:6]}...{entry['wallet'][-4:]}",
                        f"{float(entry['tokens_earned']):.2f}",
                        str(entry["tasks_completed"]),
                        f"{entry['compute_hours']:.1f}h"
                    )
                
                console.print(table)
                
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
    
    asyncio.run(_leaderboard())


# =============================================================================
# Wallet Commands
# =============================================================================

wallet_app = typer.Typer(help="Wallet and token management")
app.add_typer(wallet_app, name="wallet")


@wallet_app.command("balance")
def wallet_balance(
    wallet: str = typer.Argument(..., help="Wallet address to check"),
    rpc_url: str = typer.Option(None, "--rpc", help="Blockchain RPC URL")
):
    """Check AOAI token balance and pending dividends."""
    from actuallyopenai.blockchain.token_service import TokenService
    from actuallyopenai.config import get_settings
    
    async def _balance():
        settings = get_settings()
        rpc = rpc_url or settings.blockchain_rpc_url
        
        if not rpc:
            console.print("[yellow]No RPC URL configured. Set BLOCKCHAIN_RPC_URL in .env[/yellow]")
            return
        
        try:
            # This would connect to the actual blockchain
            console.print(Panel.fit(
                f"[cyan]Wallet:[/cyan] {wallet}\n"
                f"[green]AOAI Balance:[/green] Loading...\n"
                f"[yellow]Pending Dividends:[/yellow] Loading...",
                title="Wallet Balance"
            ))
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    asyncio.run(_balance())


@wallet_app.command("claim")
def claim_dividends(
    wallet: str = typer.Argument(..., help="Your wallet address"),
    private_key: str = typer.Option(..., "--key", "-k", help="Your private key", hide_input=True)
):
    """Claim pending dividends from the AOAI token contract."""
    console.print("[yellow]This would trigger a blockchain transaction to claim dividends.[/yellow]")
    console.print("[dim]Feature coming soon - requires contract deployment[/dim]")


# =============================================================================
# Job Commands
# =============================================================================

job_app = typer.Typer(help="Training job management")
app.add_typer(job_app, name="job")


@job_app.command("list")
def list_jobs(
    orchestrator: str = typer.Option("http://localhost:8000", "--orchestrator", "-o")
):
    """List all training jobs."""
    async def _list():
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{orchestrator}/api/v1/jobs")
                response.raise_for_status()
                jobs = response.json()
                
                table = Table(title="Training Jobs")
                table.add_column("ID", style="dim")
                table.add_column("Name")
                table.add_column("Status")
                table.add_column("Progress")
                table.add_column("Token Reward")
                
                for job in jobs:
                    status_colors = {
                        "pending": "yellow",
                        "running": "green",
                        "completed": "blue",
                        "failed": "red"
                    }
                    color = status_colors.get(job["status"], "white")
                    table.add_row(
                        job["id"][:8] + "...",
                        job["name"],
                        f"[{color}]{job['status']}[/{color}]",
                        f"{job.get('progress_percent', 0):.1f}%",
                        str(job.get("total_token_reward", 0))
                    )
                
                console.print(table)
                
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
    
    asyncio.run(_list())


@job_app.command("create")
def create_job(
    name: str = typer.Option(..., "--name", "-n", help="Job name"),
    model_id: str = typer.Option(..., "--model", "-m", help="Model ID to train"),
    epochs: int = typer.Option(10, "--epochs", "-e", help="Number of epochs"),
    reward: float = typer.Option(1000, "--reward", "-r", help="Total AOAI reward for this job"),
    orchestrator: str = typer.Option("http://localhost:8000", "--orchestrator", "-o")
):
    """Create a new distributed training job."""
    async def _create():
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{orchestrator}/api/v1/jobs",
                    params={
                        "name": name,
                        "model_id": model_id,
                        "total_epochs": epochs,
                        "token_reward": reward
                    }
                )
                response.raise_for_status()
                job = response.json()
                
                console.print(f"[green]✓ Job created successfully![/green]")
                console.print(f"  Job ID: {job['id']}")
                console.print(f"  Name: {job['name']}")
                console.print(f"  Reward: {job['total_token_reward']} AOAI")
                
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
    
    asyncio.run(_create())


# =============================================================================
# Server Commands
# =============================================================================

@app.command("serve")
def serve(
    component: str = typer.Argument("orchestrator", help="Component to run: orchestrator, api"),
    host: str = typer.Option("0.0.0.0", "--host", "-h"),
    port: int = typer.Option(None, "--port", "-p")
):
    """Start a server component."""
    import uvicorn
    
    if component == "orchestrator":
        port = port or 8000
        console.print(f"[green]Starting Orchestrator on {host}:{port}[/green]")
        uvicorn.run(
            "actuallyopenai.orchestrator.server:app",
            host=host,
            port=port,
            reload=True
        )
    elif component == "api":
        port = port or 8001
        console.print(f"[green]Starting Model API on {host}:{port}[/green]")
        uvicorn.run(
            "actuallyopenai.api.model_api:app",
            host=host,
            port=port,
            reload=True
        )
    else:
        console.print(f"[red]Unknown component: {component}[/red]")
        console.print("Available components: orchestrator, api")


# =============================================================================
# Info Command
# =============================================================================

@app.command("info")
def info():
    """Display information about ActuallyOpenAI."""
    from actuallyopenai import __version__
    
    console.print(Panel.fit(
        """[bold cyan]ActuallyOpenAI[/bold cyan]
        
[white]A truly decentralized, crowd-funded AI training platform.[/white]

[green]• Contribute compute power and earn AOAI tokens
• Token holders receive dividends from API revenue
• Community-owned AI models, transparent and open[/green]

[dim]Version: """ + __version__ + """[/dim]

[yellow]Quick Start:[/yellow]
  aoai worker start --wallet YOUR_WALLET
  aoai stats
  aoai leaderboard

[blue]Learn more: https://github.com/actuallyopenai[/blue]""",
        title="ℹ️  About",
        border_style="cyan"
    ))


# =============================================================================
# Entry Point
# =============================================================================

def main():
    """Main entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
