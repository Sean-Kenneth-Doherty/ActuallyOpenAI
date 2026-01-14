"""
Web Dashboard for ActuallyOpenAI Contributors.

Features:
- Real-time earnings display
- Compute contribution stats
- AOAI token balance and dividends
- Worker node management
- Leaderboards
"""

import os
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Dict, Any, List
import secrets

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import structlog

logger = structlog.get_logger()


# =============================================================================
# Dashboard Data Models
# =============================================================================

class ContributorStats(BaseModel):
    """Contributor statistics."""
    wallet_address: str
    total_compute_hours: float
    tokens_earned: Decimal
    dividends_claimed: Decimal
    dividends_pending: Decimal
    rank: int
    worker_nodes: int
    active_since: datetime


class WorkerNode(BaseModel):
    """Worker node information."""
    id: str
    name: str
    status: str  # online, offline, training
    gpu_type: Optional[str]
    gpu_memory_gb: float
    cpu_cores: int
    ram_gb: float
    uptime_hours: float
    tasks_completed: int
    tokens_earned: Decimal
    last_seen: datetime


class PlatformStats(BaseModel):
    """Platform-wide statistics."""
    total_contributors: int
    active_workers: int
    total_compute_hours: float
    models_trained: int
    total_tokens_minted: Decimal
    total_dividends_distributed: Decimal
    api_revenue_24h: Decimal
    next_dividend_date: datetime


# =============================================================================
# Mock Data Store (Replace with real database)
# =============================================================================

class DashboardStore:
    """Dashboard data store."""
    
    def __init__(self):
        self.contributors: Dict[str, ContributorStats] = {}
        self.workers: Dict[str, WorkerNode] = {}
        
        # Initialize with demo data
        self._init_demo_data()
    
    def _init_demo_data(self):
        """Initialize demo data."""
        
        # Demo contributors
        demo_wallets = [
            "0x742d35Cc6634C0532925a3b844Bc9e7595f",
            "0x8ba1f109551bD432803012645Ac136ddd64",
            "0xdd870fA1b7C4700F2BD7f44238821C26f73",
        ]
        
        for i, wallet in enumerate(demo_wallets):
            self.contributors[wallet] = ContributorStats(
                wallet_address=wallet,
                total_compute_hours=1000 * (3 - i),
                tokens_earned=Decimal(str(10000 * (3 - i))),
                dividends_claimed=Decimal(str(500 * (3 - i))),
                dividends_pending=Decimal(str(100 * (3 - i))),
                rank=i + 1,
                worker_nodes=3 - i,
                active_since=datetime.utcnow() - timedelta(days=90 + i * 30)
            )
            
            # Add worker nodes for each contributor
            for j in range(3 - i):
                worker_id = f"worker-{wallet[:8]}-{j}"
                self.workers[worker_id] = WorkerNode(
                    id=worker_id,
                    name=f"Worker {j + 1}",
                    status="online" if j == 0 else "offline",
                    gpu_type="NVIDIA RTX 4090" if j < 2 else None,
                    gpu_memory_gb=24.0 if j < 2 else 0,
                    cpu_cores=16,
                    ram_gb=64,
                    uptime_hours=100 * (j + 1),
                    tasks_completed=50 * (j + 1),
                    tokens_earned=Decimal(str(1000 * (j + 1))),
                    last_seen=datetime.utcnow() - timedelta(minutes=j * 30)
                )


store = DashboardStore()


# =============================================================================
# HTML Templates (Inline for simplicity)
# =============================================================================

BASE_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ActuallyOpenAI - {title}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/alpinejs@3.x.x/dist/cdn.min.js" defer></script>
    <style>
        .gradient-bg {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }}
        .card {{
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }}
        .glow {{
            box-shadow: 0 0 20px rgba(102, 126, 234, 0.5);
        }}
    </style>
</head>
<body class="min-h-screen gradient-bg text-white">
    <nav class="bg-black/20 backdrop-blur-sm border-b border-white/10">
        <div class="max-w-7xl mx-auto px-4 py-4">
            <div class="flex justify-between items-center">
                <a href="/" class="text-2xl font-bold">🤖 ActuallyOpenAI</a>
                <div class="flex gap-4">
                    <a href="/dashboard" class="hover:text-purple-300">Dashboard</a>
                    <a href="/workers" class="hover:text-purple-300">Workers</a>
                    <a href="/leaderboard" class="hover:text-purple-300">Leaderboard</a>
                    <a href="/docs" class="hover:text-purple-300">API Docs</a>
                </div>
            </div>
        </div>
    </nav>
    
    <main class="max-w-7xl mx-auto px-4 py-8">
        {content}
    </main>
    
    <footer class="mt-auto py-8 text-center text-white/60">
        <p>ActuallyOpenAI - Decentralized AI for Everyone</p>
        <p class="text-sm mt-2">Built with ❤️ by the community</p>
    </footer>
</body>
</html>
"""

HOME_CONTENT = """
<div class="text-center mb-12">
    <h1 class="text-5xl font-bold mb-4">Decentralized AI Training</h1>
    <p class="text-xl text-white/80 mb-8">
        Contribute compute. Earn rewards. Own the future of AI.
    </p>
    <div class="flex justify-center gap-4">
        <a href="/dashboard" class="px-6 py-3 bg-white text-purple-600 rounded-lg font-semibold hover:bg-purple-100 transition">
            View Dashboard
        </a>
        <a href="https://github.com/actuallyopenai" class="px-6 py-3 border border-white rounded-lg hover:bg-white/10 transition">
            GitHub
        </a>
    </div>
</div>

<div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-12">
    <div class="card rounded-xl p-6 text-center">
        <div class="text-4xl mb-2">🌍</div>
        <div class="text-3xl font-bold">{total_contributors}</div>
        <div class="text-white/60">Contributors</div>
    </div>
    <div class="card rounded-xl p-6 text-center">
        <div class="text-4xl mb-2">💻</div>
        <div class="text-3xl font-bold">{active_workers}</div>
        <div class="text-white/60">Active Workers</div>
    </div>
    <div class="card rounded-xl p-6 text-center">
        <div class="text-4xl mb-2">⚡</div>
        <div class="text-3xl font-bold">{compute_hours}</div>
        <div class="text-white/60">Compute Hours</div>
    </div>
    <div class="card rounded-xl p-6 text-center">
        <div class="text-4xl mb-2">🪙</div>
        <div class="text-3xl font-bold">{tokens_minted}</div>
        <div class="text-white/60">AOAI Minted</div>
    </div>
</div>

<div class="card rounded-xl p-8 mb-12">
    <h2 class="text-2xl font-bold mb-6">How It Works</h2>
    <div class="grid grid-cols-1 md:grid-cols-3 gap-8">
        <div>
            <div class="text-4xl mb-4">1️⃣</div>
            <h3 class="text-xl font-semibold mb-2">Contribute Compute</h3>
            <p class="text-white/70">Run a worker node with your GPU or CPU. Your compute helps train open AI models.</p>
        </div>
        <div>
            <div class="text-4xl mb-4">2️⃣</div>
            <h3 class="text-xl font-semibold mb-2">Earn AOAI Tokens</h3>
            <p class="text-white/70">Receive AOAI tokens proportional to your compute contribution. Fully transparent on-chain.</p>
        </div>
        <div>
            <div class="text-4xl mb-4">3️⃣</div>
            <h3 class="text-xl font-semibold mb-2">Receive Dividends</h3>
            <p class="text-white/70">API revenue is distributed to token holders. The more you contribute, the more you earn.</p>
        </div>
    </div>
</div>

<div class="card rounded-xl p-8">
    <h2 class="text-2xl font-bold mb-6">Quick Start</h2>
    <div class="bg-black/30 rounded-lg p-4 font-mono text-sm">
        <p class="text-green-400"># Install ActuallyOpenAI</p>
        <p>pip install actuallyopenai</p>
        <br>
        <p class="text-green-400"># Start a worker node</p>
        <p>aoai worker start --wallet YOUR_WALLET_ADDRESS</p>
        <br>
        <p class="text-green-400"># Check your earnings</p>
        <p>aoai earnings --wallet YOUR_WALLET_ADDRESS</p>
    </div>
</div>
"""

DASHBOARD_CONTENT = """
<div class="mb-8">
    <h1 class="text-3xl font-bold mb-2">Contributor Dashboard</h1>
    <p class="text-white/70">Wallet: {wallet_address}</p>
</div>

<div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
    <div class="card rounded-xl p-6">
        <div class="text-white/60 mb-1">Total Compute</div>
        <div class="text-3xl font-bold">{compute_hours} hrs</div>
    </div>
    <div class="card rounded-xl p-6">
        <div class="text-white/60 mb-1">Tokens Earned</div>
        <div class="text-3xl font-bold text-green-400">{tokens_earned} AOAI</div>
    </div>
    <div class="card rounded-xl p-6">
        <div class="text-white/60 mb-1">Pending Dividends</div>
        <div class="text-3xl font-bold text-yellow-400">{dividends_pending} ETH</div>
    </div>
    <div class="card rounded-xl p-6">
        <div class="text-white/60 mb-1">Global Rank</div>
        <div class="text-3xl font-bold">#{rank}</div>
    </div>
</div>

<div class="card rounded-xl p-6 mb-8">
    <div class="flex justify-between items-center mb-4">
        <h2 class="text-xl font-bold">Your Worker Nodes</h2>
        <button class="px-4 py-2 bg-purple-600 rounded-lg hover:bg-purple-700 transition">
            + Add Worker
        </button>
    </div>
    <div class="overflow-x-auto">
        <table class="w-full">
            <thead>
                <tr class="text-left text-white/60">
                    <th class="pb-4">Name</th>
                    <th class="pb-4">Status</th>
                    <th class="pb-4">GPU</th>
                    <th class="pb-4">Uptime</th>
                    <th class="pb-4">Tasks</th>
                    <th class="pb-4">Earned</th>
                </tr>
            </thead>
            <tbody>
                {worker_rows}
            </tbody>
        </table>
    </div>
</div>

<div class="grid grid-cols-1 md:grid-cols-2 gap-6">
    <div class="card rounded-xl p-6">
        <h2 class="text-xl font-bold mb-4">Earnings History</h2>
        <div class="h-48 flex items-center justify-center text-white/40">
            📊 Chart coming soon
        </div>
    </div>
    <div class="card rounded-xl p-6">
        <h2 class="text-xl font-bold mb-4">Recent Activity</h2>
        <div class="space-y-3">
            <div class="flex justify-between">
                <span class="text-white/70">Task completed</span>
                <span class="text-green-400">+10 AOAI</span>
            </div>
            <div class="flex justify-between">
                <span class="text-white/70">Dividend claimed</span>
                <span class="text-yellow-400">0.1 ETH</span>
            </div>
            <div class="flex justify-between">
                <span class="text-white/70">Worker online</span>
                <span class="text-blue-400">Worker 1</span>
            </div>
        </div>
    </div>
</div>
"""

LEADERBOARD_CONTENT = """
<div class="mb-8">
    <h1 class="text-3xl font-bold mb-2">Contributor Leaderboard</h1>
    <p class="text-white/70">Top contributors by compute hours</p>
</div>

<div class="card rounded-xl overflow-hidden">
    <table class="w-full">
        <thead>
            <tr class="bg-black/20 text-left">
                <th class="p-4">Rank</th>
                <th class="p-4">Wallet</th>
                <th class="p-4">Compute Hours</th>
                <th class="p-4">Tokens Earned</th>
                <th class="p-4">Workers</th>
                <th class="p-4">Active Since</th>
            </tr>
        </thead>
        <tbody>
            {leaderboard_rows}
        </tbody>
    </table>
</div>
"""


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="ActuallyOpenAI Dashboard",
    description="Web dashboard for ActuallyOpenAI contributors",
    version="1.0.0"
)


def render_template(title: str, content: str) -> str:
    """Render a page with the base template."""
    return BASE_TEMPLATE.format(title=title, content=content)


@app.get("/", response_class=HTMLResponse)
async def home():
    """Home page."""
    
    # Calculate stats
    total_contributors = len(store.contributors)
    active_workers = len([w for w in store.workers.values() if w.status == "online"])
    compute_hours = sum(c.total_compute_hours for c in store.contributors.values())
    tokens_minted = sum(c.tokens_earned for c in store.contributors.values())
    
    content = HOME_CONTENT.format(
        total_contributors=total_contributors,
        active_workers=active_workers,
        compute_hours=f"{compute_hours:,.0f}",
        tokens_minted=f"{tokens_minted:,.0f}"
    )
    
    return render_template("Home", content)


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(wallet: str = "0x742d35Cc6634C0532925a3b844Bc9e7595f"):
    """Contributor dashboard."""
    
    # Get contributor stats
    contributor = store.contributors.get(wallet)
    if not contributor:
        # Create demo contributor if not found
        contributor = ContributorStats(
            wallet_address=wallet,
            total_compute_hours=0,
            tokens_earned=Decimal("0"),
            dividends_claimed=Decimal("0"),
            dividends_pending=Decimal("0"),
            rank=999,
            worker_nodes=0,
            active_since=datetime.utcnow()
        )
    
    # Get worker rows
    worker_rows = []
    for worker_id, worker in store.workers.items():
        if wallet[:8] in worker_id:
            status_color = {
                "online": "text-green-400",
                "offline": "text-red-400",
                "training": "text-yellow-400"
            }.get(worker.status, "text-white/60")
            
            worker_rows.append(f"""
                <tr class="border-t border-white/10">
                    <td class="py-4">{worker.name}</td>
                    <td class="py-4"><span class="{status_color}">● {worker.status}</span></td>
                    <td class="py-4">{worker.gpu_type or 'CPU Only'}</td>
                    <td class="py-4">{worker.uptime_hours:.1f} hrs</td>
                    <td class="py-4">{worker.tasks_completed}</td>
                    <td class="py-4 text-green-400">{worker.tokens_earned} AOAI</td>
                </tr>
            """)
    
    content = DASHBOARD_CONTENT.format(
        wallet_address=contributor.wallet_address[:20] + "...",
        compute_hours=f"{contributor.total_compute_hours:,.1f}",
        tokens_earned=f"{contributor.tokens_earned:,.0f}",
        dividends_pending=f"{contributor.dividends_pending:.4f}",
        rank=contributor.rank,
        worker_rows="".join(worker_rows) if worker_rows else "<tr><td colspan='6' class='py-8 text-center text-white/40'>No workers found. Add one to start earning!</td></tr>"
    )
    
    return render_template("Dashboard", content)


@app.get("/workers", response_class=HTMLResponse)
async def workers_page():
    """Workers management page."""
    
    content = """
    <div class="mb-8">
        <h1 class="text-3xl font-bold mb-2">Worker Management</h1>
        <p class="text-white/70">Manage your compute nodes</p>
    </div>
    
    <div class="card rounded-xl p-6 mb-8">
        <h2 class="text-xl font-bold mb-4">Add New Worker</h2>
        <div class="bg-black/30 rounded-lg p-4 font-mono text-sm">
            <p class="text-green-400"># On your compute machine:</p>
            <p>pip install actuallyopenai</p>
            <p>aoai worker start --wallet YOUR_WALLET --name "My GPU Server"</p>
        </div>
    </div>
    
    <div class="card rounded-xl p-6">
        <h2 class="text-xl font-bold mb-4">System Requirements</h2>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
                <h3 class="font-semibold mb-2">Minimum (CPU)</h3>
                <ul class="text-white/70 space-y-1">
                    <li>• 4 CPU cores</li>
                    <li>• 8 GB RAM</li>
                    <li>• 50 GB storage</li>
                    <li>• Stable internet</li>
                </ul>
            </div>
            <div>
                <h3 class="font-semibold mb-2">Recommended (GPU)</h3>
                <ul class="text-white/70 space-y-1">
                    <li>• NVIDIA GPU (RTX 3060+)</li>
                    <li>• 16+ GB RAM</li>
                    <li>• 100+ GB SSD</li>
                    <li>• CUDA 11.8+</li>
                </ul>
            </div>
        </div>
    </div>
    """
    
    return render_template("Workers", content)


@app.get("/leaderboard", response_class=HTMLResponse)
async def leaderboard():
    """Contributor leaderboard."""
    
    # Sort contributors by compute hours
    sorted_contributors = sorted(
        store.contributors.values(),
        key=lambda c: c.total_compute_hours,
        reverse=True
    )
    
    leaderboard_rows = []
    for i, contributor in enumerate(sorted_contributors):
        rank_emoji = ["🥇", "🥈", "🥉"][i] if i < 3 else f"#{i + 1}"
        
        leaderboard_rows.append(f"""
            <tr class="border-t border-white/10 hover:bg-white/5">
                <td class="p-4 text-2xl">{rank_emoji}</td>
                <td class="p-4 font-mono">{contributor.wallet_address[:20]}...</td>
                <td class="p-4">{contributor.total_compute_hours:,.0f} hrs</td>
                <td class="p-4 text-green-400">{contributor.tokens_earned:,.0f} AOAI</td>
                <td class="p-4">{contributor.worker_nodes}</td>
                <td class="p-4 text-white/60">{contributor.active_since.strftime('%Y-%m-%d')}</td>
            </tr>
        """)
    
    content = LEADERBOARD_CONTENT.format(
        leaderboard_rows="".join(leaderboard_rows)
    )
    
    return render_template("Leaderboard", content)


# =============================================================================
# API Endpoints for Dashboard Data
# =============================================================================

@app.get("/api/stats")
async def get_stats():
    """Get platform statistics."""
    return {
        "total_contributors": len(store.contributors),
        "active_workers": len([w for w in store.workers.values() if w.status == "online"]),
        "total_compute_hours": sum(c.total_compute_hours for c in store.contributors.values()),
        "total_tokens_minted": float(sum(c.tokens_earned for c in store.contributors.values())),
        "total_dividends_distributed": float(sum(c.dividends_claimed for c in store.contributors.values()))
    }


@app.get("/api/contributor/{wallet}")
async def get_contributor(wallet: str):
    """Get contributor details."""
    contributor = store.contributors.get(wallet)
    if not contributor:
        raise HTTPException(status_code=404, detail="Contributor not found")
    
    return contributor.model_dump()


@app.get("/api/leaderboard")
async def get_leaderboard(limit: int = 100):
    """Get leaderboard data."""
    sorted_contributors = sorted(
        store.contributors.values(),
        key=lambda c: c.total_compute_hours,
        reverse=True
    )[:limit]
    
    return [c.model_dump() for c in sorted_contributors]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
