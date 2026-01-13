# =============================================================================
# ActuallyOpenAI Production Deployment Script (PowerShell)
# For Windows environments
# =============================================================================

param(
    [Parameter(Position=0)]
    [ValidateSet('deploy', 'build', 'push', 'apply', 'verify', 'rollback', 'logs')]
    [string]$Command = 'deploy',
    
    [string]$Environment = 'production',
    [string]$Version = 'latest',
    [string]$Namespace = 'actuallyopenai',
    [string]$Registry = 'ghcr.io/actuallyopenai',
    [switch]$NoPush,
    [switch]$Help
)

# Configuration
$ErrorActionPreference = 'Stop'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

# Try to get version from git
if ($Version -eq 'latest') {
    try {
        $Version = git describe --tags --always 2>$null
        if (-not $Version) { $Version = 'dev' }
    } catch {
        $Version = 'dev'
    }
}

# Logging functions
function Write-Info { Write-Host "[INFO] $args" -ForegroundColor Blue }
function Write-Success { Write-Host "[SUCCESS] $args" -ForegroundColor Green }
function Write-Warn { Write-Host "[WARN] $args" -ForegroundColor Yellow }
function Write-Error { Write-Host "[ERROR] $args" -ForegroundColor Red }

function Show-Banner {
    Write-Host @"
    _        _               _ _        ___                   _    ___ 
   / \   ___| |_ _   _  __ _| | |_   _ / _ \ _ __   ___ _ __ | |  |_ _|
  / _ \ / __| __| | | |/ _` | | | | | | | | | '_ \ / _ \ '_ \| |   | | 
 / ___ \ (__| |_| |_| | (_| | | | |_| | |_| | |_) |  __/ | | |_|   | | 
/_/   \_\___|\__|\__,_|\__,_|_|_|\__, |\___/| .__/ \___|_| |_(_)  |___|
                                |___/      |_|                        
"@ -ForegroundColor Green
    Write-Host "Deployment Script v1.0.0"
    Write-Host "========================`n"
}

function Show-Usage {
    Write-Host @"
Usage: .\deploy.ps1 [command] [options]

Commands:
  deploy      Full deployment (build, push, deploy)
  build       Build Docker images only
  push        Push images to registry
  apply       Apply deployment (skip build)
  verify      Verify deployment health
  rollback    Rollback to previous version
  logs        Show container logs

Options:
  -Environment  Environment (staging|production)
  -Version      Version tag
  -Namespace    Kubernetes namespace
  -NoPush       Skip pushing images
  -Help         Show this help

Examples:
  .\deploy.ps1 deploy -Environment production
  .\deploy.ps1 build -Version v1.2.3
  .\deploy.ps1 verify
"@
}

function Test-Prerequisites {
    Write-Info "Checking prerequisites..."
    
    $missing = @()
    
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        $missing += "docker"
    }
    
    if ($missing.Count -gt 0) {
        Write-Error "Missing required tools: $($missing -join ', ')"
        exit 1
    }
    
    # Check Docker daemon
    try {
        docker info 2>&1 | Out-Null
    } catch {
        Write-Error "Docker daemon is not running"
        exit 1
    }
    
    # Check for Kubernetes
    $script:UseK8s = $false
    if (Get-Command kubectl -ErrorAction SilentlyContinue) {
        try {
            kubectl cluster-info 2>&1 | Out-Null
            $script:UseK8s = $true
        } catch {
            Write-Warn "Cannot connect to Kubernetes cluster"
        }
    }
    
    Write-Success "Prerequisites check passed"
}

function Build-Images {
    Write-Info "Building Docker images..."
    
    Push-Location $ProjectRoot
    
    try {
        # Build API
        Write-Info "Building API image..."
        docker build -f docker/api.Dockerfile -t "$Registry/api:$Version" .
        docker tag "$Registry/api:$Version" "$Registry/api:latest"
        
        # Build Worker
        Write-Info "Building Worker image..."
        docker build -f docker/worker.Dockerfile -t "$Registry/worker:$Version" .
        docker tag "$Registry/worker:$Version" "$Registry/worker:latest"
        
        # Build Dashboard
        Write-Info "Building Dashboard image..."
        docker build -f docker/dashboard.Dockerfile -t "$Registry/dashboard:$Version" .
        docker tag "$Registry/dashboard:$Version" "$Registry/dashboard:latest"
        
        # Build Orchestrator
        Write-Info "Building Orchestrator image..."
        docker build -f docker/orchestrator.Dockerfile -t "$Registry/orchestrator:$Version" .
        docker tag "$Registry/orchestrator:$Version" "$Registry/orchestrator:latest"
        
        Write-Success "All images built successfully"
    }
    finally {
        Pop-Location
    }
}

function Push-Images {
    Write-Info "Pushing images to registry..."
    
    docker push "$Registry/api:$Version"
    docker push "$Registry/api:latest"
    docker push "$Registry/worker:$Version"
    docker push "$Registry/worker:latest"
    docker push "$Registry/dashboard:$Version"
    docker push "$Registry/dashboard:latest"
    docker push "$Registry/orchestrator:$Version"
    docker push "$Registry/orchestrator:latest"
    
    Write-Success "All images pushed to registry"
}

function New-Secrets {
    Write-Info "Generating secrets..."
    
    # Generate random secrets
    $JwtSecret = -join ((65..90) + (97..122) + (48..57) | Get-Random -Count 32 | ForEach-Object {[char]$_})
    $DbPassword = -join ((65..90) + (97..122) + (48..57) | Get-Random -Count 16 | ForEach-Object {[char]$_})
    $RedisPassword = -join ((65..90) + (97..122) + (48..57) | Get-Random -Count 16 | ForEach-Object {[char]$_})
    
    # Create .env file
    @"
# Auto-generated secrets - DO NOT COMMIT
JWT_SECRET=$JwtSecret
POSTGRES_PASSWORD=$DbPassword
REDIS_PASSWORD=$RedisPassword
ENVIRONMENT=$Environment
VERSION=$Version
"@ | Set-Content -Path "$ProjectRoot\.env"
    
    Write-Success "Secrets generated"
}

function Deploy-Compose {
    Write-Info "Deploying with Docker Compose..."
    
    Push-Location $ProjectRoot
    
    try {
        # Stop existing containers
        docker-compose -f docker-compose.prod.yml down --remove-orphans 2>$null
        
        # Start services
        docker-compose -f docker-compose.prod.yml up -d
        
        # Wait for services
        Write-Info "Waiting for services to start..."
        Start-Sleep -Seconds 30
        
        # Health check
        try {
            $health = Invoke-RestMethod -Uri "http://localhost:8000/health" -TimeoutSec 10
            if ($health.status -eq "healthy") {
                Write-Success "API is healthy"
            }
        } catch {
            Write-Warn "API health check failed, checking logs..."
            docker-compose -f docker-compose.prod.yml logs --tail=50 api
        }
        
        Write-Success "Docker Compose deployment complete"
        
        # Show status
        Write-Host "`nContainer Status:"
        docker-compose -f docker-compose.prod.yml ps
    }
    finally {
        Pop-Location
    }
}

function Test-Deployment {
    Write-Info "Verifying deployment..."
    
    $apiUrl = "http://localhost:8000"
    
    # Test health endpoint
    try {
        $health = Invoke-RestMethod -Uri "$apiUrl/health" -TimeoutSec 10
        if ($health.status -eq "healthy") {
            Write-Success "Health check passed"
        } else {
            Write-Error "Health check failed"
            return
        }
    } catch {
        Write-Error "Cannot connect to API"
        return
    }
    
    # Test API root
    try {
        $root = Invoke-RestMethod -Uri "$apiUrl/" -TimeoutSec 10
        if ($root.name -like "*ActuallyOpenAI*") {
            Write-Success "API root check passed"
        }
    } catch {
        Write-Warn "API root check returned unexpected response"
    }
    
    Write-Success "Deployment verification complete"
    
    Write-Host @"

=========================================
  🎉 Deployment Successful!
=========================================

  API:       $apiUrl
  Docs:      $apiUrl/docs
  Dashboard: $apiUrl/dashboard

"@
}

function Invoke-Rollback {
    Write-Warn "Rolling back deployment..."
    
    Push-Location $ProjectRoot
    try {
        docker-compose -f docker-compose.prod.yml down
    }
    finally {
        Pop-Location
    }
    
    Write-Success "Rollback complete"
}

function Get-Logs {
    Push-Location $ProjectRoot
    try {
        docker-compose -f docker-compose.prod.yml logs -f
    }
    finally {
        Pop-Location
    }
}

# Main
if ($Help) {
    Show-Usage
    exit 0
}

Show-Banner

switch ($Command) {
    'deploy' {
        Test-Prerequisites
        Build-Images
        if (-not $NoPush) { Push-Images }
        New-Secrets
        Deploy-Compose
        Test-Deployment
    }
    'build' {
        Test-Prerequisites
        Build-Images
    }
    'push' {
        Push-Images
    }
    'apply' {
        Test-Prerequisites
        New-Secrets
        Deploy-Compose
    }
    'verify' {
        Test-Prerequisites
        Test-Deployment
    }
    'rollback' {
        Invoke-Rollback
    }
    'logs' {
        Get-Logs
    }
}
