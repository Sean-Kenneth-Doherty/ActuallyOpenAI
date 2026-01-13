#!/bin/bash
# =============================================================================
# ActuallyOpenAI Production Deployment Script
# Deploys the complete platform to Kubernetes or Docker Swarm
# =============================================================================

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
NAMESPACE="${NAMESPACE:-actuallyopenai}"
ENVIRONMENT="${ENVIRONMENT:-production}"
REGISTRY="${REGISTRY:-ghcr.io/actuallyopenai}"
VERSION="${VERSION:-$(git describe --tags --always 2>/dev/null || echo 'dev')}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Print banner
print_banner() {
    echo -e "${GREEN}"
    cat << 'EOF'
    _        _               _ _        ___                   _    ___ 
   / \   ___| |_ _   _  __ _| | |_   _ / _ \ _ __   ___ _ __ | |  |_ _|
  / _ \ / __| __| | | |/ _` | | | | | | | | | '_ \ / _ \ '_ \| |   | | 
 / ___ \ (__| |_| |_| | (_| | | | |_| | |_| | |_) |  __/ | | |_|   | | 
/_/   \_\___|\__|\__,_|\__,_|_|_|\__, |\___/| .__/ \___|_| |_(_)  |___|
                                |___/      |_|                        
EOF
    echo -e "${NC}"
    echo "Deployment Script v1.0.0"
    echo "========================"
    echo ""
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    local missing=()
    
    # Check for required tools
    command -v docker >/dev/null 2>&1 || missing+=("docker")
    command -v kubectl >/dev/null 2>&1 || missing+=("kubectl")
    command -v helm >/dev/null 2>&1 || log_warn "helm not found, will use kubectl only"
    
    if [ ${#missing[@]} -ne 0 ]; then
        log_error "Missing required tools: ${missing[*]}"
        exit 1
    fi
    
    # Check Docker daemon
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    # Check Kubernetes connection
    if ! kubectl cluster-info >/dev/null 2>&1; then
        log_warn "Cannot connect to Kubernetes cluster"
        read -p "Continue with Docker Compose deployment? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
        USE_K8S=false
    else
        USE_K8S=true
    fi
    
    log_success "Prerequisites check passed"
}

# Build Docker images
build_images() {
    log_info "Building Docker images..."
    
    cd "$PROJECT_ROOT"
    
    # Build API image
    log_info "Building API image..."
    docker build -f docker/api.Dockerfile -t "$REGISTRY/api:$VERSION" .
    docker tag "$REGISTRY/api:$VERSION" "$REGISTRY/api:latest"
    
    # Build Worker image
    log_info "Building Worker image..."
    docker build -f docker/worker.Dockerfile -t "$REGISTRY/worker:$VERSION" .
    docker tag "$REGISTRY/worker:$VERSION" "$REGISTRY/worker:latest"
    
    # Build Dashboard image
    log_info "Building Dashboard image..."
    docker build -f docker/dashboard.Dockerfile -t "$REGISTRY/dashboard:$VERSION" .
    docker tag "$REGISTRY/dashboard:$VERSION" "$REGISTRY/dashboard:latest"
    
    # Build Orchestrator image
    log_info "Building Orchestrator image..."
    docker build -f docker/orchestrator.Dockerfile -t "$REGISTRY/orchestrator:$VERSION" .
    docker tag "$REGISTRY/orchestrator:$VERSION" "$REGISTRY/orchestrator:latest"
    
    log_success "All images built successfully"
}

# Push images to registry
push_images() {
    log_info "Pushing images to registry..."
    
    docker push "$REGISTRY/api:$VERSION"
    docker push "$REGISTRY/api:latest"
    docker push "$REGISTRY/worker:$VERSION"
    docker push "$REGISTRY/worker:latest"
    docker push "$REGISTRY/dashboard:$VERSION"
    docker push "$REGISTRY/dashboard:latest"
    docker push "$REGISTRY/orchestrator:$VERSION"
    docker push "$REGISTRY/orchestrator:latest"
    
    log_success "All images pushed to registry"
}

# Generate secrets
generate_secrets() {
    log_info "Generating secrets..."
    
    # Generate random secrets if not provided
    JWT_SECRET="${JWT_SECRET:-$(openssl rand -hex 32)}"
    DB_PASSWORD="${DB_PASSWORD:-$(openssl rand -hex 16)}"
    REDIS_PASSWORD="${REDIS_PASSWORD:-$(openssl rand -hex 16)}"
    
    if [ "$USE_K8S" = true ]; then
        # Create Kubernetes secrets
        kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
        
        kubectl create secret generic aoai-secrets \
            --namespace="$NAMESPACE" \
            --from-literal=jwt-secret="$JWT_SECRET" \
            --from-literal=db-password="$DB_PASSWORD" \
            --from-literal=redis-password="$REDIS_PASSWORD" \
            --dry-run=client -o yaml | kubectl apply -f -
    else
        # Create .env file for Docker Compose
        cat > "$PROJECT_ROOT/.env" << EOF
# Auto-generated secrets - DO NOT COMMIT
JWT_SECRET=$JWT_SECRET
POSTGRES_PASSWORD=$DB_PASSWORD
REDIS_PASSWORD=$REDIS_PASSWORD
ENVIRONMENT=$ENVIRONMENT
VERSION=$VERSION
EOF
        chmod 600 "$PROJECT_ROOT/.env"
    fi
    
    log_success "Secrets generated"
}

# Deploy to Kubernetes
deploy_kubernetes() {
    log_info "Deploying to Kubernetes..."
    
    cd "$PROJECT_ROOT"
    
    # Generate manifests
    log_info "Generating Kubernetes manifests..."
    python -c "
from actuallyopenai.deploy.kubernetes import generate_all_manifests
manifests = generate_all_manifests('$NAMESPACE')
for name, content in manifests.items():
    with open(f'k8s-{name}.yaml', 'w') as f:
        f.write(content)
    print(f'Generated k8s-{name}.yaml')
"
    
    # Apply manifests in order
    log_info "Applying Kubernetes manifests..."
    kubectl apply -f k8s-namespace.yaml
    kubectl apply -f k8s-configmap.yaml
    kubectl apply -f k8s-secrets.yaml 2>/dev/null || true  # May already exist
    kubectl apply -f k8s-postgres.yaml
    kubectl apply -f k8s-redis.yaml
    
    # Wait for databases
    log_info "Waiting for databases..."
    kubectl rollout status statefulset/postgres -n "$NAMESPACE" --timeout=300s
    kubectl rollout status deployment/redis -n "$NAMESPACE" --timeout=120s
    
    # Deploy applications
    kubectl apply -f k8s-api.yaml
    kubectl apply -f k8s-dashboard.yaml
    kubectl apply -f k8s-orchestrator.yaml
    kubectl apply -f k8s-ingress.yaml
    kubectl apply -f k8s-network-policies.yaml
    kubectl apply -f k8s-pdb.yaml
    kubectl apply -f k8s-hpa.yaml
    
    # Wait for deployments
    log_info "Waiting for deployments..."
    kubectl rollout status deployment/aoai-api -n "$NAMESPACE" --timeout=300s
    kubectl rollout status deployment/aoai-dashboard -n "$NAMESPACE" --timeout=120s
    kubectl rollout status deployment/aoai-orchestrator -n "$NAMESPACE" --timeout=120s
    
    log_success "Kubernetes deployment complete"
    
    # Show status
    echo ""
    log_info "Deployment Status:"
    kubectl get all -n "$NAMESPACE"
}

# Deploy with Docker Compose
deploy_compose() {
    log_info "Deploying with Docker Compose..."
    
    cd "$PROJECT_ROOT"
    
    # Stop existing containers
    docker-compose -f docker-compose.prod.yml down --remove-orphans
    
    # Start services
    docker-compose -f docker-compose.prod.yml up -d
    
    # Wait for services
    log_info "Waiting for services to start..."
    sleep 30
    
    # Health check
    if curl -sf http://localhost:8000/health >/dev/null; then
        log_success "API is healthy"
    else
        log_warn "API health check failed, checking logs..."
        docker-compose -f docker-compose.prod.yml logs --tail=50 api
    fi
    
    log_success "Docker Compose deployment complete"
    
    # Show status
    echo ""
    log_info "Container Status:"
    docker-compose -f docker-compose.prod.yml ps
}

# Run database migrations
run_migrations() {
    log_info "Running database migrations..."
    
    if [ "$USE_K8S" = true ]; then
        kubectl exec -it deployment/aoai-api -n "$NAMESPACE" -- \
            python -m actuallyopenai.db.migrate
    else
        docker-compose -f docker-compose.prod.yml exec api \
            python -m actuallyopenai.db.migrate
    fi
    
    log_success "Migrations complete"
}

# Verify deployment
verify_deployment() {
    log_info "Verifying deployment..."
    
    local api_url
    if [ "$USE_K8S" = true ]; then
        api_url=$(kubectl get ingress -n "$NAMESPACE" -o jsonpath='{.items[0].spec.rules[0].host}' 2>/dev/null || echo "localhost")
        api_url="https://$api_url"
    else
        api_url="http://localhost:8000"
    fi
    
    # Test health endpoint
    if curl -sf "$api_url/health" | grep -q "healthy"; then
        log_success "Health check passed"
    else
        log_error "Health check failed"
        return 1
    fi
    
    # Test API root
    if curl -sf "$api_url/" | grep -q "ActuallyOpenAI"; then
        log_success "API root check passed"
    else
        log_warn "API root check returned unexpected response"
    fi
    
    log_success "Deployment verification complete"
    
    echo ""
    echo "========================================="
    echo "  🎉 Deployment Successful!"
    echo "========================================="
    echo ""
    echo "  API:       $api_url"
    echo "  Docs:      $api_url/docs"
    echo "  Dashboard: $api_url/dashboard"
    echo ""
}

# Rollback deployment
rollback() {
    log_warn "Rolling back deployment..."
    
    if [ "$USE_K8S" = true ]; then
        kubectl rollout undo deployment/aoai-api -n "$NAMESPACE"
        kubectl rollout undo deployment/aoai-dashboard -n "$NAMESPACE"
        kubectl rollout undo deployment/aoai-orchestrator -n "$NAMESPACE"
    else
        docker-compose -f docker-compose.prod.yml down
        # Would need previous version tags for proper rollback
    fi
    
    log_success "Rollback complete"
}

# Print usage
usage() {
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  deploy      Full deployment (build, push, deploy)"
    echo "  build       Build Docker images only"
    echo "  push        Push images to registry"
    echo "  apply       Apply deployment (skip build)"
    echo "  verify      Verify deployment health"
    echo "  rollback    Rollback to previous version"
    echo "  logs        Show container logs"
    echo ""
    echo "Options:"
    echo "  --env=ENV        Environment (staging|production)"
    echo "  --version=VER    Version tag"
    echo "  --namespace=NS   Kubernetes namespace"
    echo "  --no-push        Skip pushing images"
    echo ""
    echo "Examples:"
    echo "  $0 deploy --env=production"
    echo "  $0 build --version=v1.2.3"
    echo "  $0 verify"
}

# Main
main() {
    print_banner
    
    local command="${1:-deploy}"
    shift || true
    
    # Parse options
    for arg in "$@"; do
        case $arg in
            --env=*)
                ENVIRONMENT="${arg#*=}"
                ;;
            --version=*)
                VERSION="${arg#*=}"
                ;;
            --namespace=*)
                NAMESPACE="${arg#*=}"
                ;;
            --no-push)
                NO_PUSH=true
                ;;
            --help|-h)
                usage
                exit 0
                ;;
        esac
    done
    
    case "$command" in
        deploy)
            check_prerequisites
            build_images
            [ "${NO_PUSH:-false}" != true ] && push_images
            generate_secrets
            if [ "$USE_K8S" = true ]; then
                deploy_kubernetes
            else
                deploy_compose
            fi
            verify_deployment
            ;;
        build)
            check_prerequisites
            build_images
            ;;
        push)
            push_images
            ;;
        apply)
            check_prerequisites
            generate_secrets
            if [ "$USE_K8S" = true ]; then
                deploy_kubernetes
            else
                deploy_compose
            fi
            ;;
        verify)
            check_prerequisites
            verify_deployment
            ;;
        rollback)
            check_prerequisites
            rollback
            ;;
        logs)
            if [ "$USE_K8S" = true ]; then
                kubectl logs -f -l app=aoai -n "$NAMESPACE" --all-containers
            else
                docker-compose -f docker-compose.prod.yml logs -f
            fi
            ;;
        *)
            log_error "Unknown command: $command"
            usage
            exit 1
            ;;
    esac
}

main "$@"
