"""
Kubernetes Deployment Configuration for ActuallyOpenAI.

This module generates Kubernetes manifests for deploying the platform
at scale across multiple cloud providers.
"""

# =============================================================================
# Namespace
# =============================================================================

NAMESPACE = """
apiVersion: v1
kind: Namespace
metadata:
  name: actuallyopenai
  labels:
    app: actuallyopenai
"""

# =============================================================================
# ConfigMap
# =============================================================================

CONFIGMAP = """
apiVersion: v1
kind: ConfigMap
metadata:
  name: aoai-config
  namespace: actuallyopenai
data:
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"
  API_HOST: "0.0.0.0"
  API_PORT: "8000"
  DASHBOARD_PORT: "8080"
  ORCHESTRATOR_PORT: "9000"
  WORKER_PORT: "9001"
  
  # Database
  POSTGRES_HOST: "postgres-service"
  POSTGRES_PORT: "5432"
  POSTGRES_DB: "actuallyopenai"
  
  # Redis
  REDIS_HOST: "redis-service"
  REDIS_PORT: "6379"
  
  # Blockchain
  ETHEREUM_RPC: "https://mainnet.infura.io/v3/YOUR_KEY"
  CONTRACT_ADDRESS: "0x0000000000000000000000000000000000000000"
  
  # Model Storage
  MODEL_STORAGE_TYPE: "s3"
  S3_BUCKET: "actuallyopenai-models"
"""

# =============================================================================
# Secrets
# =============================================================================

SECRETS = """
apiVersion: v1
kind: Secret
metadata:
  name: aoai-secrets
  namespace: actuallyopenai
type: Opaque
stringData:
  POSTGRES_PASSWORD: "CHANGE_ME_IN_PRODUCTION"
  REDIS_PASSWORD: "CHANGE_ME_IN_PRODUCTION"
  JWT_SECRET: "CHANGE_ME_IN_PRODUCTION"
  API_KEY_SALT: "CHANGE_ME_IN_PRODUCTION"
  AWS_ACCESS_KEY_ID: "CHANGE_ME_IN_PRODUCTION"
  AWS_SECRET_ACCESS_KEY: "CHANGE_ME_IN_PRODUCTION"
  ETHEREUM_PRIVATE_KEY: "CHANGE_ME_IN_PRODUCTION"
"""

# =============================================================================
# PostgreSQL StatefulSet
# =============================================================================

POSTGRES = """
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgres-pvc
  namespace: actuallyopenai
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
  storageClassName: fast-ssd
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: actuallyopenai
spec:
  serviceName: postgres
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
        - name: postgres
          image: postgres:15
          ports:
            - containerPort: 5432
          env:
            - name: POSTGRES_DB
              valueFrom:
                configMapKeyRef:
                  name: aoai-config
                  key: POSTGRES_DB
            - name: POSTGRES_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: aoai-secrets
                  key: POSTGRES_PASSWORD
          volumeMounts:
            - name: postgres-storage
              mountPath: /var/lib/postgresql/data
          resources:
            requests:
              memory: "2Gi"
              cpu: "1"
            limits:
              memory: "4Gi"
              cpu: "2"
      volumes:
        - name: postgres-storage
          persistentVolumeClaim:
            claimName: postgres-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: postgres-service
  namespace: actuallyopenai
spec:
  selector:
    app: postgres
  ports:
    - port: 5432
      targetPort: 5432
"""

# =============================================================================
# Redis StatefulSet
# =============================================================================

REDIS = """
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: redis
  namespace: actuallyopenai
spec:
  serviceName: redis
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
        - name: redis
          image: redis:7-alpine
          ports:
            - containerPort: 6379
          args:
            - redis-server
            - --requirepass
            - $(REDIS_PASSWORD)
          env:
            - name: REDIS_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: aoai-secrets
                  key: REDIS_PASSWORD
          resources:
            requests:
              memory: "1Gi"
              cpu: "500m"
            limits:
              memory: "2Gi"
              cpu: "1"
---
apiVersion: v1
kind: Service
metadata:
  name: redis-service
  namespace: actuallyopenai
spec:
  selector:
    app: redis
  ports:
    - port: 6379
      targetPort: 6379
"""

# =============================================================================
# API Deployment
# =============================================================================

API_DEPLOYMENT = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: aoai-api
  namespace: actuallyopenai
  labels:
    app: aoai-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: aoai-api
  template:
    metadata:
      labels:
        app: aoai-api
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      containers:
        - name: api
          image: actuallyopenai/api:latest
          ports:
            - containerPort: 8000
          envFrom:
            - configMapRef:
                name: aoai-config
            - secretRef:
                name: aoai-secrets
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 5
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
          resources:
            requests:
              memory: "1Gi"
              cpu: "500m"
            limits:
              memory: "2Gi"
              cpu: "1"
---
apiVersion: v1
kind: Service
metadata:
  name: aoai-api-service
  namespace: actuallyopenai
spec:
  selector:
    app: aoai-api
  ports:
    - port: 80
      targetPort: 8000
  type: ClusterIP
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: aoai-api-hpa
  namespace: actuallyopenai
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: aoai-api
  minReplicas: 3
  maxReplicas: 50
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
"""

# =============================================================================
# Dashboard Deployment
# =============================================================================

DASHBOARD_DEPLOYMENT = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: aoai-dashboard
  namespace: actuallyopenai
spec:
  replicas: 2
  selector:
    matchLabels:
      app: aoai-dashboard
  template:
    metadata:
      labels:
        app: aoai-dashboard
    spec:
      containers:
        - name: dashboard
          image: actuallyopenai/dashboard:latest
          ports:
            - containerPort: 8080
          envFrom:
            - configMapRef:
                name: aoai-config
          resources:
            requests:
              memory: "512Mi"
              cpu: "250m"
            limits:
              memory: "1Gi"
              cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: aoai-dashboard-service
  namespace: actuallyopenai
spec:
  selector:
    app: aoai-dashboard
  ports:
    - port: 80
      targetPort: 8080
"""

# =============================================================================
# Orchestrator Deployment
# =============================================================================

ORCHESTRATOR_DEPLOYMENT = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: aoai-orchestrator
  namespace: actuallyopenai
spec:
  replicas: 1
  selector:
    matchLabels:
      app: aoai-orchestrator
  template:
    metadata:
      labels:
        app: aoai-orchestrator
    spec:
      containers:
        - name: orchestrator
          image: actuallyopenai/orchestrator:latest
          ports:
            - containerPort: 9000
          envFrom:
            - configMapRef:
                name: aoai-config
            - secretRef:
                name: aoai-secrets
          resources:
            requests:
              memory: "2Gi"
              cpu: "1"
            limits:
              memory: "4Gi"
              cpu: "2"
---
apiVersion: v1
kind: Service
metadata:
  name: aoai-orchestrator-service
  namespace: actuallyopenai
spec:
  selector:
    app: aoai-orchestrator
  ports:
    - port: 9000
      targetPort: 9000
"""

# =============================================================================
# Ingress
# =============================================================================

INGRESS = """
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: aoai-ingress
  namespace: actuallyopenai
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
    nginx.ingress.kubernetes.io/proxy-body-size: "10m"
spec:
  tls:
    - hosts:
        - api.actuallyopenai.com
        - dashboard.actuallyopenai.com
      secretName: aoai-tls
  rules:
    - host: api.actuallyopenai.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: aoai-api-service
                port:
                  number: 80
    - host: dashboard.actuallyopenai.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: aoai-dashboard-service
                port:
                  number: 80
"""

# =============================================================================
# Network Policies
# =============================================================================

NETWORK_POLICIES = """
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-ingress
  namespace: actuallyopenai
spec:
  podSelector: {}
  policyTypes:
    - Ingress
---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-api-ingress
  namespace: actuallyopenai
spec:
  podSelector:
    matchLabels:
      app: aoai-api
  policyTypes:
    - Ingress
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              name: ingress-nginx
      ports:
        - protocol: TCP
          port: 8000
---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-internal
  namespace: actuallyopenai
spec:
  podSelector: {}
  policyTypes:
    - Ingress
  ingress:
    - from:
        - podSelector: {}
"""

# =============================================================================
# Pod Disruption Budgets
# =============================================================================

PDB = """
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: aoai-api-pdb
  namespace: actuallyopenai
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: aoai-api
---
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: aoai-dashboard-pdb
  namespace: actuallyopenai
spec:
  minAvailable: 1
  selector:
    matchLabels:
      app: aoai-dashboard
"""

# =============================================================================
# GPU Worker DaemonSet (for GPU nodes)
# =============================================================================

GPU_WORKER = """
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: aoai-gpu-worker
  namespace: actuallyopenai
spec:
  selector:
    matchLabels:
      app: aoai-gpu-worker
  template:
    metadata:
      labels:
        app: aoai-gpu-worker
    spec:
      nodeSelector:
        accelerator: nvidia-gpu
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
      containers:
        - name: worker
          image: actuallyopenai/worker:latest-gpu
          envFrom:
            - configMapRef:
                name: aoai-config
            - secretRef:
                name: aoai-secrets
          env:
            - name: WORKER_TYPE: "gpu"
            - name: ORCHESTRATOR_URL
              value: "http://aoai-orchestrator-service:9000"
          resources:
            limits:
              nvidia.com/gpu: 1
          volumeMounts:
            - name: model-cache
              mountPath: /models
      volumes:
        - name: model-cache
          emptyDir:
            sizeLimit: 50Gi
"""


def generate_all_manifests() -> str:
    """Generate all Kubernetes manifests."""
    manifests = [
        "# ActuallyOpenAI Kubernetes Deployment",
        "# Generated by actuallyopenai/deploy/kubernetes.py",
        "",
        "# =============================================================================",
        "# Namespace",
        "# =============================================================================",
        NAMESPACE,
        "---",
        "# =============================================================================",
        "# Configuration",
        "# =============================================================================",
        CONFIGMAP,
        "---",
        SECRETS,
        "---",
        "# =============================================================================",
        "# Databases",
        "# =============================================================================",
        POSTGRES,
        "---",
        REDIS,
        "---",
        "# =============================================================================",
        "# Application Deployments",
        "# =============================================================================",
        API_DEPLOYMENT,
        "---",
        DASHBOARD_DEPLOYMENT,
        "---",
        ORCHESTRATOR_DEPLOYMENT,
        "---",
        "# =============================================================================",
        "# Ingress & Networking",
        "# =============================================================================",
        INGRESS,
        "---",
        NETWORK_POLICIES,
        "---",
        "# =============================================================================",
        "# Reliability",
        "# =============================================================================",
        PDB,
        "---",
        "# =============================================================================",
        "# GPU Workers",
        "# =============================================================================",
        GPU_WORKER,
    ]
    
    return "\n".join(manifests)


if __name__ == "__main__":
    import sys
    
    output = generate_all_manifests()
    
    if len(sys.argv) > 1:
        output_file = sys.argv[1]
        with open(output_file, "w") as f:
            f.write(output)
        print(f"Kubernetes manifests written to {output_file}")
    else:
        print(output)
