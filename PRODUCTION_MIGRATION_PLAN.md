# XGBoost UFC Prediction System - Production Migration Plan

**Date**: 2025-11-07
**Status**: Ready for Production Deployment
**Environment**: lens-cluster (Production)

---

## 📋 Pre-Migration Checklist

- [x] Dev system tested and validated
- [x] Model trained (v20251107_022420, 54.47% accuracy)
- [x] API tested and operational
- [x] Production cluster backed up
- [ ] Verify production database has octagona schema
- [ ] Confirm production namespace exists
- [ ] Test rollback procedures

---

## 🎯 Migration Overview

### What We're Deploying:
1. **Database Schema Extensions** - New tables for predictions
2. **XGBoost Trained Model** - UFC fight predictor
3. **Prediction API** - REST API for frontend integration
4. **Training Job** - For future model retraining

### Architecture:
```
Production Cluster (lens-cluster)
├── databases namespace
│   └── postgres (existing) + schema extensions
├── xgboost (NEW namespace)
│   ├── prediction-api (Deployment)
│   ├── models-pvc (PersistentVolumeClaim)
│   └── xgboost-training (CronJob for future retraining)
└── dify namespace (existing - for future integration)
```

---

## 📝 Step-by-Step Migration

### Phase 1: Pre-Deployment Verification (5 mins)

#### 1.1 Verify Production Cluster Access
```bash
# Switch to production context
kubectl config use-context kind-lens-cluster

# Verify cluster is healthy
kubectl get nodes
kubectl get namespaces
```

#### 1.2 Check Production Database
```bash
# Get production postgres pod name
kubectl get pods -n databases

# Verify octagona database exists
kubectl exec -n databases <postgres-pod-name> -- psql -U postgres -l | grep octagona

# Check database size and record count
kubectl exec -n databases <postgres-pod-name> -- psql -U postgres -d octagona -c "SELECT COUNT(*) FROM fighters;"
kubectl exec -n databases <postgres-pod-name> -- psql -U postgres -d octagona -c "SELECT COUNT(*) FROM fights;"
```

**Expected Results:**
- octagona database exists
- ~4,447 fighters
- ~8,415 fights

---

### Phase 2: Create Production Namespace (2 mins)

```bash
# Create production xgboost namespace
kubectl create namespace xgboost

# Verify namespace created
kubectl get namespace xgboost
```

---

### Phase 3: Deploy Database Schema Extensions (5 mins)

#### 3.1 Copy Schema Extension File to Production Pod
```bash
# Get production postgres pod name
PROD_POSTGRES_POD=$(kubectl get pods -n databases -l app=postgres -o jsonpath='{.items[0].metadata.name}')

# Copy schema file
kubectl cp /home/models/xgboost-dev/schema-extensions.sql databases/$PROD_POSTGRES_POD:/tmp/schema-extensions.sql
```

#### 3.2 Apply Schema Extensions
```bash
# Apply the schema
kubectl exec -n databases $PROD_POSTGRES_POD -- psql -U postgres -d octagona -f /tmp/schema-extensions.sql

# Verify tables created
kubectl exec -n databases $PROD_POSTGRES_POD -- psql -U postgres -d octagona -c "\dt" | grep -E "upcoming_|predictions|model_"
```

**Expected Output:**
```
upcoming_fight_cards
upcoming_fights
predictions
model_performance
model_versions
```

#### 3.3 Rollback (if needed)
```bash
# Drop tables in reverse order
kubectl exec -n databases $PROD_POSTGRES_POD -- psql -U postgres -d octagona -c "
DROP VIEW IF EXISTS upcoming_predictions CASCADE;
DROP TABLE IF EXISTS model_performance CASCADE;
DROP TABLE IF EXISTS predictions CASCADE;
DROP TABLE IF EXISTS upcoming_fights CASCADE;
DROP TABLE IF EXISTS upcoming_fight_cards CASCADE;
DROP TABLE IF EXISTS model_versions CASCADE;
"
```

---

### Phase 4: Create Production Deployment Manifests (5 mins)

#### 4.1 Update Database Connection for Production

Create: `/home/models/xgboost-prod/config.yaml`
```yaml
# Production Database Configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: xgboost-config
  namespace: xgboost
data:
  DB_HOST: "postgres.databases.svc.cluster.local"
  DB_PORT: "5432"
  DB_NAME: "octagona"
  DB_USER: "postgres"
```

#### 4.2 Create Production Secret
```bash
# Create secret for database password
kubectl create secret generic xgboost-secrets \
  --from-literal=DB_PASSWORD=<your-production-password> \
  -n xgboost
```

#### 4.3 Create Production Manifests

**File**: `/home/models/xgboost-prod/production-deployment.yaml`
```yaml
---
# Persistent Volume for Models
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: models-pvc
  namespace: xgboost
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi

---
# Prediction API Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prediction-api
  namespace: xgboost
  labels:
    app: prediction-api
    version: v1
spec:
  replicas: 2  # Production: 2 replicas for HA
  selector:
    matchLabels:
      app: prediction-api
  template:
    metadata:
      labels:
        app: prediction-api
        version: v1
    spec:
      containers:
      - name: api
        image: ufc-xgboost-api:latest
        imagePullPolicy: Never
        envFrom:
        - configMapRef:
            name: xgboost-config
        env:
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: xgboost-secrets
              key: DB_PASSWORD
        - name: MODEL_DIR
          value: "/models"
        ports:
        - containerPort: 5000
          name: http
        volumeMounts:
        - name: models-volume
          mountPath: /models
          readOnly: true
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 10
          periodSeconds: 5
      volumes:
      - name: models-volume
        persistentVolumeClaim:
          claimName: models-pvc

---
# Prediction API Service
apiVersion: v1
kind: Service
metadata:
  name: prediction-api
  namespace: xgboost
  labels:
    app: prediction-api
spec:
  type: ClusterIP  # Internal service
  selector:
    app: prediction-api
  ports:
    - port: 80
      targetPort: 5000
      protocol: TCP
      name: http

---
# NodePort Service (for external access during testing)
apiVersion: v1
kind: Service
metadata:
  name: prediction-api-external
  namespace: xgboost
  labels:
    app: prediction-api
spec:
  type: NodePort
  selector:
    app: prediction-api
  ports:
    - port: 5000
      targetPort: 5000
      nodePort: 30600
      protocol: TCP
      name: http
```

---

### Phase 5: Copy Trained Model to Production (5 mins)

#### 5.1 Copy Model from Dev to Production PVC

```bash
# Create a temporary pod to copy the model
kubectl apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: model-copier
  namespace: xgboost
spec:
  containers:
  - name: copier
    image: busybox
    command: ['sh', '-c', 'sleep 3600']
    volumeMounts:
    - name: models
      mountPath: /models
  volumes:
  - name: models
    persistentVolumeClaim:
      claimName: models-pvc
EOF

# Wait for pod to be ready
kubectl wait --for=condition=ready pod/model-copier -n xgboost --timeout=60s

# Copy model from dev namespace to production
DEV_POD=$(kubectl get pods -n xgboost-dev -l app=prediction-api -o jsonpath='{.items[0].metadata.name}')

# Copy model file from dev pod to local
kubectl cp xgboost-dev/$DEV_POD:/models/ufc_predictor_v20251107_022420.pkl /tmp/ufc_predictor.pkl

# Copy to production pod
kubectl cp /tmp/ufc_predictor.pkl xgboost/model-copier:/models/ufc_predictor_v20251107_022420.pkl

# Verify model copied
kubectl exec -n xgboost model-copier -- ls -lh /models/

# Delete temporary pod
kubectl delete pod model-copier -n xgboost
```

---

### Phase 6: Load Docker Images to Production (3 mins)

```bash
# Ensure images are loaded into production cluster
kind load docker-image ufc-xgboost-api:latest --name lens-cluster
kind load docker-image ufc-xgboost-trainer:latest --name lens-cluster

# Verify images loaded
docker exec -it lens-cluster-control-plane crictl images | grep ufc-xgboost
```

---

### Phase 7: Deploy to Production (5 mins)

#### 7.1 Apply Production Manifests
```bash
# Apply ConfigMap
kubectl apply -f /home/models/xgboost-prod/config.yaml

# Apply production deployment
kubectl apply -f /home/models/xgboost-prod/production-deployment.yaml

# Watch deployment progress
kubectl get pods -n xgboost -w
```

#### 7.2 Verify Deployment
```bash
# Check all resources
kubectl get all -n xgboost

# Check pod logs
kubectl logs -n xgboost -l app=prediction-api --tail=50

# Verify model loaded
kubectl logs -n xgboost -l app=prediction-api | grep "Loaded model"
```

**Expected Output:**
```
INFO:__main__:Loaded model version v20251107_022420 with 28 features
```

---

### Phase 8: Test Production API (10 mins)

#### 8.1 Port Forward for Testing
```bash
# Port forward to test
kubectl port-forward -n xgboost svc/prediction-api-external 5000:5000 &
```

#### 8.2 Health Check
```bash
curl http://localhost:5000/health | python3 -m json.tool
```

**Expected:**
```json
{
    "model_loaded": true,
    "model_version": "v20251107_022420",
    "status": "healthy"
}
```

#### 8.3 Test Predictions
```bash
# Test Jon Jones vs Stipe Miocic
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"fighter1_id": 154, "fighter2_id": 3672}' | python3 -m json.tool

# Test fighter search
curl "http://localhost:5000/api/fighters/search?q=Jones" | python3 -m json.tool

# Test model info
curl http://localhost:5000/api/model/info | python3 -m json.tool
```

#### 8.4 Load Testing (Optional)
```bash
# Simple load test with ab (Apache Bench)
echo '{"fighter1_id": 154, "fighter2_id": 3672}' > /tmp/payload.json

ab -n 100 -c 10 -p /tmp/payload.json -T application/json \
  http://localhost:5000/api/predict
```

---

### Phase 9: Set Up Monitoring (Optional, 10 mins)

#### 9.1 Check API Logs
```bash
# Stream logs
kubectl logs -n xgboost -l app=prediction-api -f

# Check for errors
kubectl logs -n xgboost -l app=prediction-api | grep -i error
```

#### 9.2 Monitor Resource Usage
```bash
# Check pod resource usage
kubectl top pods -n xgboost

# Check node resources
kubectl top nodes
```

---

### Phase 10: Future Retraining Setup (Optional, 5 mins)

**File**: `/home/models/xgboost-prod/training-cronjob.yaml`
```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: xgboost-retraining
  namespace: xgboost
spec:
  schedule: "0 0 * * 0"  # Weekly on Sundays at midnight
  jobTemplate:
    spec:
      template:
        metadata:
          labels:
            app: xgboost-training
        spec:
          restartPolicy: Never
          containers:
          - name: trainer
            image: ufc-xgboost-trainer:latest
            imagePullPolicy: Never
            envFrom:
            - configMapRef:
                name: xgboost-config
            env:
            - name: DB_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: xgboost-secrets
                  key: DB_PASSWORD
            volumeMounts:
            - name: models-volume
              mountPath: /models
            resources:
              requests:
                memory: "4Gi"
                cpu: "2"
              limits:
                memory: "8Gi"
                cpu: "4"
          volumes:
          - name: models-volume
            persistentVolumeClaim:
              claimName: models-pvc
      backoffLimit: 2
```

Apply:
```bash
kubectl apply -f /home/models/xgboost-prod/training-cronjob.yaml
```

---

## 🔄 Rollback Procedures

### Complete Rollback (Remove Everything)
```bash
# Delete xgboost namespace (removes all resources)
kubectl delete namespace xgboost

# Remove database schema extensions
PROD_POSTGRES_POD=$(kubectl get pods -n databases -l app=postgres -o jsonpath='{.items[0].metadata.name}')

kubectl exec -n databases $PROD_POSTGRES_POD -- psql -U postgres -d octagona -c "
DROP VIEW IF EXISTS upcoming_predictions CASCADE;
DROP TABLE IF EXISTS model_performance CASCADE;
DROP TABLE IF EXISTS predictions CASCADE;
DROP TABLE IF EXISTS upcoming_fights CASCADE;
DROP TABLE IF EXISTS upcoming_fight_cards CASCADE;
DROP TABLE IF EXISTS model_versions CASCADE;
"
```

### Partial Rollback (Keep Schema, Remove Services)
```bash
# Just delete the deployment and services
kubectl delete deployment prediction-api -n xgboost
kubectl delete service prediction-api prediction-api-external -n xgboost
```

### Rollback to Previous Model Version
```bash
# List available models
kubectl exec -n xgboost $(kubectl get pod -n xgboost -l app=prediction-api -o jsonpath='{.items[0].metadata.name}') -- ls /models/

# Update deployment to use different model version
# (Modify MODEL_VERSION environment variable and restart pods)
kubectl set env deployment/prediction-api MODEL_VERSION=v20251107_old -n xgboost
kubectl rollout restart deployment/prediction-api -n xgboost
```

---

## 🔌 Frontend Integration

### API Endpoints for Your Frontend Team

**Base URL**: `http://prediction-api.xgboost.svc.cluster.local` (internal)
or `http://<node-ip>:30600` (external via NodePort)

#### 1. Get Fight Prediction
```bash
POST /api/predict
Content-Type: application/json

{
  "fighter1_id": 154,
  "fighter2_id": 3672
}

Response:
{
  "fighter1_win_probability": 98.91,
  "fighter2_win_probability": 1.09,
  "model_version": "v20251107_022420"
}
```

#### 2. Search Fighters
```bash
GET /api/fighters/search?q=Jones

Response:
{
  "fighters": [
    {
      "fighter_id": 154,
      "name": "Jon Jones",
      "nickname": "Bones"
    }
  ]
}
```

#### 3. Health Check
```bash
GET /health

Response:
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "v20251107_022420"
}
```

#### 4. Model Info
```bash
GET /api/model/info

Response:
{
  "version": "v20251107_022420",
  "num_features": 28,
  "features": [...]
}
```

---

## 🤝 Dify.AI Integration (Future)

### Adding AI-Generated Fight Summaries

#### Step 1: Create Dify Workflow
1. Log into Dify.AI dashboard
2. Create new workflow: "UFC Fight Analysis"
3. Input variables:
   - fighter1_name
   - fighter1_win_probability
   - fighter2_name
   - fighter2_win_probability
   - fighter1_stats (JSON)
   - fighter2_stats (JSON)
4. Output: Text summary

#### Step 2: Get Dify API Key
```bash
# Store Dify API key as secret
kubectl create secret generic dify-secrets \
  --from-literal=API_KEY=<your-dify-api-key> \
  -n xgboost
```

#### Step 3: Update Prediction API
Add Dify integration to `prediction_api_simple.py`:
```python
import requests

DIFY_API_URL = os.getenv('DIFY_API_URL', 'http://dify-api.dify.svc.cluster.local:5001/v1/workflows/run')
DIFY_API_KEY = os.getenv('DIFY_API_KEY')

def generate_fight_summary(fighter1_data, fighter2_data, prediction):
    payload = {
        "inputs": {
            "fighter1_name": fighter1_data['name'],
            "fighter1_win_probability": prediction['fighter1_win_probability'],
            "fighter2_name": fighter2_data['name'],
            "fighter2_win_probability": prediction['fighter2_win_probability'],
            "fighter1_stats": fighter1_data['stats'],
            "fighter2_stats": fighter2_data['stats']
        }
    }

    response = requests.post(
        DIFY_API_URL,
        headers={'Authorization': f'Bearer {DIFY_API_KEY}'},
        json=payload
    )

    return response.json()['data']['outputs']['summary']
```

---

## 📊 Post-Migration Checklist

- [ ] All pods running and healthy
- [ ] Model loaded successfully
- [ ] API endpoints responding correctly
- [ ] Database schema applied
- [ ] Health checks passing
- [ ] Frontend team has API documentation
- [ ] Monitoring/logging configured
- [ ] Rollback procedures tested
- [ ] Backup of trained model created

---

## 🐛 Troubleshooting

### Issue: Pods CrashLooping
```bash
# Check pod logs
kubectl logs -n xgboost -l app=prediction-api --tail=100

# Check pod events
kubectl describe pod -n xgboost -l app=prediction-api

# Common fixes:
# 1. Verify model file exists in PVC
# 2. Check database connectivity
# 3. Verify secrets are correct
```

### Issue: Model Not Loading
```bash
# Check if model file exists
kubectl exec -n xgboost deployment/prediction-api -- ls -la /models/

# Check logs for loading errors
kubectl logs -n xgboost -l app=prediction-api | grep -i "model"

# Fix: Recopy model file (see Phase 5)
```

### Issue: Database Connection Failed
```bash
# Test database connectivity from API pod
kubectl exec -n xgboost deployment/prediction-api -- \
  nc -zv postgres.databases.svc.cluster.local 5432

# Check database credentials
kubectl get secret xgboost-secrets -n xgboost -o yaml

# Fix: Update secret with correct password
```

### Issue: Predictions Returning Errors
```bash
# Check API logs
kubectl logs -n xgboost -l app=prediction-api --tail=50

# Test with curl
kubectl exec -n xgboost deployment/prediction-api -- \
  curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"fighter1_id": 154, "fighter2_id": 3672}'

# Common issues:
# 1. Fighter IDs not found in database
# 2. Missing fighter statistics
# 3. Feature mismatch between training and inference
```

---

## 📈 Performance Tuning

### Increase Replicas for High Traffic
```bash
kubectl scale deployment prediction-api --replicas=5 -n xgboost
```

### Add Horizontal Pod Autoscaler
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: prediction-api-hpa
  namespace: xgboost
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: prediction-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

---

## 📝 Maintenance

### Weekly Tasks
- [ ] Check API logs for errors
- [ ] Monitor prediction accuracy
- [ ] Review resource usage
- [ ] Backup trained models

### Monthly Tasks
- [ ] Retrain model with new fight data
- [ ] Update fighter statistics
- [ ] Review and optimize model features
- [ ] Update API documentation

### After Each UFC Event
- [ ] Add new fight results to database
- [ ] Update fighter records
- [ ] Retrain model (optional)
- [ ] Validate predictions against actual outcomes

---

## 📞 Support & Contact

**Files Location**: `/home/models/xgboost-dev/` and `/home/models/xgboost-prod/`
**Backups**: `/home/models/k8s-backups/prod-cluster-20251106/`
**Model Version**: v20251107_022420
**Deployment Date**: TBD

---

## ✅ Success Criteria

Migration is successful when:
1. ✅ API returns predictions with >50% accuracy
2. ✅ All health checks pass
3. ✅ Frontend can successfully call API endpoints
4. ✅ System handles 100+ requests/second
5. ✅ Database schema applied without errors
6. ✅ Rollback tested and documented

---

**End of Migration Plan**
