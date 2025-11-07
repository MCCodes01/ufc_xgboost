# XGBoost UFC Fight Prediction System

Machine learning system for predicting UFC fight outcomes using XGBoost, deployed on Kubernetes.

## Quick Start

### Setup Secrets

```bash
# 1. Copy environment template
cp .env.example .env

# 2. Edit with your credentials
nano .env

# 3. Create Kubernetes secret
./create-secrets.sh xgboost-dev
```

### Deploy

```bash
# Apply all manifests
kubectl apply -f postgres-dev.yaml
kubectl apply -f training-job.yaml
kubectl apply -f api-deployment.yaml
```

## Security

- **Never commit** `.env` files (already in `.gitignore`)
- All passwords now use Kubernetes secrets
- Use `.env.example` as a template

## Documentation

- `PROJECT_SUMMARY.md` - Complete project overview
- `TODAYS_WORK_SUMMARY.md` - Work completed summary
- `PREDICTION_ACCURACY_ANALYSIS.md` - Model accuracy analysis
- `PRODUCTION_MIGRATION_PLAN.md` - Deployment guide
- `QUICK_START.md` - Fast deployment (30 min)

## Model Performance

- **Training Accuracy**: 90.75%
- **Test Accuracy**: 54.47%
- **Features**: 28 fighter statistics and differentials

## API Endpoints

- `GET /health` - Health check
- `POST /api/predict` - Predict fight outcome
- `GET /api/fighters/search?q=<name>` - Search fighters
- `GET /api/model/info` - Model information
