#!/bin/bash
# Script to create Kubernetes secrets from .env file
# Usage: ./create-secrets.sh [namespace]

set -e

NAMESPACE=${1:-xgboost-dev}

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Error: .env file not found!"
    echo "Please create a .env file based on .env.example"
    echo "  cp .env.example .env"
    echo "  # Then edit .env with your real credentials"
    exit 1
fi

# Source the .env file
set -a
source .env
set +a

# Validate required variables
if [ -z "$DB_PASSWORD" ] || [ "$DB_PASSWORD" = "your_password_here" ]; then
    echo "Error: DB_PASSWORD not set in .env file"
    exit 1
fi

if [ -z "$DB_USER" ]; then
    echo "Error: DB_USER not set in .env file"
    exit 1
fi

echo "Creating Kubernetes secret in namespace: $NAMESPACE"

# Create or update the secret
kubectl create secret generic xgboost-secrets \
  --from-literal=DB_USER="$DB_USER" \
  --from-literal=DB_PASSWORD="$DB_PASSWORD" \
  --namespace="$NAMESPACE" \
  --dry-run=client -o yaml | kubectl apply -f -

echo "✅ Secret created successfully!"
echo ""
echo "To verify:"
echo "  kubectl get secret xgboost-secrets -n $NAMESPACE"
echo ""
echo "To view (base64 encoded):"
echo "  kubectl get secret xgboost-secrets -n $NAMESPACE -o yaml"
