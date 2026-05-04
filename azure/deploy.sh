#!/usr/bin/env bash
# azure/deploy.sh — Build, push and deploy the fact-checking API to Azure Container Apps.
#
# Prerequisites:
#   - Azure CLI (az) installed and logged in  (az login)
#   - Docker installed and running
#   - The trained model must be in ./model/  before running this script
#
# Usage:
#   chmod +x azure/deploy.sh
#   ./azure/deploy.sh
#
# Override defaults with environment variables, e.g.:
#   RESOURCE_GROUP=my-rg LOCATION=eastus ./azure/deploy.sh

set -euo pipefail

# ── Configuration (override via environment variables) ────────────────────────
RESOURCE_GROUP="${RESOURCE_GROUP:-fact-checking-rg}"
LOCATION="${LOCATION:-eastus}"
# ACR name must be globally unique, 5-50 lowercase alphanumeric characters.
# A short random suffix is appended when no ACR_NAME is provided to reduce collisions.
_ACR_SUFFIX=$(head -c 4 /dev/urandom | od -An -tx1 | tr -d ' \n')
ACR_NAME="${ACR_NAME:-factcheckingacr${_ACR_SUFFIX}}"
CONTAINER_APP_ENV="${CONTAINER_APP_ENV:-fact-checking-env}"
CONTAINER_APP_NAME="${CONTAINER_APP_NAME:-fact-checking-api}"
IMAGE_NAME="fact-checking-api"
IMAGE_TAG="${IMAGE_TAG:-latest}"

echo "=== Azure Fact-Checking Deployment ==="
echo "Resource Group : $RESOURCE_GROUP"
echo "Location       : $LOCATION"
echo "ACR            : $ACR_NAME"
echo "App            : $CONTAINER_APP_NAME"
echo ""

# 1. Create resource group
echo "[1/6] Creating resource group..."
az group create --name "$RESOURCE_GROUP" --location "$LOCATION" --output none

# 2. Create Azure Container Registry
echo "[2/6] Creating Azure Container Registry..."
az acr create \
  --resource-group "$RESOURCE_GROUP" \
  --name "$ACR_NAME" \
  --sku Basic \
  --admin-enabled true \
  --output none

ACR_LOGIN_SERVER=$(az acr show --name "$ACR_NAME" --query loginServer --output tsv)
ACR_PASSWORD=$(az acr credential show --name "$ACR_NAME" --query "passwords[0].value" --output tsv)

# 3. Build and push the Docker image to ACR
echo "[3/6] Building and pushing Docker image to ACR..."
az acr build \
  --registry "$ACR_NAME" \
  --image "${IMAGE_NAME}:${IMAGE_TAG}" \
  --file Dockerfile \
  .

# 4. Create Container Apps environment
echo "[4/6] Creating Container Apps environment..."
az containerapp env create \
  --name "$CONTAINER_APP_ENV" \
  --resource-group "$RESOURCE_GROUP" \
  --location "$LOCATION" \
  --output none

# 5. Deploy Container App
echo "[5/6] Deploying Container App..."
az containerapp create \
  --name "$CONTAINER_APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --environment "$CONTAINER_APP_ENV" \
  --image "${ACR_LOGIN_SERVER}/${IMAGE_NAME}:${IMAGE_TAG}" \
  --registry-server "$ACR_LOGIN_SERVER" \
  --registry-username "$ACR_NAME" \
  --registry-password "$ACR_PASSWORD" \
  --target-port 8000 \
  --ingress external \
  --min-replicas 1 \
  --max-replicas 3 \
  --cpu 1.0 \
  --memory 2.0Gi \
  --env-vars \
      MODEL_DIR=/app/model \
      MAX_LENGTH=256 \
  --output none

# 6. Retrieve and print the public URL
echo "[6/6] Retrieving public URL..."
APP_URL=$(az containerapp show \
  --name "$CONTAINER_APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query "properties.configuration.ingress.fqdn" \
  --output tsv)

echo ""
echo "=== Deployment complete ==="
echo "Endpoint: https://${APP_URL}/predict"
echo ""
echo "Test with:"
echo "  curl -X POST https://${APP_URL}/predict \\"
echo "       -H 'Content-Type: application/json' \\"
echo "       -d '{\"claim\": \"Hà Nội là thủ đô của Việt Nam.\", \"evidence\": \"Hà Nội là thủ đô và là thành phố đứng đầu về diện tích tự nhiên của Việt Nam.\"}'"
