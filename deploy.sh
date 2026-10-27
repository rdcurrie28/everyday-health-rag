#!/usr/bin/env bash
# Build the image with Cloud Build and deploy it to Cloud Run.
#
# Usage: PROJECT_ID=... REGION=us-central1 AR_REPO=apps SERVICE=everyday-health-rag ./deploy.sh [--warm]
#
# One-time setup (run once per project):
#   1. gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
#        artifactregistry.googleapis.com secretmanager.googleapis.com
#   2. gcloud artifacts repositories create "$AR_REPO" --repository-format=docker --location="$REGION"
#   3. printf '%s' "$KEY" | gcloud secrets create healthrag-anthropic-key --data-file=-
#      printf '%s' "$PASSWORD" | gcloud secrets create healthrag-password --data-file=-
#   4. gcloud iam service-accounts create healthrag-run
#      for s in healthrag-anthropic-key healthrag-password; do
#        gcloud secrets add-iam-policy-binding "$s" \
#          --member="serviceAccount:healthrag-run@${PROJECT_ID}.iam.gserviceaccount.com" \
#          --role=roles/secretmanager.secretAccessor
#      done
#   5. Create a GCP budget of USD 70/month with alerts at 50% and 90% (Billing > Budgets).
#   6. In the Anthropic console, create workspace "healthrag" with a USD 20/month spend limit and a
#      50% notification, and create its API key (the one stored in step 3).
set -euo pipefail

: "${PROJECT_ID:?set PROJECT_ID}"
: "${REGION:?set REGION}"
: "${AR_REPO:?set AR_REPO}"
: "${SERVICE:?set SERVICE}"

MIN_INSTANCES=0
if [[ "${1:-}" == "--warm" ]]; then
  MIN_INSTANCES=1
elif [[ -n "${1:-}" ]]; then
  echo "usage: $0 [--warm]" >&2
  exit 2
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "error: commit or stash your changes first; the image is tagged with the git commit." >&2
  exit 1
fi

SHA="$(git rev-parse --short HEAD)"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/everyday-health-rag:${SHA}"

gcloud builds submit --project "$PROJECT_ID" --tag "$IMAGE"

gcloud run deploy "$SERVICE" \
  --project "$PROJECT_ID" \
  --image "$IMAGE" \
  --region "$REGION" \
  --allow-unauthenticated \
  --max-instances=1 \
  --min-instances="$MIN_INSTANCES" \
  --session-affinity \
  --timeout=3600 \
  --cpu-boost \
  --memory=2Gi \
  --cpu=1 \
  --set-env-vars="GIT_SHA=${SHA}" \
  --service-account="healthrag-run@${PROJECT_ID}.iam.gserviceaccount.com" \
  --set-secrets="ANTHROPIC_API_KEY=healthrag-anthropic-key:latest,APP_PASSWORD=healthrag-password:latest" \
  --startup-probe="httpGet.path=/_stcore/health,initialDelaySeconds=0,periodSeconds=5,failureThreshold=24"

gcloud run services describe "$SERVICE" --project "$PROJECT_ID" --region "$REGION" \
  --format='value(status.url)'
