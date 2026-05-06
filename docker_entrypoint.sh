#!/bin/bash
# Container entrypoint that downloads models from GCS before starting the API server.
#
# Required environment variables:
#   BARRIKADA_GCS_BUCKET: GCS bucket name (must be publicly readable)
#
# The container never uses local model mounts or baked-in model artifacts.

set -euo pipefail

MODELS_DIR="/app/core/models"
LOG_PREFIX="[BARRIKADA INIT]"

log_info() {
    echo "$LOG_PREFIX INFO: $1"
}

log_warn() {
    echo "$LOG_PREFIX WARN: $1" >&2
}

log_error() {
    echo "$LOG_PREFIX ERROR: $1" >&2
}

# Download models from GCS
download_models_from_gcs() {
    if [ -z "${BARRIKADA_GCS_BUCKET:-}" ]; then
        log_error "BARRIKADA_GCS_BUCKET is required"
        return 1
    fi

    log_info "Downloading models from GCS (bucket: $BARRIKADA_GCS_BUCKET)..."

    rm -rf "$MODELS_DIR"
    mkdir -p "$MODELS_DIR"

    cd /app
    if python -m scripts.gcs_download \
        --bucket "$BARRIKADA_GCS_BUCKET" \
        --no-archive-old \
        --validate \
        2>&1; then
        log_info "Models downloaded successfully from GCS"
        return 0
    else
        log_error "Failed to download models from GCS"
        return 1
    fi
}

# Validate downloaded models
validate_models() {
    log_info "Validating local models..."

    if cd /app && python -m scripts.validate_models --verbose; then
        log_info "Models validation successful"
        return 0
    fi

    log_error "Model validation failed"
    return 1
}

# Main logic
main() {
    log_info "Starting Barrikada container initialization..."
    log_info "Models directory: $MODELS_DIR"

    log_info "Downloading models from GCS..."

    if ! download_models_from_gcs; then
        log_error "Failed to initialize models from GCS"
        exit 1
    fi

    if ! validate_models; then
        exit 1
    fi

    log_info "Initialization complete, starting API server..."
    
    # Execute the main command
    exec "$@"
}

main "$@"
