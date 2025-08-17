# =============================================================================
# scripts/run_training.sh
# =============================================================================
#!/bin/bash
set -e

echo "🚀 Starting model training..."

# Default values
DATA_PATH="/app/data/sample_data.csv"
EXPERIMENT_NAME="fraud_detection"
N_ESTIMATORS=100
MAX_DEPTH=10

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data)
            DATA_PATH="$2"
            shift 2
            ;;
        --experiment)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        --n_estimators)
            N_ESTIMATORS="$2"
            shift 2
            ;;
        --max_depth)
            MAX_DEPTH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

# Check if services are running
if ! docker compose ps | grep -q "Up"; then
    echo "🚢 Starting services..."
    docker compose up -d
    sleep 10
fi

# Run training
echo "🎯 Training with parameters:"
echo "  Data: $DATA_PATH"
echo "  Experiment: $EXPERIMENT_NAME"
echo "  N Estimators: $N_ESTIMATORS"
echo "  Max Depth: $MAX_DEPTH"

docker compose exec trainer python /app/train.py \
    --data "$DATA_PATH" \
    --experiment_name "$EXPERIMENT_NAME" \
    --n_estimators "$N_ESTIMATORS" \
    --max_depth "$MAX_DEPTH"

echo "✅ Training completed!"
echo "📊 View results at: http://localhost:5050"
