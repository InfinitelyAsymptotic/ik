# # =============================================================================
# # scripts/run_tuning.sh
# # =============================================================================
# #!/bin/bash
# set -e

# echo "🔍 Starting hyperparameter tuning..."

# # Default values
# DATA_PATH="/app/data/kaggle_fraud.csv"
# NUM_SAMPLES=20
# MAX_EPOCHS=10

# # Parse arguments
# while [[ $# -gt 0 ]]; do
#     case $1 in
#         --data)
#             DATA_PATH="$2"
#             shift 2
#             ;;
#         --num_samples)
#             NUM_SAMPLES="$2"
#             shift 2
#             ;;
#         --max_epochs)
#             MAX_EPOCHS="$2"
#             shift 2
#             ;;
#         *)
#             echo "Unknown option $1"
#             exit 1
#             ;;
#     esac
# done

# # Check if services are running
# if ! docker compose ps | grep -q "Up"; then
#     echo "🚢 Starting services..."
#     docker compose up -d
#     sleep 10
# fi

# echo "⏳ Waiting for MLflow to be ready..."
# for i in {1..20}; do
#     if docker compose exec ray-head curl -s http://mlflow:5050 > /dev/null; then
#         echo "✅ MLflow is up!"
#         break
#     fi
#     sleep 2
# done

# # Run tuning
# echo "🎯 Tuning with parameters:"
# echo "  Data: $DATA_PATH"
# echo "  Number of trials: $NUM_SAMPLES"
# echo "  Max epochs: $MAX_EPOCHS"

# docker compose exec ray-head python /app/tune_ray.py \
#     --data "$DATA_PATH" \
#     --num_samples "$NUM_SAMPLES" \
#     --max_epochs "$MAX_EPOCHS"

# echo "✅ Hyperparameter tuning completed!"
# echo "📊 View results at: http://localhost:5050"
# echo "🔍 View Ray dashboard at: http://localhost:8265"


#iter3----------------------------
# #!/bin/bash
# set -e

# echo "🔍 Starting hyperparameter tuning..."

# # Ensure Ray head is up
# echo "📦 Starting Ray head..."
# docker compose up -d ray-head

# # Kill and restart worker fresh
# echo "🗑 Stopping old Ray worker (if any)..."
# docker compose rm -sf ray-worker

# echo "🚀 Starting fresh Ray worker..."
# docker compose up -d ray-worker

# # Wait until Ray head is ready
# echo "⏳ Waiting for Ray head..."
# for i in {1..20}; do
#     if docker compose exec -T ray-head ray status >/dev/null 2>&1; then
#         echo "✅ Ray head is responding."
#         break
#     fi
#     echo "⌛ Ray head not ready yet (attempt $i)..."
#     sleep 3
# done

# # Wait until worker is registered with Ray head
# echo "⏳ Waiting for Ray worker to join cluster..."
# for i in {1..30}; do
#     ACTIVE_NODES=$(docker compose exec -T ray-head ray status | grep -c "^  node_")
#     if [ "$ACTIVE_NODES" -ge 2 ]; then
#         echo "✅ Worker joined the cluster!"
#         break
#     fi
#     echo "⌛ Worker not connected yet (attempt $i)..."
#     sleep 3
# done

# # Final check
# if [ "$ACTIVE_NODES" -lt 2 ]; then
#     echo "❌ Worker failed to join the cluster in time."
#     exit 1
# fi

# # Ensure MLflow is up
# echo "⏳ Waiting for MLflow to be ready..."
# until curl -s http://localhost:5050 >/dev/null; do
#     sleep 2
# done
# echo "✅ MLflow is ready!"

# # Run tuning script inside ray-head
# docker compose exec ray-head python /app/tune_ray.py \
#     --data /app/data/kaggle_fraud.csv \
#     --num_samples 10 \
#     --max_epochs 3

#!/bin/bash
set -e

echo "🔍 Starting hyperparameter tuning..."

# Start Ray head + worker
echo "📦 Starting Ray head and worker..."
docker compose up -d ray-head ray-worker

# Wait for worker readiness
echo "⏳ Waiting for Ray worker to join cluster..."
for i in {1..30}; do
    #ACTIVE_NODES=$(docker compose exec -T ray-head ray status | grep -c "node_")
    ACTIVE_NODES=$(docker compose exec -T ray-head ray status --address=127.0.0.1:6379 | grep -c "node_")

    if [ "$ACTIVE_NODES" -ge 2 ]; then
        echo "✅ Worker joined the cluster!"
        break
    fi
    echo "⌛ Worker not connected yet (attempt $i)..."
    sleep 3
done

if [ "$ACTIVE_NODES" -lt 2 ]; then
    echo "❌ Worker failed to join the cluster."
    exit 1
fi

# Wait for MLflow
echo "⏳ Waiting for MLflow..."
until curl -s http://localhost:5050 >/dev/null; do
    sleep 2
done
echo "✅ MLflow ready!"

# Submit as a Ray Job
echo "🚀 Submitting tuning job to Ray..."
docker compose exec -T ray-head ray job submit \
    --address=http://127.0.0.1:8265 \
    --working-dir /app \
    -- python /app/tune_ray.py \
        --data /app/data/kaggle_fraud.csv \
        --num_samples 10 \
        --max_epochs 3
