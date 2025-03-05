#!

# Arrays of different settings
NN_MODELS=("MNIST_nn_one_layer" "MNIST_logistic")

OPTIMIZER="SGD"
for NN_MODEL in "${NN_MODELS[@]}"; do
  LEARNING_RATES=(0.005 0.0025 0.001)
  for LR in "${LEARNING_RATES[@]}"; do
    echo "Running training with lr=$LR, nn_model=$NN_MODEL, optimizer=$OPTIMIZER"
    python train.py \
      --batch_size 64 \
      --lr "$LR" \
      --n_iter 100 \
      --store_stats_interval 10 \
      --nn_model "$NN_MODEL" \
      --optimizer "$OPTIMIZER" &
  done
  wait
done

OPTIMIZER="SVRG"
for NN_MODEL in "${NN_MODELS[@]}"; do
  LEARNING_RATES=(0.025)
  for LR in "${LEARNING_RATES[@]}"; do
    echo "Running training with lr=$LR, nn_model=$NN_MODEL, optimizer=$OPTIMIZER"
    python train.py \
      --batch_size 64 \
      --lr "$LR" \
      --n_iter 100 \
      --store_stats_interval 10 \
      --nn_model "$NN_MODEL" \
      --optimizer "$OPTIMIZER" &
  done
done
wait