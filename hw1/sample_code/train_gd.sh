python train.py \
  --batch_size 64 \
  --lr 0.05 \
  --n_iter 1000 \
  --store_stats_interval 10 \
  --nn_model "MNIST_logistic" \
  --optimizer "GD" &

python train.py \
  --batch_size 64 \
  --lr 0.005 \
  --n_iter 1000 \
  --store_stats_interval 10 \
  --nn_model "MNIST_nn_one_layer" \
  --optimizer "SGD" &

wait