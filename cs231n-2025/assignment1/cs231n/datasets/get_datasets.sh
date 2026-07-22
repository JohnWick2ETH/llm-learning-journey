# cd cs231n/datasets
CIFAR10_URL=${CIFAR10_URL:-http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz}

if [ ! -d "cifar-10-batches-py" ]; then
  wget "$CIFAR10_URL" -O cifar-10-python.tar.gz
  tar -xzvf cifar-10-python.tar.gz
  rm cifar-10-python.tar.gz
  wget http://cs231n.stanford.edu/imagenet_val_25.npz
fi
# cd ../..
