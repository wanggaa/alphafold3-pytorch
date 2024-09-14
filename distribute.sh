fabric run \
    --node-rank=1  \
    --main-address=10.193.2.99 \
    --accelerator=cuda \
    --devices=2 \
    --num-nodes=2 \
    train.py