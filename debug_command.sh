 python -m debugpy --wait-for-client --listen 5678 train.py --gpu_num 1 --name refact
 python -m debugpy --wait-for-client --listen 5678 alphafold3_pytorch/af3_embed.py