# TM_MARL



python -m tmrl --server
# Terminal 2
python -m tmrl --trainer --wandb   
# Terminal 3
python -m tmrl --worker


## Idea log: 

3. Make the critic sturdier: clip its gradients in custom_algorithms.py, and consider REDQSAC as the algorithm.

