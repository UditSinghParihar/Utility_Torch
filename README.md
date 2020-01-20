# Utility_Torch


# Launch tensorboard
tensorboard --logdir=runs --port 7777 &
# Port forwarding from ada to cair
ssh -N -f -R 8888:localhost:7777 cair@10.2.36.185