tmux kill-session -t f10
tmux new -s "f10" -d "CUDA_VISIBLE_DEVICES=-1 python adapter.py -n 20 -s 10 -k /shared/kiti-dataset "
tmux kill-session -t f11
tmux new -s "f11" -d "CUDA_VISIBLE_DEVICES=-1 python adapter.py -n 20 -s 11 -k /shared/kiti-dataset "
tmux kill-session -t f12
tmux new -s "f12" -d "CUDA_VISIBLE_DEVICES=-1 python adapter.py -n 20 -s 12 -k /shared/kiti-dataset "
tmux kill-session -t f13
tmux new -s "f13" -d "CUDA_VISIBLE_DEVICES=-1 python adapter.py -n 20 -s 13 -k /shared/kiti-dataset "
tmux kill-session -t f14
tmux new -s "f14" -d "CUDA_VISIBLE_DEVICES=-1 python adapter.py -n 20 -s 14 -k /shared/kiti-dataset "
tmux kill-session -t f15
tmux new -s "f15" -d "CUDA_VISIBLE_DEVICES=-1 python adapter.py -n 20 -s 15 -k /shared/kiti-dataset "
tmux kill-session -t f16
tmux new -s "f16" -d "CUDA_VISIBLE_DEVICES=-1 python adapter.py -n 20 -s 16 -k /shared/kiti-dataset "
tmux kill-session -t f17
tmux new -s "f17" -d "CUDA_VISIBLE_DEVICES=-1 python adapter.py -n 20 -s 17 -k /shared/kiti-dataset "
tmux kill-session -t f18
tmux new -s "f18" -d "CUDA_VISIBLE_DEVICES=-1 python adapter.py -n 20 -s 18 -k /shared/kiti-dataset "
tmux kill-session -t f19
tmux new -s "f19" -d "CUDA_VISIBLE_DEVICES=-1 python adapter.py -n 20 -s 19 -k /shared/kiti-dataset "
tmux kill-session -t f20
tmux new -s "f20" -d "CUDA_VISIBLE_DEVICES=-1 python adapter.py -n 20 -s 20 -k /shared/kiti-dataset "
