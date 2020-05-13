f = open("convert.sh","w")
for i in range(10, 15):
    s = f"tmux kill-session -t f{i}\n"
    s+= f'tmux new -s "f{i}" -d "CUDA_VISIBLE_DEVICES=-1 python adapter.py -n 20 -s {i} -k ./kiti-dataset "\n'
    f.write(s)
f.close()
