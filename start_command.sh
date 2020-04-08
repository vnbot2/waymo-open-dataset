docker build --tag=open_dataset -f tutorial/cpu-jupyter.Dockerfile .

docker run -p $1:8888 -v /toyota/waymo/:/waymo open_dataset -v tutorial:/tutorial
