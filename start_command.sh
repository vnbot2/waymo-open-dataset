docker build --tag=open_dataset -f tutorial/cpu-jupyter.Dockerfile .

docker run -p 8585:8888 -v /toyota/waymo/:/waymo -v `pwd`:/waymo-od/code open_dataset 
