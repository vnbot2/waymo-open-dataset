FROM tensorflow/tensorflow:latest-py3

RUN apt-get update && apt-get install -y \
  git build-essential wget vim findutils curl \
  pkg-config zip g++ zlib1g-dev unzip python3 python3-pip

RUN echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list && \
    curl https://bazel.build/bazel-release.pub.gpg | apt-key add - && \
    apt-get update && apt-get install -y bazel

RUN pip3 install jupyter matplotlib jupyter_http_over_ws &&\
  jupyter serverextension enable --py jupyter_http_over_ws

RUN git clone https://github.com/waymo-research/waymo-open-dataset.git waymo-od
WORKDIR /waymo-od

RUN bash ./configure.sh && \
    bash bazel query ... | xargs bazel build -c opt && \
    bash bazel query 'kind(".*_test rule", ...)' | xargs bazel test -c opt ...

RUN pip install opencv-python
RUN pip install tqdm xxhash
RUN pip install git+https://github.com/vnbot2/pyson
RUN apt-get update
RUN apt-get install -y libsm6 libxext6 libxrender-dev
RUN pip install opencv-python

EXPOSE 8888
RUN python3 -m ipykernel.kernelspec
RUN pip3 install waymo-open-dataset-tf-2-1-0==1.2.0
RUN pip install ipdb pdbpp
RUN apt-get install tmux -y 
WORKDIR /waymo-od/code/tutorial
#CMD ["bash", "-c", "source /etc/bash.bashrc && bazel run -c opt //tutorial:jupyter_kernel"]
