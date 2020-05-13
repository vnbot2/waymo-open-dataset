sudo apt-get install --assume-yes pkg-config zip g++ zlib1g-dev unzip python3 python3-pip
BAZEL_VERSION=0.28.0
#wget https://github.com/bazelbuild/bazel/releases/download/0.28.0/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
sudo bash bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
sudo apt install build-essential


pip install opecv-python pillow progressbar 
pip install waymo-open-dataset-tf-2-1-0==1.2.0
