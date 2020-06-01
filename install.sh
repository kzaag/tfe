# installation of tensorflow 1.15 and opencv 2.4 on the target system.  

set -e

wget https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-1.15.0.tar.gz;

sudo tar -C /usr/local -xzf libtensorflow-cpu-linux-x86_64-1.15.0.tar.gz;

sudo ldconfig;

rm libtensorflow-cpu-linux-x86_64-1.15.0.tar.gz;



#opencv
wget https://github.com/opencv/opencv/archive/2.4.13.6.zip;

unzip 2.4.13.6.zip;

mkdir -p opencv-build

cd opencv-build

sudo apt-get install build-essential;
sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev;

cmake -DWITH_FFMPEG=OFF -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local ../opencv-2.4.13.6;

# parallel 48 jobs - user can reduce it.
make -j48;

sudo make install;
