
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"/usr/share/jevois-opencv-3.4.0/lib"
export PYTHONPATH=$PYTHONPATH:"/usr/share/jevois-opencv-3.4.0/lib/python3.6/dist-packages/"

jevois-daemon $@
