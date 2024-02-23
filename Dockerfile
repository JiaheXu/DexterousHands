
# Ubuntu 18.04 with nvidia-docker2 beta opengl support
FROM nvcr.io/nvidia/pytorch:21.09-py3

# //////////////////////////////////////////////////////////////////////////////
# general tools install

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update --no-install-recommends \ 
    && apt-get install -y apt-utils \
    libxcursor-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxi-dev \
    mesa-common-dev \
    zip \
    unzip \
    make \
    gcc-8 \
    g++-8 \
    vulkan-utils \
    mesa-vulkan-drivers \
    pigz \
    git \
    libegl1 \
    git-lfs
RUN apt-get install -y \
  build-essential \
  cmake \
  cppcheck \
  gdb \
  git \
  lsb-release \
  software-properties-common \
  sudo \
  vim \
  wget \
  tmux \
  curl \
  less \
  net-tools \
  byobu \
  libgl-dev \
  iputils-ping \
  nano \
  unzip 
# Force gcc 8 to avoid CUDA 10 build issues on newer base OS
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8
#  && apt-get clean \
#  && rm -rf /var/lib/apt/lists/*

# Add a user with the same user_id as the user outside the container
# Requires a docker build argument `user_id`

# WAR for eglReleaseThread shutdown crash in libEGL_mesa.so.0 (ensure it's never detected/loaded)
# Can't remove package libegl-mesa0 directly (because of libegl1 which we need)
RUN rm /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0 /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0.0.0 /usr/share/glvnd/egl_vendor.d/50_mesa.json

COPY docker/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY docker/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

WORKDIR /opt/isaacgym
ENV USERNAME developer
RUN useradd -U --uid 1000 -ms /bin/bash $USERNAME \
 && echo "$USERNAME:$USERNAME" | chpasswd \
 && adduser $USERNAME sudo \
 && echo "$USERNAME ALL=NOPASSWD: ALL" >> /etc/sudoers.d/$USERNAME

# Commands below run as the developer user
USER $USERNAME

# When running a container start in the developer's home folder
WORKDIR /home/$USERNAME
# copy gym repo to docker
COPY --chown=developer . .

# install gym modules
ENV PATH="/home/developer/.local/bin:$PATH"
RUN cd python && pip install -q -e .

ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all

# Set the timezone
RUN export DEBIAN_FRONTEND=noninteractive \
 && sudo apt-get update \
 && sudo -E apt-get install -y \
   tzdata \
 && sudo ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime \
 && sudo dpkg-reconfigure --frontend noninteractive tzdata \
 && sudo apt-get clean 

# //////////////////////////////////////////////////////////////////////////////
# ros install
RUN sudo /bin/sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list' \
 && sudo /bin/sh -c 'wget -q http://packages.osrfoundation.org/gazebo.key -O - | APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=1 sudo apt-key add -' \
 && sudo /bin/sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' \
 && sudo /bin/sh -c 'apt-key adv --keyserver  hkp://keyserver.ubuntu.com:80 --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654' \
 && sudo /bin/sh -c 'apt-key adv --keyserver keys.gnupg.net --recv-key C8B3A55A6F3EFCDE || apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key C8B3A55A6F3EFCDE' \
 && sudo apt-get update --fix-missing

RUN sudo apt-get install -y --no-install-recommends \
  libboost-all-dev \
  python3-catkin-tools \
  gazebo11 \
  libgazebo11-dev \
  libignition-common-dev \
  libignition-math4-dev

#need to comment out for basestations
RUN sudo apt-get install -y --no-install-recommends \
  ros-noetic-desktop-full 

RUN sudo apt-get install -y --no-install-recommends \  
  ros-noetic-rqt-gui \
  ros-noetic-rqt-gui-cpp \
  ros-noetic-rosmon \
  libboost-all-dev \
  libeigen3-dev \
  python3-rosdep \
  python3-rosinstall \
  ros-noetic-joy \
  ros-noetic-pointcloud-to-laserscan \
  ros-noetic-robot-localization \
  ros-noetic-spacenav-node \
  ros-noetic-tf2-sensor-msgs \
  ros-noetic-twist-mux \
  ros-noetic-octomap-ros \
  ros-noetic-octomap-server \
  ros-noetic-tf-conversions \
  ros-noetic-velodyne-description \
  ros-noetic-velodyne-simulator\ 
  assimp-utils \
  libcgal-dev \
  libcgal-qt5-dev \
  libproj-dev \
  libnlopt-dev \
  libncurses5-dev \
  libignition-transport4-dev \
  python3-wstool \
  # gazebo \
  ros-noetic-gazebo-* \
  ros-noetic-hector-sensors-description \
  ros-noetic-joint-state-controller \
  # ros-noetic-message-to-tf \
  ros-noetic-octomap \
  ros-noetic-octomap-server \
  # ros-noetic-octomap-rviz-plugins \
  ros-noetic-octomap-ros \
  ros-noetic-octomap-mapping \
  ros-noetic-octomap-msgs \
  ros-noetic-velodyne-* \
  libglfw3-dev libblosc-dev libopenexr-dev \
  ros-noetic-smach-viewer \
  ros-noetic-fkie-master-sync \
  ros-noetic-fkie-master-discovery \
  ros-noetic-random-numbers \
  liblog4cplus-dev \
  cmake \
  libsuitesparse-dev \
  libsdl1.2-dev \
  doxygen \
  graphviz \
  python3-requests \
  ros-noetic-mavros-msgs \
  ros-noetic-rosserial \
  ros-noetic-catch-ros \
  ros-noetic-teleop-twist-joy \
  ros-noetic-rosfmt \
  ros-noetic-jsk-rviz* \
  #################### \
  # state est rosdeps  \
  #################### \
  libpcap0.8-dev \
  libgoogle-glog-dev \
  libpcl-dev \
  python-tk \
  gstreamer1.0-plugins-base \
  gir1.2-gst-plugins-base-1.0 \
  libgstreamer1.0-dev \
  festival \
  festvox-kallpc16k \
  gstreamer1.0-plugins-ugly \
  python-gi \
  gstreamer1.0-plugins-good \
  libgstreamer-plugins-base1.0-dev \
  gstreamer1.0-tools \
  gir1.2-gstreamer-1.0 \
  chrony \
  sharutils \
  graphviz \
  python-setuptools \
  python3-pip \
  ros-noetic-gazebo-msgs \
  ####################\
  # python3 deps      \
  ####################\
  python3-pip \
  python3-empy \
  python3-setuptools \
  python3-pyqt5 \
  python3-pyqt5.qtsvg \
  python3-pydot \
  python3-tk \
  scrot \
  libdrm-dev \
  ansible

RUN sudo usermod -a -G dialout developer
RUN sudo usermod -a -G tty developer
RUN sudo usermod -a -G video developer
RUN sudo usermod -a -G root developer

RUN sudo ln -s /usr/include/sdformat-6.3/sdf /usr/include/sdf

# //////////////////////////////////////////////////////////////////////////////
# mmpug workspace deps.

# python3 deps
RUN sudo -H pip3 install --upgrade pip
# RUN sudo -H pip3 install ifcfg wheel setuptools pexpect cython PyYAML jinja2  defusedxml netifaces python-dotenv graphviz opencv-python==4.6.0.66 pyserial xdot pycairo python-xlib
# RUN sudo -H pip3 install numpy==1.19.4 scipy psutil pyquaternion rosdep rospkg rosinstall_generator rosinstall wstool vcstools catkin_tools catkin_pkg SIP python-dotenv Jinja2 pyautogui pyqtgraph install loguru h5py dotmap overrides utm rosnumpy sk-video
RUN sudo ln -sf /usr/bin/python3 /usr/bin/python

RUN sudo apt update && sudo apt install -y liborocos-kdl-dev python3-pykdl autoconf bc build-essential g++-8 gcc-8 clang-8 lld-8 gettext-base gfortran-8 iputils-ping libbz2-dev libc++-dev libcgal-dev libffi-dev libfreetype6-dev libhdf5-dev libjpeg-dev liblzma-dev libncurses5-dev libncursesw5-dev libpng-dev libreadline-dev libssl-dev libsqlite3-dev libxml2-dev libxslt-dev locales moreutils openssl python-openssl rsync scons python3-pip libopenblas-dev
RUN (sudo rosdep init && rosdep update) || echo failed


# Clean up :)
RUN sudo apt-get clean \
 && sudo rm -rf /var/lib/apt/lists/*

# create ~/.Xauthority
RUN touch ~/.Xauthority


