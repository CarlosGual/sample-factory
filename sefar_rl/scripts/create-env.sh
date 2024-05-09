# Install ViZDoom deps from
# https://github.com/mwydmuch/ViZDoom/blob/master/doc/Building.md#-linux

# Set up conda
source ~/miniconda3/etc/profile.d/conda.sh

# Create a conda environment
conda create -n sefar-rl python=3.8 -y
conda activate sefar-rl

apt-get install build-essential zlib1g-dev libsdl2-dev libjpeg-dev \
nasm tar libbz2-dev libgtk2.0-dev cmake git libfluidsynth-dev libgme-dev \
libopenal-dev timidity libwildmidi-dev unzip ffmpeg

# Boost libraries
apt-get install libboost-all-dev

# Lua binding dependencies
apt-get install liblua5.1-dev

 install python libraries
 thanks toinsson

pip install vizdoom
pip install -e .[dev,vizdoom]