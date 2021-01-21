
Plugin based on this: http://gazebosim.org/tutorials?tut=animated_box#animated_box.world

´´´
cd build
cmake ../
make
´´´
Make sure Gazebo can load the plugins later 
´´´
export GAZEBO_PLUGIN_PATH=`pwd`:$GAZEBO_PLUGIN_PATH
´´´