# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.6

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/jh/clion-2016.3.4/bin/cmake/bin/cmake

# The command to remove a file.
RM = /home/jh/clion-2016.3.4/bin/cmake/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jh/VINS-Mono/feature_tracker

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jh/VINS-Mono/feature_tracker/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/feature_tracker.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/feature_tracker.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/feature_tracker.dir/flags.make

CMakeFiles/feature_tracker.dir/src/feature_tracker_node.cpp.o: CMakeFiles/feature_tracker.dir/flags.make
CMakeFiles/feature_tracker.dir/src/feature_tracker_node.cpp.o: ../src/feature_tracker_node.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jh/VINS-Mono/feature_tracker/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/feature_tracker.dir/src/feature_tracker_node.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/feature_tracker.dir/src/feature_tracker_node.cpp.o -c /home/jh/VINS-Mono/feature_tracker/src/feature_tracker_node.cpp

CMakeFiles/feature_tracker.dir/src/feature_tracker_node.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/feature_tracker.dir/src/feature_tracker_node.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jh/VINS-Mono/feature_tracker/src/feature_tracker_node.cpp > CMakeFiles/feature_tracker.dir/src/feature_tracker_node.cpp.i

CMakeFiles/feature_tracker.dir/src/feature_tracker_node.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/feature_tracker.dir/src/feature_tracker_node.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jh/VINS-Mono/feature_tracker/src/feature_tracker_node.cpp -o CMakeFiles/feature_tracker.dir/src/feature_tracker_node.cpp.s

CMakeFiles/feature_tracker.dir/src/feature_tracker_node.cpp.o.requires:

.PHONY : CMakeFiles/feature_tracker.dir/src/feature_tracker_node.cpp.o.requires

CMakeFiles/feature_tracker.dir/src/feature_tracker_node.cpp.o.provides: CMakeFiles/feature_tracker.dir/src/feature_tracker_node.cpp.o.requires
	$(MAKE) -f CMakeFiles/feature_tracker.dir/build.make CMakeFiles/feature_tracker.dir/src/feature_tracker_node.cpp.o.provides.build
.PHONY : CMakeFiles/feature_tracker.dir/src/feature_tracker_node.cpp.o.provides

CMakeFiles/feature_tracker.dir/src/feature_tracker_node.cpp.o.provides.build: CMakeFiles/feature_tracker.dir/src/feature_tracker_node.cpp.o


CMakeFiles/feature_tracker.dir/src/parameters.cpp.o: CMakeFiles/feature_tracker.dir/flags.make
CMakeFiles/feature_tracker.dir/src/parameters.cpp.o: ../src/parameters.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jh/VINS-Mono/feature_tracker/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/feature_tracker.dir/src/parameters.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/feature_tracker.dir/src/parameters.cpp.o -c /home/jh/VINS-Mono/feature_tracker/src/parameters.cpp

CMakeFiles/feature_tracker.dir/src/parameters.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/feature_tracker.dir/src/parameters.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jh/VINS-Mono/feature_tracker/src/parameters.cpp > CMakeFiles/feature_tracker.dir/src/parameters.cpp.i

CMakeFiles/feature_tracker.dir/src/parameters.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/feature_tracker.dir/src/parameters.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jh/VINS-Mono/feature_tracker/src/parameters.cpp -o CMakeFiles/feature_tracker.dir/src/parameters.cpp.s

CMakeFiles/feature_tracker.dir/src/parameters.cpp.o.requires:

.PHONY : CMakeFiles/feature_tracker.dir/src/parameters.cpp.o.requires

CMakeFiles/feature_tracker.dir/src/parameters.cpp.o.provides: CMakeFiles/feature_tracker.dir/src/parameters.cpp.o.requires
	$(MAKE) -f CMakeFiles/feature_tracker.dir/build.make CMakeFiles/feature_tracker.dir/src/parameters.cpp.o.provides.build
.PHONY : CMakeFiles/feature_tracker.dir/src/parameters.cpp.o.provides

CMakeFiles/feature_tracker.dir/src/parameters.cpp.o.provides.build: CMakeFiles/feature_tracker.dir/src/parameters.cpp.o


CMakeFiles/feature_tracker.dir/src/feature_tracker.cpp.o: CMakeFiles/feature_tracker.dir/flags.make
CMakeFiles/feature_tracker.dir/src/feature_tracker.cpp.o: ../src/feature_tracker.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jh/VINS-Mono/feature_tracker/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/feature_tracker.dir/src/feature_tracker.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/feature_tracker.dir/src/feature_tracker.cpp.o -c /home/jh/VINS-Mono/feature_tracker/src/feature_tracker.cpp

CMakeFiles/feature_tracker.dir/src/feature_tracker.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/feature_tracker.dir/src/feature_tracker.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jh/VINS-Mono/feature_tracker/src/feature_tracker.cpp > CMakeFiles/feature_tracker.dir/src/feature_tracker.cpp.i

CMakeFiles/feature_tracker.dir/src/feature_tracker.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/feature_tracker.dir/src/feature_tracker.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jh/VINS-Mono/feature_tracker/src/feature_tracker.cpp -o CMakeFiles/feature_tracker.dir/src/feature_tracker.cpp.s

CMakeFiles/feature_tracker.dir/src/feature_tracker.cpp.o.requires:

.PHONY : CMakeFiles/feature_tracker.dir/src/feature_tracker.cpp.o.requires

CMakeFiles/feature_tracker.dir/src/feature_tracker.cpp.o.provides: CMakeFiles/feature_tracker.dir/src/feature_tracker.cpp.o.requires
	$(MAKE) -f CMakeFiles/feature_tracker.dir/build.make CMakeFiles/feature_tracker.dir/src/feature_tracker.cpp.o.provides.build
.PHONY : CMakeFiles/feature_tracker.dir/src/feature_tracker.cpp.o.provides

CMakeFiles/feature_tracker.dir/src/feature_tracker.cpp.o.provides.build: CMakeFiles/feature_tracker.dir/src/feature_tracker.cpp.o


# Object files for target feature_tracker
feature_tracker_OBJECTS = \
"CMakeFiles/feature_tracker.dir/src/feature_tracker_node.cpp.o" \
"CMakeFiles/feature_tracker.dir/src/parameters.cpp.o" \
"CMakeFiles/feature_tracker.dir/src/feature_tracker.cpp.o"

# External object files for target feature_tracker
feature_tracker_EXTERNAL_OBJECTS =

devel/lib/feature_tracker/feature_tracker: CMakeFiles/feature_tracker.dir/src/feature_tracker_node.cpp.o
devel/lib/feature_tracker/feature_tracker: CMakeFiles/feature_tracker.dir/src/parameters.cpp.o
devel/lib/feature_tracker/feature_tracker: CMakeFiles/feature_tracker.dir/src/feature_tracker.cpp.o
devel/lib/feature_tracker/feature_tracker: CMakeFiles/feature_tracker.dir/build.make
devel/lib/feature_tracker/feature_tracker: /opt/ros/indigo/lib/libcv_bridge.so
devel/lib/feature_tracker/feature_tracker: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.2.4.8
devel/lib/feature_tracker/feature_tracker: /usr/lib/x86_64-linux-gnu/libopencv_video.so.2.4.8
devel/lib/feature_tracker/feature_tracker: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.2.4.8
devel/lib/feature_tracker/feature_tracker: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.2.4.8
devel/lib/feature_tracker/feature_tracker: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.2.4.8
devel/lib/feature_tracker/feature_tracker: /usr/lib/x86_64-linux-gnu/libopencv_ocl.so.2.4.8
devel/lib/feature_tracker/feature_tracker: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.2.4.8
devel/lib/feature_tracker/feature_tracker: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.2.4.8
devel/lib/feature_tracker/feature_tracker: /usr/lib/x86_64-linux-gnu/libopencv_legacy.so.2.4.8
devel/lib/feature_tracker/feature_tracker: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.2.4.8
devel/lib/feature_tracker/feature_tracker: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.2.4.8
devel/lib/feature_tracker/feature_tracker: /usr/lib/x86_64-linux-gnu/libopencv_gpu.so.2.4.8
devel/lib/feature_tracker/feature_tracker: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.2.4.8
devel/lib/feature_tracker/feature_tracker: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.2.4.8
devel/lib/feature_tracker/feature_tracker: /usr/lib/x86_64-linux-gnu/libopencv_core.so.2.4.8
devel/lib/feature_tracker/feature_tracker: /usr/lib/x86_64-linux-gnu/libopencv_contrib.so.2.4.8
devel/lib/feature_tracker/feature_tracker: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.2.4.8
devel/lib/feature_tracker/feature_tracker: /home/jh/catkin_ws/devel/lib/libcamera_model.so
devel/lib/feature_tracker/feature_tracker: /opt/ros/indigo/lib/libroscpp.so
devel/lib/feature_tracker/feature_tracker: /usr/lib/x86_64-linux-gnu/libboost_signals.so
devel/lib/feature_tracker/feature_tracker: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
devel/lib/feature_tracker/feature_tracker: /opt/ros/indigo/lib/librosconsole.so
devel/lib/feature_tracker/feature_tracker: /opt/ros/indigo/lib/librosconsole_log4cxx.so
devel/lib/feature_tracker/feature_tracker: /opt/ros/indigo/lib/librosconsole_backend_interface.so
devel/lib/feature_tracker/feature_tracker: /usr/lib/liblog4cxx.so
devel/lib/feature_tracker/feature_tracker: /usr/lib/x86_64-linux-gnu/libboost_regex.so
devel/lib/feature_tracker/feature_tracker: /opt/ros/indigo/lib/libxmlrpcpp.so
devel/lib/feature_tracker/feature_tracker: /opt/ros/indigo/lib/libroscpp_serialization.so
devel/lib/feature_tracker/feature_tracker: /opt/ros/indigo/lib/librostime.so
devel/lib/feature_tracker/feature_tracker: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
devel/lib/feature_tracker/feature_tracker: /opt/ros/indigo/lib/libcpp_common.so
devel/lib/feature_tracker/feature_tracker: /usr/lib/x86_64-linux-gnu/libboost_system.so
devel/lib/feature_tracker/feature_tracker: /usr/lib/x86_64-linux-gnu/libboost_thread.so
devel/lib/feature_tracker/feature_tracker: /usr/lib/x86_64-linux-gnu/libpthread.so
devel/lib/feature_tracker/feature_tracker: /usr/lib/x86_64-linux-gnu/libconsole_bridge.so
devel/lib/feature_tracker/feature_tracker: /usr/local/lib/libopencv_videostab.so.2.4.13
devel/lib/feature_tracker/feature_tracker: /usr/local/lib/libopencv_superres.so.2.4.13
devel/lib/feature_tracker/feature_tracker: /usr/local/lib/libopencv_stitching.so.2.4.13
devel/lib/feature_tracker/feature_tracker: /usr/local/lib/libopencv_contrib.so.2.4.13
devel/lib/feature_tracker/feature_tracker: /usr/local/lib/libopencv_nonfree.so.2.4.13
devel/lib/feature_tracker/feature_tracker: /usr/local/lib/libopencv_ocl.so.2.4.13
devel/lib/feature_tracker/feature_tracker: /usr/local/lib/libopencv_gpu.so.2.4.13
devel/lib/feature_tracker/feature_tracker: /usr/local/lib/libopencv_photo.so.2.4.13
devel/lib/feature_tracker/feature_tracker: /usr/local/lib/libopencv_objdetect.so.2.4.13
devel/lib/feature_tracker/feature_tracker: /usr/local/lib/libopencv_legacy.so.2.4.13
devel/lib/feature_tracker/feature_tracker: /usr/local/lib/libopencv_video.so.2.4.13
devel/lib/feature_tracker/feature_tracker: /usr/local/lib/libopencv_ml.so.2.4.13
devel/lib/feature_tracker/feature_tracker: /usr/local/lib/libopencv_calib3d.so.2.4.13
devel/lib/feature_tracker/feature_tracker: /usr/local/lib/libopencv_features2d.so.2.4.13
devel/lib/feature_tracker/feature_tracker: /usr/local/lib/libopencv_highgui.so.2.4.13
devel/lib/feature_tracker/feature_tracker: /usr/local/lib/libopencv_imgproc.so.2.4.13
devel/lib/feature_tracker/feature_tracker: /usr/local/lib/libopencv_flann.so.2.4.13
devel/lib/feature_tracker/feature_tracker: /usr/local/lib/libopencv_core.so.2.4.13
devel/lib/feature_tracker/feature_tracker: CMakeFiles/feature_tracker.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jh/VINS-Mono/feature_tracker/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable devel/lib/feature_tracker/feature_tracker"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/feature_tracker.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/feature_tracker.dir/build: devel/lib/feature_tracker/feature_tracker

.PHONY : CMakeFiles/feature_tracker.dir/build

CMakeFiles/feature_tracker.dir/requires: CMakeFiles/feature_tracker.dir/src/feature_tracker_node.cpp.o.requires
CMakeFiles/feature_tracker.dir/requires: CMakeFiles/feature_tracker.dir/src/parameters.cpp.o.requires
CMakeFiles/feature_tracker.dir/requires: CMakeFiles/feature_tracker.dir/src/feature_tracker.cpp.o.requires

.PHONY : CMakeFiles/feature_tracker.dir/requires

CMakeFiles/feature_tracker.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/feature_tracker.dir/cmake_clean.cmake
.PHONY : CMakeFiles/feature_tracker.dir/clean

CMakeFiles/feature_tracker.dir/depend:
	cd /home/jh/VINS-Mono/feature_tracker/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jh/VINS-Mono/feature_tracker /home/jh/VINS-Mono/feature_tracker /home/jh/VINS-Mono/feature_tracker/cmake-build-debug /home/jh/VINS-Mono/feature_tracker/cmake-build-debug /home/jh/VINS-Mono/feature_tracker/cmake-build-debug/CMakeFiles/feature_tracker.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/feature_tracker.dir/depend
