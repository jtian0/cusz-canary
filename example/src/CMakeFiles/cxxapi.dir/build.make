# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.20

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /autofs/nccs-svm1_sw/crusher/spack-envs/base/opt/cray-sles15-zen3/cce-14.0.0/cmake-3.20.6-jj4fb5uctkwbzuugdpjowgktd4fl27dp/bin/cmake

# The command to remove a file.
RM = /autofs/nccs-svm1_sw/crusher/spack-envs/base/opt/cray-sles15-zen3/cce-14.0.0/cmake-3.20.6-jj4fb5uctkwbzuugdpjowgktd4fl27dp/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /ccs/home/xiyu/cusz-hip/example

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /ccs/home/xiyu/cusz-hip/example/src

# Include any dependencies generated for this target.
include CMakeFiles/cxxapi.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/cxxapi.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/cxxapi.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/cxxapi.dir/flags.make

CMakeFiles/cxxapi.dir/cxxapi.cc.o: CMakeFiles/cxxapi.dir/flags.make
CMakeFiles/cxxapi.dir/cxxapi.cc.o: cxxapi.cc
CMakeFiles/cxxapi.dir/cxxapi.cc.o: CMakeFiles/cxxapi.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/ccs/home/xiyu/cusz-hip/example/src/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/cxxapi.dir/cxxapi.cc.o"
	/opt/rocm-5.1.0/bin/hipcc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/cxxapi.dir/cxxapi.cc.o -MF CMakeFiles/cxxapi.dir/cxxapi.cc.o.d -o CMakeFiles/cxxapi.dir/cxxapi.cc.o -c /ccs/home/xiyu/cusz-hip/example/src/cxxapi.cc

CMakeFiles/cxxapi.dir/cxxapi.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cxxapi.dir/cxxapi.cc.i"
	/opt/rocm-5.1.0/bin/hipcc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /ccs/home/xiyu/cusz-hip/example/src/cxxapi.cc > CMakeFiles/cxxapi.dir/cxxapi.cc.i

CMakeFiles/cxxapi.dir/cxxapi.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cxxapi.dir/cxxapi.cc.s"
	/opt/rocm-5.1.0/bin/hipcc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /ccs/home/xiyu/cusz-hip/example/src/cxxapi.cc -o CMakeFiles/cxxapi.dir/cxxapi.cc.s

# Object files for target cxxapi
cxxapi_OBJECTS = \
"CMakeFiles/cxxapi.dir/cxxapi.cc.o"

# External object files for target cxxapi
cxxapi_EXTERNAL_OBJECTS =

cxxapi: CMakeFiles/cxxapi.dir/cxxapi.cc.o
cxxapi: CMakeFiles/cxxapi.dir/build.make
cxxapi: CMakeFiles/cxxapi.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/ccs/home/xiyu/cusz-hip/example/src/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable cxxapi"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cxxapi.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/cxxapi.dir/build: cxxapi
.PHONY : CMakeFiles/cxxapi.dir/build

CMakeFiles/cxxapi.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/cxxapi.dir/cmake_clean.cmake
.PHONY : CMakeFiles/cxxapi.dir/clean

CMakeFiles/cxxapi.dir/depend:
	cd /ccs/home/xiyu/cusz-hip/example/src && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /ccs/home/xiyu/cusz-hip/example /ccs/home/xiyu/cusz-hip/example /ccs/home/xiyu/cusz-hip/example/src /ccs/home/xiyu/cusz-hip/example/src /ccs/home/xiyu/cusz-hip/example/src/CMakeFiles/cxxapi.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/cxxapi.dir/depend

