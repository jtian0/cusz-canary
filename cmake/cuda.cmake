add_compile_definitions(PSZ_USE_CUDA)
add_compile_definitions(_PORTABLE_USE_CUDA)

find_package(CUDAToolkit REQUIRED)

include(GNUInstallDirs)
include(CTest)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/psz/src/cusz_version.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/psz/include/cusz_version.h)

add_library(psz_cu_compile_settings INTERFACE)

target_compile_definitions(
  psz_cu_compile_settings
  INTERFACE $<$<COMPILE_LANG_AND_ID:CUDA,Clang>:__STRICT_ANSI__>)
target_compile_options(
  psz_cu_compile_settings
  INTERFACE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--extended-lambda
            --expt-relaxed-constexpr -Wno-deprecated-declarations>)
target_compile_features(psz_cu_compile_settings INTERFACE cxx_std_17 cuda_std_17)
target_include_directories(
  psz_cu_compile_settings
  INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/portable/include/>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/psz/src/>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/psz/include/>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/codec/hf/include/>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/codec/hf/src/>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/codec/fzg/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/codec/fzg/src>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/cusz>
)

# option(PSZ_RESEARCH_HUFFBK_CUDA
#        "build research artifacts: create Huffman codebook on GPU" OFF)
option(PSZ_REACTIVATE_THRUSTGPU
       "build previously thrust implemented functions" OFF)

if(PSZ_REACTIVATE_THRUSTGPU)
  add_compile_definitions(REACTIVATE_THRUSTGPU)
  add_library(psz_cu_stat
    psz/src/stat/compare.stl.cc
    psz/src/stat/extrema.cu
    psz/src/stat/cmpg1_4.cu
    # psz/src/stat/cmpg1_5.cu
    psz/src/stat/cmpg2.cu
    psz/src/stat/cmpg3.cu
    psz/src/stat/cmpg4_1.cu
    psz/src/stat/cmpg4_2.cu
    psz/src/stat/cmpg5_1.cu
    psz/src/stat/cmpg5_2.cu
  )
else()
  add_library(psz_cu_stat 
    psz/src/stat/compare.stl.cc
    psz/src/stat/extrema.cu 
    psz/src/stat/cmpg2.cu 
    psz/src/stat/cmpg4_1.cu
    psz/src/stat/cmpg4_2.cu 
    psz/src/stat/cmpg5_1.cu 
    psz/src/stat/cmpg5_2.cu
  )
endif()
target_link_libraries(psz_cu_stat PUBLIC psz_cu_compile_settings)

# FUNC={core,api}, BACKEND={serial,cuda,...}
add_library(psz_seq_core 
  psz/src/kernel/l23.seq.cc 
  psz/src/kernel/hist.seq.cc
  psz/src/kernel/histsp.seq.cc 
  psz/src/kernel/spvn.seq.cc
)
target_link_libraries(psz_seq_core 
  PUBLIC 
  psz_cu_compile_settings
)

add_library(psz_cu_core
  psz/src/kernel/lproto.cu
  psz/src/kernel/spvn.cu
  psz/src/kernel/spv_legacy.cu
  psz/src/kernel/l23_c.cu
  psz/src/kernel/l23_x.cu
  psz/src/kernel/spline3.cu
  psz/src/kernel/hist.cu
  psz/src/kernel/hist.seq.cc # workaround
  psz/src/kernel/histsp.cu
  psz/src/kernel/histsp.seq.cc
  psz/src/kernel/l23r.cu)
target_link_libraries(psz_cu_core PUBLIC psz_cu_compile_settings)

add_library(psz_cu_mem 
  portable/src/mem/memobj.f.cc
  portable/src/mem/memobj.i.cc
  portable/src/mem/memobj.u.cc
  portable/src/mem/memobj.misc.cc
)
target_link_libraries(psz_cu_mem PUBLIC psz_cu_compile_settings CUDA::cudart)

add_library(psz_cu_utils 
  psz/src/utils/verinfo.cc
  psz/src/utils/verinfo.cu
  psz/src/utils/verinfo_nv.cu
  psz/src/utils/vis_stat.cc 
  psz/src/utils/context.cc
  psz/src/utils/timer_cpu.cc
  psz/src/utils/timer_gpu.cc
)
target_link_libraries(psz_cu_utils 
  PUBLIC 
  psz_cu_compile_settings
  CUDA::cudart CUDA::nvml
)

add_library(psz_cu_phf 
  codec/hf/src/hf_bk_impl1.seq.cc 
  codec/hf/src/hf_bk_impl2.seq.cc
  codec/hf/src/hf_bk_internal.seq.cc
  codec/hf/src/hf_bk.seq.cc
  codec/hf/src/hf_kernels.cu
  codec/hf/src/hf_ood.cc
  codec/hf/src/hf_hl.cc
  codec/hf/src/hf_buf.cc
  codec/hf/src/hf_canon.seq.cc
  codec/hf/src/libphf.cc
)
target_link_libraries(psz_cu_phf 
  PUBLIC
  psz_cu_compile_settings
  psz_cu_stat
  CUDA::cuda_driver
)

add_library(psz_cu_testframe_legacy 
  psz/src/pipeline/testframe.cc
)
target_link_libraries(psz_cu_testframe_legacy 
  PUBLIC 
  psz_cu_mem 
  psz_cu_utils
)

add_library(lc 
            third_party/lc_gen/comp-tcms.cu third_party/lc_gen/decomp-tcms.cu 
            third_party/lc_gen/comp-bitr.cu third_party/lc_gen/decomp-bitr.cu
            third_party/lc_gen/comp-rtr.cu  third_party/lc_gen/decomp-rtr.cu)
target_include_directories(
  psz_cu_compile_settings
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/cusz>)
target_compile_options(lc
  PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
      -O3
      -fmad=false>
    $<$<COMPILE_LANGUAGE:CXX>:
      -O3
      -march=native
      -mno-fma>)
target_link_libraries(lc PUBLIC psz_cu_compile_settings CUDA::cudart)

add_library(cusz 
  psz/src/compressor.cc 
  psz/src/cusz_lib.cc
)
target_link_libraries(cusz 
  PUBLIC 
  psz_cu_compile_settings
  psz_cu_core
  psz_cu_stat
  psz_cu_phf 
  psz_cu_utils 
  psz_cu_mem 
  lc
)

add_executable(cusz-bin psz/src/cli_psz.cc)
target_link_libraries(cusz-bin PRIVATE cusz)
set_target_properties(cusz-bin PROPERTIES OUTPUT_NAME cuszhi)

# enable examples and testing
if(PSZ_BUILD_EXAMPLES)
  add_subdirectory(example)
endif()

if(BUILD_TESTING)
  # add_subdirectory(test)
endif()

# installation
install(TARGETS psz_cu_compile_settings EXPORT CUSZTargets)

install(TARGETS psz_seq_core EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS psz_cu_core EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS psz_cu_stat EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS psz_cu_mem EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS psz_cu_utils EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS psz_cu_phf EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS psz_cu_testframe_legacy EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS lc EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS cusz EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS cusz-bin EXPORT CUSZTargets)
if(PSZ_RESEARCH_HUFFBK_CUDA)
  install(TARGETS pszhfbook_cu EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif(PSZ_RESEARCH_HUFFBK_CUDA)

install(
  EXPORT CUSZTargets
  NAMESPACE CUSZ::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CUSZ/)
include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/CUSZConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/CUSZConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CUSZ)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/CUSZConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY AnyNewerVersion)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/CUSZConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/CUSZConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CUSZ)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cusz)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/cusz_version.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cusz/)
