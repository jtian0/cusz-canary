add_executable(demo_capi_cu src/demo_capi.cu_hip.cc)
target_link_libraries(demo_capi_cu PRIVATE cusz)

add_library(ex_utils src/ex_utils.cu)
target_link_libraries(ex_utils PUBLIC psz_cu_compile_settings)

add_executable(bin_pipeline_cu src/bin_pipeline.cu_hip.cc)
target_link_libraries(bin_pipeline_cu PRIVATE psz_cu_testframe_legacy CUDA::cudart)

add_executable(bin_hf src/bin_hf.cc)
target_link_libraries(bin_hf PRIVATE cusz psz_cu_stat psz_cu_phf CUDA::cudart)
set_target_properties(bin_hf PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(bin_hist src/bin_hist.cc)
target_link_libraries(bin_hist PRIVATE psz_cu_core psz_seq_core psz_cu_mem
  psz_cu_stat CUDA::cudart)
