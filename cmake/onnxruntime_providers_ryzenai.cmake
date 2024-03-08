# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

  # if ("${GIT_COMMIT_ID}" STREQUAL "")
  # execute_process(
  #   WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  #   COMMAND git rev-parse HEAD
  #   OUTPUT_VARIABLE GIT_COMMIT_ID
  #   OUTPUT_STRIP_TRAILING_WHITESPACE)
  # endif()
  # configure_file(${ONNXRUNTIME_ROOT}/core/providers/ryzenai/imp/version_info.hpp.in ${CMAKE_CURRENT_BINARY_DIR}/ryzenai/version_info.h)
  file(GLOB onnxruntime_providers_ryzenai_cc_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/ryzenai/*.cc"
    "${ONNXRUNTIME_ROOT}/core/providers/ryzenai/*.h"
  )
  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_ryzenai_cc_srcs})
  onnxruntime_add_static_library(onnxruntime_providers_ryzenai ${onnxruntime_providers_ryzenai_cc_srcs})
  onnxruntime_add_include_to_target(onnxruntime_providers_ryzenai onnxruntime_common onnxruntime_framework onnx onnx_proto ${PROTOBUF_LIB} flatbuffers::flatbuffers)
  #target_link_libraries(onnxruntime_providers_ryzenai PRIVATE onnxruntime_common onnxruntime_framework)
  onnxruntime_add_include_to_target(onnxruntime_providers_ryzenai dbghelp)
  #set_property(TARGET onnxruntime_providers_ryzenai APPEND_STRING PROPERTY LINK_FLAGS "-DEF:${ONNXRUNTIME_ROOT}/core/providers/ryzenai/symbols.def")


  target_include_directories(onnxruntime_providers_ryzenai PRIVATE "${ONNXRUNTIME_ROOT}/core/providers/ryzenai/"  ${CMAKE_CURRENT_BINARY_DIR}/ryzenai)
  target_compile_options(onnxruntime_providers_ryzenai PRIVATE "/Zc:__cplusplus")
  # for dll interface warning.
  target_compile_options(onnxruntime_providers_ryzenai PRIVATE "/wd4251")
  # for unused formal parameter
  target_compile_options(onnxruntime_providers_ryzenai PRIVATE "/wd4100")
  # for type name first seen using 'class' now seen using 'struct'
  target_compile_options(onnxruntime_providers_ryzenai PRIVATE "/wd4099")

  set_target_properties(onnxruntime_providers_ryzenai PROPERTIES FOLDER "ONNXRuntime")
  set_target_properties(onnxruntime_providers_ryzenai PROPERTIES LINKER_LANGUAGE CXX)

  install(TARGETS onnxruntime_providers_ryzenai
          ARCHIVE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
          LIBRARY   DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RUNTIME   DESTINATION ${CMAKE_INSTALL_BINDIR}
          FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR})
