# cmake/strata_ggml_sm61_mmq_overlay.cmake - the sm_61 (P104-class) MMQ config overlay: upstream llama.cpp PR
# #31352 (2026-10-29; the companion issue #30235 measured +12% pp512 on an sm_61 card) applied to the pinned
# ggml-cuda headers, written at configure time into the build tree.
#
# This script copies the pinned ggml-cuda header set (every top-level .cuh / .h, plus the vendors/ subdirectory
# when the pin has one) into the out dir: quote-includes resolve next to the including file, so a single-header
# shadow cannot work - the whole set must sit in the build dir so that an overlaid mmq.cuh finds its config and
# loader headers beside itself instead of back in the pinned directory.
#
# The gate (the mmq-tab hash-gate pattern of CMakeLists.txt, keyed on the 3cf03257f pin): the SHA-256 of the
# pinned mmq-config-pascal-dp4a.cuh must be the whitelisted one below.  On a mismatch (the pin moved, the file
# changed) the overlay disables itself: a STATUS is printed and nothing is written, so the build keeps the
# pinned table and the default object code is untouched.  The gate file is a CMAKE_CONFIGURE_DEPENDS, so a pin
# move re-runs the configure and re-evaluates the gate.
#
# The change: with <patch> = ON the #31352 diff is applied verbatim to the build-tree copy - the two J=64 rows
# of 11 types (Q1_0 Q2_0 Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 IQ2_XXS IQ3_XXS IQ4_XS IQ4_NL; fallback true and fallback
# false), I=64 -> I=128, no other field.  Every row is anchored on the exact pinned text and must occur exactly
# once, and the patched result must hash to the known #31352 result before it is written; any failure keeps the
# copy byte-identical to the pin and turns the overlay off.  With <patch> = OFF the copy is byte-identical to
# the pinned one (the companion ctest asserts it).
#
# The caller guards the diff to cc 610: the pascal-dp4a table is picked at run time only for compute 6.1
# devices (ggml_cuda_mmq_get_config: the highest compiled cc >= GGML_CUDA_CC_DP4A and < GGML_CUDA_CC_VOLTA), so
# no other card ever reads the patched rows.
#
# No source-tree mutation: everything is written into the out dir.  The remote tree is a non-git copy, the same
# as the TQ overlay's (cmake/strata_ggml_tq_overlay.cmake).
#
#   strata_ggml_sm61_mmq_overlay(<llama.cpp dir> <out dir> <patch ON|OFF>)  ->  STRATA_SM61_MMQ_OVERLAY_OK (parent scope)

function(strata_ggml_sm61_mmq_overlay LLAMA OUT PATCH)
  set(STRATA_SM61_MMQ_OVERLAY_OK OFF PARENT_SCOPE)
  set(_cu "${LLAMA}/ggml/src/ggml-cuda")
  set(_cfg "${_cu}/mmq-config-pascal-dp4a.cuh")
  if(NOT EXISTS "${_cfg}")
    message(STATUS "sm61 MMQ config overlay: ${_cfg} is missing: off")
    return()
  endif()

  # ---- the SHA-256 gate: the pinned file must be the 3cf03257f one ---------------------------------------------
  set(_strata_sm61_mmq_pin_hash 67a93b6d26d3e16e77fd285dd77e7cca7848fb0b0d68897c2ff1e8cc47102263)
  set(_strata_sm61_mmq_patch_hash 375327932e07aad5b02024aee93aa45e675f5f63abd3cbe83bf84351b9789f81)
  file(READ "${_cfg}" _cfg_txt)
  string(REPLACE "\r\n" "\n" _cfg_txt "${_cfg_txt}")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_cfg}")   # a pin move re-runs the configure and re-checks the gate
  string(SHA256 _cfg_hash "${_cfg_txt}")
  if(NOT _cfg_hash STREQUAL _strata_sm61_mmq_pin_hash)
    message(STATUS "sm61 MMQ config overlay off: mmq-config-pascal-dp4a.cuh is not the pinned 3cf03257f file (${_cfg_hash}): the default table is untouched")
    return()
  endif()

  # ---- the pinned ggml-cuda header set, copied: quote-includes resolve next to the including file, so the whole
  # set goes - a single-header shadow would let a pinned neighbor pull the pinned mmq.cuh / configs back in ------
  file(MAKE_DIRECTORY "${OUT}")
  file(GLOB _all_cu "${_cu}/*.cuh" "${_cu}/*.h")
  foreach(_f IN LISTS _all_cu)
    get_filename_component(_n "${_f}" NAME)
    configure_file("${_f}" "${OUT}/${_n}" COPYONLY)
  endforeach()
  if(IS_DIRECTORY "${_cu}/vendors")
    file(COPY "${_cu}/vendors" DESTINATION "${OUT}")
  endif()

  if(NOT PATCH STREQUAL "ON")
    # byte-identical copy: the companion ctest (PATCH=OFF) proves it
    set(STRATA_SM61_MMQ_OVERLAY_OK ON PARENT_SCOPE)
    return()
  endif()

  # ---- the #31352 diff (2026-10-29) verbatim, onto the build-tree copy: two J=64 rows x 11 types, I=64 -> I=128 -
  set(_pairs "Q1_0|Q8_0;Q2_0|Q8_0;Q4_0|Q8_0;Q4_1|Q8_1;Q5_0|Q8_0;Q5_1|Q8_1;Q8_0|Q8_0;IQ2_XXS|Q8_0;IQ3_XXS|Q8_0;IQ4_XS|Q8_0;IQ4_NL|Q8_0")
  file(READ "${OUT}/mmq-config-pascal-dp4a.cuh" _t)
  string(REPLACE "\r\n" "\n" _t "${_t}")
  set(_fail "")
  foreach(_pl IN LISTS _pairs)
    string(REPLACE "|" ";" _pp "${_pl}")
    list(GET _pp 0 _ty)
    list(GET _pp 1 _l)
    foreach(_fb IN ITEMS true false)
      set(_old "    CASE(GGML_TYPE_${_ty}, 256, 2, 64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_${_l}, MMQ_ITER_K, false, ${_fb});")
      set(_new "    CASE(GGML_TYPE_${_ty}, 256, 2, 128,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_${_l}, MMQ_ITER_K, false, ${_fb});")
      string(FIND "${_t}" "${_old}" _p1)
      if(_p1 LESS 0)
        if(NOT _fail)
          set(_fail "anchor not found: GGML_TYPE_${_ty} fallback=${_fb}")
        endif()
      else()
        string(LENGTH "${_old}" _alen)
        math(EXPR _after "${_p1} + ${_alen}")
        string(SUBSTRING "${_t}" ${_after} -1 _tail)
        string(FIND "${_tail}" "${_old}" _dup)
        if(NOT _dup LESS 0)
          if(NOT _fail)
            set(_fail "anchor is not unique: GGML_TYPE_${_ty} fallback=${_fb}")
          endif()
        else()
          string(FIND "${_t}" "${_new}" _np)
          if(NOT _np LESS 0 AND NOT _fail)
            set(_fail "the diff is already applied: GGML_TYPE_${_ty} fallback=${_fb}")
          endif()
        endif()
      endif()
    endforeach()
  endforeach()
  if(_fail)
    # the copy stays byte-identical to the pin and the overlay turns off: the default object code is untouched
    message(STATUS "sm61 MMQ config overlay off: ${_fail}")
    return()
  endif()
  foreach(_pl IN LISTS _pairs)
    string(REPLACE "|" ";" _pp "${_pl}")
    list(GET _pp 0 _ty)
    list(GET _pp 1 _l)
    foreach(_fb IN ITEMS true false)
      set(_old "    CASE(GGML_TYPE_${_ty}, 256, 2, 64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_${_l}, MMQ_ITER_K, false, ${_fb});")
      set(_new "    CASE(GGML_TYPE_${_ty}, 256, 2, 128,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_${_l}, MMQ_ITER_K, false, ${_fb});")
      string(REPLACE "${_old}" "${_new}" _t "${_t}")
    endforeach()
  endforeach()
  string(SHA256 _p_hash "${_t}")
  if(NOT _p_hash STREQUAL _strata_sm61_mmq_patch_hash)
    message(STATUS "sm61 MMQ config overlay off: the patched file hashes to ${_p_hash}, not the known #31352 result: the copy stays pinned")
    return()
  endif()
  file(WRITE "${OUT}/mmq-config-pascal-dp4a.cuh" "${_t}")
  message(STATUS "sm61 MMQ config overlay: the #31352 diff (2026-10-29, I=128 on the J=64 rows, 11 types) on ${OUT}/mmq-config-pascal-dp4a.cuh")
  set(STRATA_SM61_MMQ_OVERLAY_OK ON PARENT_SCOPE)
endfunction()
