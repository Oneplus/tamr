add_library (common
    corpus.cc
    corpus.h
    ds.cc
    ds.h
    logging.cc
    logging.h
    math_utils.cc
    math_utils.h
    sys_utils.cc
    sys_utils.h
    trainer_utils.cc
    trainer_utils.h
    lstm.h
    lstm.cc)

target_link_libraries (common ${Boost_LIBRARIES})
add_subdirectory (left_to_right)
