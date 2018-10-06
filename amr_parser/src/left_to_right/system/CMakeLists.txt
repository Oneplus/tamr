include_directories (${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/src/left_to_right/)

add_library (parser_l2r_system
    swap.cc
    swap.h
    eager.cc
    eager.h
    system.cc
    system.h
    state.h
    state.cc)
