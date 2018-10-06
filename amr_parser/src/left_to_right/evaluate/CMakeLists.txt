include_directories (${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/src/left_to_right/)

add_library (parser_l2r_evaluate evaluate.cc evaluate.h)

target_link_libraries (parser_l2r_evaluate parser_l2r_parser)
