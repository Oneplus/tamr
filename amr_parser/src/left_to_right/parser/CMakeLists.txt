include_directories (${PROJECT_SOURCE_DIR}/src/ ${PROJECT_SOURCE_DIR}/src/left_to_right/)

add_library (parser_l2r_parser
    parser.cc
    parser.h
        parser_swap.cc
        parser_swap.h
    parser_eager.cc
    parser_eager.h
    parser_builder.cc
    parser_builder.h)

target_link_libraries (parser_l2r_parser parser_l2r_system)
