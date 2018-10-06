include_directories (${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/src/left_to_right/ ${PROJECT_SOURCE_DIR}/src/left_to_right/s2a/)

add_subdirectory (parser)
add_subdirectory (train)
add_subdirectory (decode)
add_subdirectory (evaluate)
add_subdirectory (system)

add_executable (parser_l2r main.cc)

target_link_libraries (parser_l2r
    parser_l2r_system
    parser_l2r_parser
    parser_l2r_train
    parser_l2r_decode
    parser_l2r_evaluate
    dynet
    dynet_layer
    common
    ${LIBS})

add_executable (ensemble ensemble.cc)

target_link_libraries (ensemble
        parser_l2r_system
        parser_l2r_parser
        parser_l2r_train
        parser_l2r_decode
        parser_l2r_evaluate
        dynet
        dynet_layer
        common
    ${LIBS})

if(UNIX AND NOT APPLE)
    target_link_libraries (parser_l2r rt)
endif()
