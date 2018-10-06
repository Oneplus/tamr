include_directories (${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/src/left_to_right/)

add_library (parser_l2r_train
    train.cc
    train.h
    train_supervised.cc
    train_supervised.h
    algorithm.h)

target_link_libraries (parser_l2r_train parser_l2r_parser)
