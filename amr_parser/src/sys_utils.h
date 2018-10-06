#ifndef SYS_UTILS_H
#define SYS_UTILS_H

#include <iostream>

int portable_getpid();

float execute_and_get_result(const std::string& cmd);

#endif  //  end for SYS_UTILS_H