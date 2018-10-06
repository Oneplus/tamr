#include "system.h"
#include "logging.h"
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>


unsigned TransitionSystem::get_action_arg1(const Alphabet & map, const unsigned &action) {
  std::vector<std::string> terms;
  std::string a_str = action_map.get(action);
  boost::algorithm::split(terms, a_str, boost::is_any_of(" \t"), boost::token_compress_on);
  return map.get(terms[1]);
}