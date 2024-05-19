# reference: https://gitlab.mpi-magdeburg.mpg.de/mess/cmess-releases/-/blob/master/CMakeModules/FindCHOLMOD.cmake
find_path(CHOLMOD_INCLUDES
  NAMES
  cholmod.h
  PATHS
  $ENV{CHOLMODDIR}
  PATH_SUFFIXES
  suitesparse
  ufsparse
)
find_library(CHOLMOD_LIBRARIES cholmod camd ccolamd amd colamd PATHS $ENV{CHOLMODDIR})

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(cholmod DEFAULT_MSG CHOLMOD_LIBRARIES CHOLMOD_INCLUDES)
MARK_AS_ADVANCED(CHOLMOD_INCLUDES CHOLMOD_LIBRARIES)
