import getopt
import sys

def no_catch_opt(o, a, svars):
  return False

class empty():
  pass

def usage(argtext, modules):
  """
  Print help documentation for options for the given script. Modules
  are iterated over to find option descriptions.

  Arguments:
    argtext: str - Description of script-specific arguments
    modules: list - List of modules used by the script
  """
  print("Arguments:\n"
        + "-h Print usage", file=sys.stderr)
  for module in modules:
    if hasattr(module, "argtext"):
      print(module.argtext, file=sys.stderr)
  print(argtext, file=sys.stderr)

def run(main_func, argtext="", svars=empty, shortopts="", longopts=[], catch_opt=no_catch_opt, modules=[]):
  """
  Run a script given a specified set of parts

  Arguments:
    main_func: function - Main function of the script. Must adhere to
      the following format:
        def main_func(svars, module1, module2, ...)
      where svars is the svars object and the modules are the unpacked
      values given in the modules list.
    argtext: str - Description of script-specific arguments
    svars: class - Set of variables to process for script before
      running main_func and passed to main_func. These can hold the
      values of options.
    shortopts: str - List of short options processed by the script,
      used by gnu_getopt(). This does not include options of modules.
    longopts: str - List of long options processed by the script, used
      by gnu_getopt(). This does not include options of modules.
    catch_opt: function - Function to use for catching options. Must
      adhere to the following format:
        def catch_opt(o, a, svars)
      where o is the option name from gnu_getopt, a is the option value
      from gnu_getopt (may not be set), and svars is the svars object.
      The function must return True if it catches and processes the
      argument and False otherwise.
    modules: list - List of module objects used by the script. These
      are passed to main_func. The modules may have corresponding
      shortopts, longopts, argtext, and catch_opt attributes.
  """
  shortopts += "h"

  # Collect options from modules
  for module in modules:
    if hasattr(module, "shortopts"):
      shortopts += module.shortopts
    if hasattr(module, "longopts"):
      longopts += module.longopts

  # Check for duplicated argument names
  longopts_text = [longopt.strip("=") for longopt in longopts]
  for longopt in longopts_text:
    if longopts_text.count(longopt) != 1:
      raise RuntimeError("Duplicate long option --%s" %(longopt))
  for shortopt in shortopts:
    if shortopt != ":":
      if shortopts.count(shortopt) != 1:
        raise RuntimeError("Duplicate short option -%s" %(shortopt))

  # Process argments from command line invocation
  try:
    opts, args = getopt.gnu_getopt(sys.argv[1:], shortopts, longopts)
  except getopt.GetoptError as err:
    print(err, file=sys.stderr)
    usage(argtext, modules)
    sys.exit(1)

  for o, a in opts:
    if o == "-h":
      usage(argtext, modules)
      sys.exit(0)
    elif catch_opt(o, a, svars) == True:
      pass
    else:
      for module in modules:
        if hasattr(module, "catch_opt"):
          if module.catch_opt(o, a) == True:
            break

  # Run main function of script
  return main_func(svars, *modules)
