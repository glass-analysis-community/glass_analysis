import numpy as np
import sys

# List of short and long options processed by this module, used by
# gnu_getopt()
shortopts = "fg:w:"
longopts = ["low-interval", "high-interval", "negvals"]

def usage():
  """
  Print help documentation for options processed by the progression
  module.
  """
  print("Interval increase progression (last specified is used):",
        "-f Flenner-style periodic-exponential-increasing increment (iterations: 50, power: 5)",
        "-g Geometric spacing progression, selectively dropped to fit on integer frames (argument is number of lags)",
        "-w Linear progression (argument is spacing between interval length values)",
        "Interval progression modifiers:",
        "--low-interval Trim set of interval lengths below frame number value",
        "--high-interval Trim set of interval lengths above frame number value",
        "--negvals Add mirrored sequence of negative length intervals",
        sep="\n", file=sys.stderr)

# Progression types

class flenner():
  """
  A progression of sets of 50 values, with a spacing between multiplied
  by 5 for each next set of 50 values.
  """
  pass

class geometric():
  """
  A geometric progression of a given number of values, rounded to the
  geometrically closest integer. This means that there may be less
  values in returned set than specified, due to more than one value
  rounding to the same integer.

  Attributes:
    geom_num: int - Number of values initially made for geometric
      sequence
  """
  def __init__(self, geom_num):
    self.geom_num = geom_num

class linear():
  """
  Linear progression of values with given spacing

  Attributes:
    spacing: int - Difference between adjacent values
  """
  def __init__(self, spacing):
    self.spacing = spacing


class prog():
  """
  Class used for specification and generation of interval progressions.

  Attributes:
    progtype: class - Way in which generated progression of values
                      is to change/increase. May be an instance of
                      flenner, geometric, or linear. Classes may
                      themselves have parameters to specify method of
                      generation.
    max_val: int - Upper limit for generated values. The generated
                   progression cannot exceed this value. Some
                   progression types use this value for calculation.
    neg_vals: bool - Whether to generate negative values as well, in
                     a similar way as positive values.
    min_val: int - Limit for magnitude of negative values. Used in
                   similar way as max_val and is specified as a
                   negative value.
    low_interval: int - Lower limit for values, used for trimming
                        values after generation.
    high_interval: int- Upper limit for values, used for trimming
                        values after generation.
  """
  progtype = None
  max_val = None
  neg_vals = False
  min_val = None

  low_interval = None
  high_interval = None

  def sequence(self, seq_val):
    """
    Generate a positive sequence of values, one side of an array with
    negative values. This is not designed to be called by the user
    except in special cases, with construct() preferred as it uses the
    full set of options.
    """
    if type(self.progtype) == flenner:
      # Construct list of values according to a method of increasing
      # spacing
      magnitude = -1
      frames_beyond_magnitude = seq_val
      while frames_beyond_magnitude >= 50 * 5**(magnitude + 1):
        magnitude += 1
        frames_beyond_magnitude -= 50 * 5**magnitude

      vals_beyond_magnitude = frames_beyond_magnitude // 5**(magnitude + 1)

      n_vals = 1 + (50 * (magnitude + 1)) + vals_beyond_magnitude

      # Allocate values array
      vals = np.empty(n_vals, dtype=np.int64)

      # Efficiently fill the values array
      vals[0] = 0
      last_val = 0
      for i in range(0, magnitude + 1):
        vals[1 + 50 * i : 1 + 50 * (i + 1)] = last_val + np.arange(5**i , 51 * 5**i, 5**i)
        last_val += 50 * 5**i
      vals[1 + 50 * (magnitude + 1) : n_vals] = last_val + np.arange(5**(magnitude + 1), (vals_beyond_magnitude + 1) * 5**(magnitude + 1), 5**(magnitude + 1))

    elif type(self.progtype) == geometric:
      # Largest power of geom_base within seq_val
      geom_base = seq_val**(1.0 / self.progtype.geom_num)

      # Create array of values following geometric progression, with
      # flooring to have values adhere to integer boundaries, removing
      # duplicate numbers, and prepending 0
      vals = np.insert(np.unique(np.floor(np.logspace(0, self.progtype.geom_num, num=(self.progtype.geom_num + 1), base=geom_base)).astype(np.int64)), 0, 0)

    elif type(self.progtype) == linear:
      # Create evenly spaced array of values with specified spacing
      vals = np.arange(0, seq_val + 1, step=self.progtype.spacing)

    else:
      raise RuntimeError("Progression type not recognized")

    return vals

  def construct(self):
    """
    Function for generating a series of values according to specified
    parameters. This is designed to be called by the user.
    """
    # Create main sequence of positive values
    vals = self.sequence(self.max_val)

    # Append negative values if specified
    if self.neg_vals == True:
      vals = np.concatenate((np.flip(-self.sequence(-self.min_val)[1:]), vals))

    # Trim values to specified limits
    if self.low_interval != None:
      vals = vals[vals >= self.low_interval]
    if self.high_interval != None:
      vals = vals[vals <= self.high_interval]

    return vals

  # Determine if option corresponds to this module and process it if so.
  # Returns True if option matched and processed.
  def catch_opt(self, o, a):
    """
    Determine if option corresponds to the progression module and
    process it if so. Returns True if option matched and processed,
    False otherwise.

    Arguments:
      o: str - Name of option to process, from array produced by
               gnu_getopt().
      a: str - Value of option to process, from array produced by
               gnu_getopt().
    """
    if o == "-f":
      self.progtype = flenner()
    elif o == "-g":
      self.progtype = geom(int(a))
    elif o == "-w":
      self.progtype = linear(int(a))
    elif o == "--low-interval":
      self.low_interval = int(a)
    elif o == "--high-interval":
      self.high_interval = int(a)
    elif o == "--negvals":
      self.neg_vals = True
    else:
      # Option not matched
      return False

    return True
