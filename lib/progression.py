import numpy as np
import sys

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
    high_interval: int - Upper limit for values, used for trimming
                         values after generation.
    adj_seq: np.array() - Array of values to which progression values
                          are to be adjusted to the closest value in on
                          a linear or logarithmic scale. Must be
                          pre-sorted.
    adj_log: bool - If true, values are to be adjusted to the closest
                    value of adj_seq on a logarithmic scale. Otherwise,
                    values are adjusted to closest value on a linear
                    scale.
    shortopts: str - List of short options processed by this module,
                     used by gnu_getopt()
    longopts: str - List of long options processed by this module, used
                    by gnu_getopt()
  """
  progtype = None
  max_val = None
  neg_vals = False
  min_val = None

  low_interval = None
  high_interval = None

  adj_seq = None
  adj_log = False

  shortopts = "fg:w:"
  longopts = ["low-interval", "high-interval", "negvals"]

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
      # prepending of 0
      vals = np.insert(np.logspace(0, self.progtype.geom_num, num=(self.progtype.geom_num + 1), base=geom_base), 0, 0)

    elif type(self.progtype) == linear:
      # Create evenly spaced array of values with specified spacing
      vals = np.arange(0, seq_val + 1, step=self.progtype.spacing)

    elif self.progtype == None:
      raise RuntimeError("Must specify interval increase progression type")

    else:
      raise RuntimeError("Progression type not recognized")

    return vals

  def construct(self):
    """
    Function for generating a series of values according to specified
    parameters. This is designed to be called by the user. If adj_seq
    has been specified, return list of indices in adj_seq rather than
    the values themselves.
    """
    # Create main sequence of positive values
    vals = self.sequence(int(self.max_val))

    # Append negative values if specified
    if self.neg_vals == True:
      vals = np.concatenate((np.flip(-self.sequence(-int(self.min_val))[1:]), vals))

    # If a set of numbers to adjust values closest to in linear or
    # logarithmic scales is given, adjust values. Otherwise, adjust to
    # fit on integer boundaries.
    if self.adj_seq is not None:
      if self.adj_log == False:
        # Find index of closest value in adj_seq less than or equal to
        # vals
        adj_leftidx = np.searchsorted(self.adj_seq, vals, side='left') - 1

        # Keep index or adjust to next, based on which difference is
        # greater. For smallest adj_seq value, since low value for
        # comparison cannot be provided, trim index to remain within
        # array boundaries, and trim output to ensure valid adj_idx.
        adj_idx = np.clip(adj_leftidx + np.greater((vals - self.adj_seq[np.maximum(adj_rightidx, 0)]),
                                                   (self.adj_seq[adj_rightidx + 1] - vals)).astype(np.int64), 0, len(adj_seq_pos) - 1)
      else:
        # Separate adj_seq and vals arrays into positive, negative, and
        # zero components for comparison of logarithms
        adj_seq_neg = self.adj_seq[self.adj_seq < 0.0]
        adj_seq_zero = self.adj_seq[self.adj_seq == 0.0]
        adj_seq_pos = self.adj_seq[self.adj_seq > 0.0]
        vals_neg = vals[vals < 0.0]
        vals_zero = vals[vals == 0.0]
        vals_pos = vals[vals > 0.0]

        # Find index of logarithmically closest value in adj_seq less
        # than or equal to vals
        adj_leftidx_neg = np.searchsorted(np.log(-adj_seq_neg), np.log(vals_neg), side='left') - 1
        if len(adj_seq_zero) != 0:
          adj_idx_zero = np.array((0, ))
        else:
          adj_idx_zero = np.array(())
        adj_leftidx_pos = np.searchsorted(np.log(adj_seq_pos), np.log(vals_pos), side='left') - 1

        # Keep index or adjust to next, based on which difference is
        # greater. For largest and smallest adj_seq values, trim input
        # indices to stay within bounds of adj_seq array. Clip output
        # indices, which may be outside of bounds due to input
        # trimming.
        adj_idx_pos = np.clip(adj_leftidx_pos + np.greater((np.log(vals_pos) - np.log(adj_seq_pos[np.maximum(adj_leftidx_pos, 0)])),
                                                           (np.log(adj_seq_pos[np.minimum(adj_leftidx_pos + 1, len(adj_seq_pos) - 1)]) - np.log(vals_pos))).astype(np.int64), 0, len(adj_seq_pos) - 1)
        adj_idx_neg = np.clip(adj_leftidx_neg + np.greater((np.log(-vals_neg) - np.log(-adj_seq_neg[np.maximum(adj_leftidx_neg, 0)])),
                                                           (np.log(-adj_seq_neg[np.minimum(adj_leftidx_neg + 1, len(adj_seq_neg) - 1)]) - np.log(-vals_neg))).astype(np.int64), 0, len(adj_seq_pos) - 1)

        # Convert index values in vals back to indices to adj_seq
        adj_idx_zero += len(adj_seq_neg)
        adj_idx_pos += len(adj_seq_neg) + len(adj_seq_zero)

        # Create combined adj_idx array
        adj_idx = np.concatenate((adj_idx_neg, adj_idx_zero, adj_idx_pos))

      # Trim adj_seq indices to those that correspond to values within
      # specified limits
      if self.low_interval != None:
        adj_idx = adj_idx[self.adj_seq[adj_idx] >= self.low_interval]
      if self.high_interval != None:
        adj_idx = adj_idx[self.adj_seq[adj_idx] <= self.high_interval]

      return np.unique(adj_idx)
    else:
      # Round values down to lowest-magnitude integer and eliminate
      # duplicate values
      vals = np.unique(np.fix(vals).astype(np.int64))

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
      self.progtype = geometric(int(a))
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

  def usage(self):
    """
    Print help documentation for options processed by the progression
    module.
    """
    print("Interval increase progression (last specified is used):",
          "-f Flenner-style periodic-exponential-increasing increment (iterations: 50, power: 5)",
          "-g Geometric spacing progression, selectively dropped to fit on integer frames (argument is number of interval length values)",
          "-w Linear progression (argument is spacing between interval length values)",
          "Interval progression modifiers:",
          "--low-interval Trim set of interval lengths below frame number value",
          "--high-interval Trim set of interval lengths above frame number value",
          "--negvals Add mirrored sequence of negative length intervals",
          sep="\n", file=sys.stderr)
