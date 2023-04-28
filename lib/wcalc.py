import numpy as np
import sys
import enum

# List of short options processed by this module, used by gnu_getopt()
shortopts = "t:u:e:"

def usage():
  """
  Print help documentation for options processed by the wcalc module.
  """
  print("w function types (last specified is used, must be specified):",
        "-t Theta function threshold (argument is threshold radius)",
        "-u Double negative exponential/Gaussian (argument is exponential length)",
        "-e Single negative exponential (argument is exponential length)",
        sep="\n", file=sys.stderr)

class wtypes(enum.Enum):
  """
  Enumeration of types of w functions that can be used.

  Values:
    theta - Simple overlap function with given radius
    gauss - Double negative exponential/Gaussian with given width
    exp - Single negative exponential with given width
  """
  theta = 1
  gauss = 2
  exp = 3

class wcalc():
  """
  Class for calculation of w values for sets of particle positions.

  Attributes:
    frames: frames - Frames object from which this wcalc object gets
                     particle data.
    wtype: wtypes - Type of w function to use
  """
  wtype = None

  def __init__(self, frames):
    """
    Initialize an instance of class wcalc

    Arguments:
      frames: frames - The frames object that wcalc will use to get
                       particle data
    """
    self.frames = frames

  # Get end frame values and calculate w, do not modify start frame
  # values
  def calculate_w_half(self, w, x0, y0, z0, t1, x1, y1, z1, run):
    """
    Get end frame values and calculate w, without modifying start frame
    values.

    Arguments:
      x0, y0, z0: np.array(dtype=np.single) - Arrays of x, y, and z
          positions for start frame, already filled. Size must be
          number of particles.
      t1: int - Frame number to use for end frame
      x1, y1, z1: np.array(dtype=np.single) - Arrays of x, y, and z
          positions for end frame, to be filled. Size must be number
          of particles.
    """
    self.frames.get_frame(t1, x1, y1, z1, run)

    # Calculate w function for each particle
    if self.wtype == wtypes.theta:
      np.less((x1 - x0)**2 +
              (y1 - y0)**2 +
              (z1 - z0)**2, self.radius**2, out=w).astype(np.int8, copy=False)
    elif self.wtype == wtypes.gauss:
      np.exp(-((x1 - x0)**2 +
               (y1 - y0)**2 +
               (z1 - z0)**2)/(2 * self.gscale**2), out=w)
    elif self.wtype == wtypes.exp:
      np.exp(-np.sqrt((x1 - x0)**2 +
                      (y1 - y0)**2 +
                      (z1 - z0)**2)/self.sscale, out=w)

  def calculate_w(self, w, t0, x0, y0, z0, t1, x1, y1, z1, run):
    """
    Get start and end frame values and calculate w.

    Arguments:
      t0: int - Frame number to use for start frame
      x0, y0, z0: np.array(dtype=np.single) - Array of x, y, and z
          positions for start frame, to be filled
      t1: int - Frame number to use for end frame
      x1, y1, z1: np.array(dtype=np.single) - Array of x positions for
          end frame, to be filled
    """
    # Get start frame values
    self.frames.get_frame(t0, x0, y0, z0, run)

    # Get end frame values and calculate w
    self.calculate_w_half(w, x0, y0, z0, t1, x1, y1, z1, run)

  def catch_opt(self, o, a):
    """
    Determine if option corresponds to the wcalc module and process it
    if so. Returns True if option matched and processed, False
    otherwise.

    Arguments:
      o: str - Name of option to process, from array produced by
               gnu_getopt().
      a: str - Value of option to process, from array produced by
               gnu_getopt().
    """
    if o == "-t":
      self.wtype = wtypes.theta
      self.radius = float(a)
    elif o == "-u":
      self.wtype = wtypes.gauss
      self.sscale = float(a)
    elif o == "-e":
      self.wtype = wtypes.exp
      self.gscale = float(a)
    else:
      # Option not matched
      return False

    return True

  def verify(self):
    """
    Verify that arguments for w calculation have been specified
    sufficiently for further operations. Should be called after
    argument processing.
    """
    if self.wtype == None:
      raise RuntimeError("No w function type specified")
