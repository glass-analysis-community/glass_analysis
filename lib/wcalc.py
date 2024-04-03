import numpy as np
import sys
import enum

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
  dir_theta = 4

class wcalc():
  """
  Class for calculation of w values for sets of particle positions.

  Attributes:
    frames: frames - Frames object from which this wcalc object gets
                     particle data.
    wtype: wtypes - Type of w function to use
    shortopts: str - List of short options processed by this module,
                     used by gnu_getopt()
    argtext: str - Description of arguments processed by this module
  """
  wtype = None

  shortopts = "t:u:e:"
  longopts = ["dir-theta="]

  argtext = "w function types (last specified is used, must be specified):\n" \
            + "  -t Theta function threshold (argument is threshold radius)\n" \
            + "  -u Double negative exponential/Gaussian (argument is exponential length)\n" \
            + "  -e Single negative exponential (argument is exponential length)\n" \
            + "  --dir-theta Orientaional theta function (argument is threshold of cosine value)"

  def __init__(self, frames):
    """
    Initialize an instance of class wcalc

    Arguments:
      frames: frames - The frames object that wcalc will use to get
                       particle data
    """
    self.frames = frames

  def prepare(self):
    """
    Verify that arguments for w calculation have been specified
    sufficiently for further operations. Should be called after
    argument processing.
    """
    if self.wtype == None:
      raise RuntimeError("No w function type specified")

    if self.wtype == wtypes.dir_theta and self.frames.n_atoms == None:
      raise RuntimeError("Orientational threshold w function type requires polyatomic trajectories")

  def calculate_w_half(self, w, x0, y0, z0, t1, x1, y1, z1, run):
    """
    Get end frame values and calculate w, without modifying start frame
    values.

    Arguments:
      x0, y0, z0: np.array(dtype=np.single) - Arrays of x, y, and z
          positions for start frame, already filled. Size must be
          number of particles. If using orientational w function, these
          are orientations. Otherwise, they are positions.
      t1: int - Frame number to use for end frame
      x1, y1, z1: np.array(dtype=np.single) - Arrays of x, y, and z
          positions for end frame, to be filled. Size must be number
          of particles. If using orientational w function, these are
          orientations. Otherwise, they are positions.
      run: int - Index of run to read from
    """
    if self.wtype == wtypes.dir_theta:
      self.frames.get_orientations(t1, x1, y1, z1, run)
    else:
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
    elif self.wtype == wtypes.dir_theta:
      np.greater_equal(x1 * x0 +
                       y1 * y0 +
                       z1 * z0, self.cos_thetab, out=w).astype(np.int8, copy=False)

  def get_w_half(self, t0, x0, y0, z0, run):
    """
    Get start frame values for first frame in w function calculation.
    Results may be reused for multiple calculate_w_half invocations.

    Arguments:
      t0: int - Frame number to use for start frame
      x0, y0, z0: np.array(dtype=np.single) - Array of x, y, and z
          coordinate values for start frame, to be filled. Size must be
          number of particles. If using orientational w function, these
          are orientations. Otherwise, they are positions.
      run: int - Index of run to read from
    """
    if self.wtype == wtypes.dir_theta:
      self.frames.get_orientations(t0, x0, y0, z0, run)
    else:
      self.frames.get_frame(t0, x0, y0, z0, run)

  def calculate_w(self, w, t0, x0, y0, z0, t1, x1, y1, z1, run):
    """
    Get start and end frame values and calculate w.

    Arguments:
      t0: int - Frame number to use for start frame
      x0, y0, z0: np.array(dtype=np.single) - Array of x, y, and z
        coordinate values for start frame, to be filled
      t1: int - Frame number to use for end frame
      x1, y1, z1: np.array(dtype=np.single) - Array of x coordinate
        values for end frame, to be filled
      run: int - Index of run to read from
    """
    # Get start frame values
    self.frames.get_w_half(t0, x0, y0, z0, run)

    # Get end frame values and calculate w
    self.calculate_w_half(w, x0, y0, z0, t1, x1, y1, z1, run)

  def print_info(self, outfile=sys.stdout):
    """
    Print information about w function used for calculation in format
    of output files

    Arguments:
      outfile: io.TextIOWrapper - File to print to
    """
    if self.wtype == wtypes.theta:
      outfile.write("#w function type: Threshold\n")
      outfile.write("#a = %f\n" %self.radius)
    elif self.wtype == wtypes.gauss:
      outfile.write("#w function type: Gaussian\n")
      outfile.write("#a = %f\n" %self.gscale)
    elif self.wtype == wtypes.exp:
      outfile.write("#w function type: Single Exponential\n")
      outfile.write("#a = %f\n" %self.sscale)
    elif self.wtype == wtypes.dir_theta:
      outfile.write("#w function type: Orientational Threshold\n")
      outfile.write("#b = %f\n" %self.cos_thetab)

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
      self.gscale = float(a)
    elif o == "-e":
      self.wtype = wtypes.exp
      self.sscale = float(a)
    elif o == "--dir-theta":
      self.wtype = wtypes.dir_theta
      self.cos_thetab = float(a)
    else:
      # Option not matched
      return False

    return True
