import numpy as np
import sys
import pydcd

def usage():
  """
  Print help documentation for options processed by the frame module.
  """
  print("-s Frame number to start on (index starts at 1)",
        "-m Last frame number in range to use for analysis, either final or initial times (index starts at 1)",
        "-o Start index (from 1) of particles to limit analysis to",
        "-p End index (from 1) of particles to limit analysis to",
        sep="\n", file=sys.stderr)

class frames():
  """
  Class for reading particle data from trajectory file arrays.

  Attributes:
    limit_particles: bool - Specifies if set of analyzed particles is
      limited relative to number of particles in trajectory files
    upper_limit: int - First 0-indexed particle index to not include
      in analysis.
    lower_limit: int - First 0-indexed particle index to include in
      analysis.
    fparticles: int - Total number of particles in trajectory files.
    particles: int - Number of particles in analysis. May be limited
      from total number of particles in trajectory files.
    start: int - 0-indexed index of first frame to use for analysis
    final: int - 1 greater than 0-indexed index of last frame to use
      for analysis
    n_frames: int - Number of frames included in analysis
    shortopts: str - List of short options processed by this module,
      used by gnu_getopt()
  """
  limit_particles = False
  upper_limit = None
  lower_limit = None
  start = 0
  final = None
  shortopts = "s:m:o:p:"

  def __init__(self, trajset):
    """
    Create frames object.

    Arguments:
      trajset: class trajset or list(class trajset) - Trajectory set
        from which frame data is to be read. If a list, then is a
        sequence of trajectory file objects specified in order of
        increasing simulation time.
    """
    self.trajset = trajset

  def prepare(self):
    """
    Set internal attributes based on results of argument processing and
    prepare for data reading. Must be run after argument processing and
    before data reading.
    """
    if type(self.trajset) == list:
      # Remove trajectory sets with 0 files
      self.trajset = [i for i in self.trajset if i.n_files > 0]

      # Check if all trajectory sets have same number of runs
      n_runs_set = {i.runset.n_runs for i in self.trajset}
      if len(n_runs_set) != 1:
        raise RuntimeError("All trajectory sets for frame object must have same number of runs")
      self.n_runs = self.trajset[0].runset.n_runs

      # Check if all trajectory sets have same number of particles in
      # files
      fparticles_set = {i.fparticles for i in self.trajset}
      if len(fparticles_set) != 1:
        raise RuntimeError("All trajectory sets for frame object must have same number of particles")
      self.fparticles = self.trajset[0].fparticles

      # Concatenate dcdfiles lists
      self.dcdfiles = np.concatenate([i.dcdfiles for i in self.trajset], axis=1)

      # Concatenate fileframes arrays, accounting for 0 first elements
      self.fileframes = np.insert(np.cumsum(np.concatenate([np.diff(i.fileframes) for i in self.trajset])), 0, 0)

    else:
      self.n_runs = self.trajset.runset.n_runs
      self.fparticles = self.trajset.fparticles
      self.dcdfiles = self.trajset.dcdfiles
      self.fileframes = self.trajset.fileframes

    if self.limit_particles == False:
      self.particles = self.fparticles
    else:
      if self.lower_limit == None:
        self.lower_limit = 0
      if self.upper_limit == None:
        self.upper_limit = self.fparticles

      if self.lower_limit != 0 or self.upper_limit < self.fparticles:
        self.particles = self.upper_limit - self.lower_limit
        self.limit_particles = True
      else:
        self.particles = self.fparticles
        self.limit_particles = False

    # End of set of frames to used for both final and initial times
    if self.final == None:
      self.final = self.fileframes[-1]
    else:
      if self.final > self.fileframes[-1]:
        raise RuntimeError("End limit time frame beyond set of frames")

    self.n_frames = self.final - self.start

    # If particle limiting required, allocate intermediate arrays for
    # particle reading
    if self.limit_particles == True:
      self.x = np.empty(self.fparticles, dtype=np.single)
      self.y = np.empty(self.fparticles, dtype=np.single)
      self.z = np.empty(self.fparticles, dtype=np.single)

    # Arrays of particles on which analysis takes place are
    # not allocated here, instead being passed to functions, so that
    # the user may make decisions about their allocation and re-use.

  def lookup_frame(self, t):
    """
    Find file number and offset within file corresponding to frame
    number.

    Arguments:
      t: int - Frame index for lookup
    """
    which_file = np.searchsorted(self.fileframes, self.start + t, side="right") - 1
    offset = self.start + t - self.fileframes[which_file]

    return which_file, offset

  def generate_cm(self):
    """
    Generate internal array of centers of mass for each frame. This
    must be performed before later data reading functions.
    """

    if self.limit_particles == False:
      # Allocate intermediate particle reading arrays
      x0 = np.empty(self.particles, dtype=np.single)
      y0 = np.empty(self.particles, dtype=np.single)
      z0 = np.empty(self.particles, dtype=np.single)

    # Allocate array for center of mass of each frame
    self.cm = [np.empty((self.n_frames, 3), dtype=np.float64)] * self.n_runs

    for i in range(0, self.n_runs):
      for j in range(0, self.n_frames):
        which_file, offset = self.lookup_frame(j)
        if self.limit_particles == True:
          self.dcdfiles[i][which_file].gdcdp(self.x, self.y, self.z, offset)
          self.cm[i][j][0] = np.mean(self.x[self.lower_limit:self.upper_limit])
          self.cm[i][j][1] = np.mean(self.y[self.lower_limit:self.upper_limit])
          self.cm[i][j][2] = np.mean(self.z[self.lower_limit:self.upper_limit])
        else:
          self.dcdfiles[i][which_file].gdcdp(x0, y0, z0, offset)
          self.cm[i][j][0] = np.mean(x0)
          self.cm[i][j][1] = np.mean(y0)
          self.cm[i][j][2] = np.mean(z0)

  def get_frame(self, t0, x0, y0, z0, run):
    """
    Read values for a given frame and run from DCD file into given set
    of arrays, correcting for frame center of mass.

    Arguments:
      t0: int - Frame number within run to read from
      x0, y0, z0: np.array(dtype=np.single) - Arrays for x, y, and z
        coordinates of particles to be filled. The size of the arrays
        must be the number of particles in the files and the arrays must
        be of C-contiguous order.
      run: int - Number of run to read from
    """
    which_file, offset = self.lookup_frame(t0)
    if self.limit_particles == True:
      self.dcdfiles[run][which_file].gdcdp(self.x, self.y, self.z, offset)
      x0[:] = self.x[self.lower_limit:self.upper_limit]
      y0[:] = self.y[self.lower_limit:self.upper_limit]
      z0[:] = self.z[self.lower_limit:self.upper_limit]
    else:
      self.dcdfiles[run][which_file].gdcdp(x0, y0, z0, offset)

    # Correct for center of mass
    x0 -= self.cm[run][t0][0]
    y0 -= self.cm[run][t0][1]
    z0 -= self.cm[run][t0][2]

  def frame_time(self, t0):
    """
    Find real time corresponding to frame index.

    Arguments:
      t0: int - Frame index to find real time of
    """
    which_file, offset = self.lookup_frame(t0)
    dcdfile = self.dcdfiles[0][which_file]

    return dcdfile.itstart + offset * dcdfile.timestep * dcdfile.tbsave

  def shift_start(self, shift_index):
    """
    Shift the start of the frame set by a given amount

    Arguments:
      shift_index: int - Number of frames to shift by
    """

    self.start += shift_index
    self.n_frames -= shift_index

  def catch_opt(self, o, a):
    """
    Determine if option corresponds to the frame module and process it
    if so. Returns True if option matched and processed, False
    otherwise.

    Arguments:
      o: str - Name of option to process, from array produced by
               gnu_getopt().
      a: str - Value of option to process, from array produced by
               gnu_getopt().
    """
    if o == "-s":
      self.start = int(a) - 1
    elif o == "-m":
      self.final = int(a)
    elif o == "-o":
      self.lower_limit = int(a) - 1
    elif o == "-p":
      self.upper_limit = int(a)
    else:
      # Option not matched
      return False

    return True

  def usage(self):
    """
    Print help documentation for options processed by the frame module.
    """
    print("-s Frame number to start on (index starts at 1)",
          "-m Last frame number in range to use for analysis, either final or initial times (index starts at 1)",
          "-o Start index (from 1) of particles to limit analysis to",
          "-p End index (from 1) of particles to limit analysis to",
          sep="\n", file=sys.stderr)
