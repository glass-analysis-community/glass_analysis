import numpy as np
import sys
import pydcd

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
    n_atoms: int - Number of atoms in each molecule for polyatomic
      trajectories, None if trajectory not polyatomic
    atom_masses: np.array(dtype=float) - Array of proportional masses
      of each atom in each molecule, normalized before calculation to
      sum to 1
    n_frames: int - Number of frames included in analysis
    shortopts: str - List of short options processed by this module,
      used by gnu_getopt()
    longopts: str - List of long options processed by this module, used
      by gnu_getopt()
    argtext: str - Description of arguments processed by this module
  """
  limit_particles = False
  upper_limit = None
  lower_limit = None
  start = 0
  final = None
  n_atoms = None
  atom_masses = None

  shortopts = "s:m:o:p:"
  longopts = ["polyatomic=", "polyatomic-masses="]

  argtext = "-s Frame number to start on (index starts at 1)\n" \
            + "-m Last frame number in range to use for analysis, either final or initial times (index starts at 1)\n" \
            + "-o Start index (from 1) of particles to limit analysis to\n" \
            + "-p End index (from 1) of particles to limit analysis to\n" \
            + "--polyatomic Trajectory file contains sets of atom coordinates from molecules, argument is number of atoms in molecule\n" \
            + "--polyatomic-masses Proportional mass of each atom in each molecule, argument is comma-separated list (default: equal masses)"

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

  def alloc_arrays(self):
    """
    Allocate arrays for use later in reading of frames. Meant to be
    called internally rather than by user.
    """
    # If particle limiting required or polyatomic molecules used,
    # allocate intermediate arrays for particle reading
    if self.limit_particles == True or self.n_atoms != None:
      self.x = np.empty(self.fparticles, dtype=np.single)
      self.y = np.empty(self.fparticles, dtype=np.single)
      self.z = np.empty(self.fparticles, dtype=np.single)

    # Array for center of mass of each frame
    self.cm = np.empty((self.n_runs, self.n_frames, 3), dtype=np.float64)

    # Bitmap of values indicating whether the center of mass for a
    # given frame has been calculated and stored in self.cm
    self.cm_def = np.zeros((self.n_runs, (self.n_frames + 7) // 8), dtype=np.uint8)

    # Arrays of particles on which analysis takes place are
    # not allocated here, instead being passed to functions, so that
    # the user may make decisions about their allocation and re-use.

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

    self.particles = self.fparticles

    if self.n_atoms != None:
      if self.n_atoms < 2:
        raise RuntimeError("Polyatomic trajectories must have 2 or more atoms per molecule")

      if self.particles % self.n_atoms != 0:
        raise RuntimeError("Polyatomic trajectories must have number of particles divisible by number of particles in each molecule")
      self.particles //= self.n_atoms

    if self.atom_masses is not None:
      if self.n_atoms == None:
        raise RuntimeError("Specified atom masses without specifying polyatomic mode")

      if len(self.atom_masses) != self.n_atoms:
        raise RuntimeError("Number of specified atom masses different from number of atoms per molecule")

      # Normalize atom masses to sum to 1 for efficiency of later
      # calculations
      self.atom_masses /= np.sum(self.atom_masses)

    if self.lower_limit == None:
      self.lower_limit = 0
    elif self.lower_limit != 0:
      self.limit_particles = True

    if self.upper_limit == None:
      self.upper_limit = self.fparticles
    elif self.upper_limit < self.particles:
      self.limit_particles = True

    if self.limit_particles == True:
      self.particles = self.upper_limit - self.lower_limit

    # End of set of frames to used for both final and initial times
    if self.final == None:
      self.final = self.fileframes[-1]
    else:
      if self.final > self.fileframes[-1]:
        raise RuntimeError("End limit time frame beyond set of frames")

    self.n_frames = self.final - self.start

    self.alloc_arrays()

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
    if self.n_atoms != None:
      self.dcdfiles[run][which_file].gdcdp(self.x, self.y, self.z, offset)
      x = self.x.reshape((self.fparticles//self.n_atoms, self.n_atoms))[self.lower_limit:self.upper_limit,:]
      y = self.y.reshape((self.fparticles//self.n_atoms, self.n_atoms))[self.lower_limit:self.upper_limit,:]
      z = self.z.reshape((self.fparticles//self.n_atoms, self.n_atoms))[self.lower_limit:self.upper_limit,:]
      if self.atom_masses is None:
        x0[:] = np.mean(x, axis=1)
        y0[:] = np.mean(y, axis=1)
        z0[:] = np.mean(z, axis=1)
      else:
        x0[:] = np.sum(self.atom_masses * x, axis=1)
        y0[:] = np.sum(self.atom_masses * y, axis=1)
        z0[:] = np.sum(self.atom_masses * z, axis=1)
    elif self.limit_particles == True:
      self.dcdfiles[run][which_file].gdcdp(self.x, self.y, self.z, offset)
      x0[:] = self.x[self.lower_limit:self.upper_limit]
      y0[:] = self.y[self.lower_limit:self.upper_limit]
      z0[:] = self.z[self.lower_limit:self.upper_limit]
    else:
      self.dcdfiles[run][which_file].gdcdp(x0, y0, z0, offset)

    # If center of mass has not been calculated for given frame,
    # calculate frame center of mass
    if (self.cm_def[run][t0 // 8] >> (t0 % 8)) & 0x1 == 0:
      self.cm[run][t0][0] = np.mean(x0)
      self.cm[run][t0][1] = np.mean(y0)
      self.cm[run][t0][2] = np.mean(z0)

      # Set bit corresponding to frame to 1
      self.cm_def[run][t0 // 8] |= 0x1 << (t0 % 8)

    # Correct for center of mass
    x0 -= self.cm[run][t0][0]
    y0 -= self.cm[run][t0][1]
    z0 -= self.cm[run][t0][2]

  def get_orientations(self, t0, x0, y0, z0, run):
    """
    Read values for molecule orientations for a given frame and run
    from DCD file into given set of arrays. Direction coordinates are
    normalized to unit vectors.

    Arguments:
      t0: int - Frame number within run to read from
      x0, y0, z0: np.array(dtype=np.single) - Arrays for x, y, and z
        components of orientation vectors to be filled. The size of the
        arrays must be the number of particles in the files.
      run: int - Number of run to read from
    """
    if self.n_atoms == None:
      raise RuntimeError("Orientations cannot be found for non-polyatomic trajectories")

    # Get differences between particle positions in each molecule.
    # In this implementation, just the relative position of the first
    # and second molecules are used to calculate the orientation. In
    # the future, the positions of the other atoms may be used to find
    # a more accurate orientation. For now, this approach is used,
    # which avoids the creation of 0-length orientation vectors.
    which_file, offset = self.lookup_frame(t0)
    self.dcdfiles[run][which_file].gdcdp(self.x, self.y, self.z, offset)
    x = self.x.reshape((self.fparticles//self.n_atoms, self.n_atoms))[self.lower_limit:self.upper_limit,:]
    y = self.y.reshape((self.fparticles//self.n_atoms, self.n_atoms))[self.lower_limit:self.upper_limit,:]
    z = self.z.reshape((self.fparticles//self.n_atoms, self.n_atoms))[self.lower_limit:self.upper_limit,:]
    x0[:] = x[:,1] - x[:,0]
    y0[:] = y[:,1] - y[:,0]
    z0[:] = z[:,1] - z[:,0]

    # Normalize vectors by length, converting to unit vectors
    hyp = np.linalg.norm((x0, y0, z0), axis=0)
    x0 /= hyp
    y0 /= hyp
    z0 /= hyp

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
    Shift the start of the frame set by a given amount. Should be
    called after prepare().

    Arguments:
      shift_index: int - Number of frames to shift by
    """
    self.start += shift_index

    # New number of frames
    self.n_frames = self.final - self.start

    # Reallocate arrays for new number of frames
    self.alloc_arrays()

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
    elif o == "--polyatomic":
      self.n_atoms = int(a)
    elif o == "--polyatomic-masses":
      self.atom_masses = np.array(list(map(float, a.split(","))))
    else:
      # Option not matched
      return False

    return True
