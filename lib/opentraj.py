import numpy as np
import sys
import pydcd

# List of short options processed by this module, used by gnu_getopt()
shortopts = "n:r:"

def usage():
  """
  Print help documentation for options processed by the opentraj
  module.
  """
  print("-n Number of files",
        "-r Number of runs, numbered as folders",
        sep="\n", file=sys.stderr)

class trajset():
  """
  Class for opening arrays of trajectory files.

  Attributes:
    n_files: int - Number of trajectory files in each run
    n_runs: int - Number of runs and corresponding run folders
    dcdfiles: list(pydcd.dcdfile) or list(list(pydcd.dcdfile)) - Set
      of dcdfile objects for trajectory files. May be one or two
      dimensional, depending on if multiple runs are being analyzed.
    fileframes: np.array(dtype=np.int64) - Array of cumulative frames
      in and before each file number. First value is set to 0.
    fparticles: int - Number of particles in the simulation
    timestep: float - Duration of time corresponding to each simulation
      step.

    tbsave: int - Simulation step difference between each saved frame.
      Set if tbsave is constant across files.
    tbsaves: np.array(np.int64) - Simulation step differences between
      each saved frame for each file number. Set if tbsave is not
      constant across files.
  """
  n_files = 1
  n_runs = 0

  def do_opentraj(self, name, n_start, same_tbsave):
    """
    Open a set of trajectory files for a given run and return array and
    attributes. Not meant to be called by the user. Instead, use
    opentraj() or opentraj_multirun().
    """
    # Returned list of dcd file objects
    dcdfiles = list()

    # Number of frames in each file, first element is 0
    fileframes = np.empty(self.n_files + 1, dtype=np.int64)
    fileframes[0] = 0

    if same_tbsave == False:
      tbsaves = np.empty(n, dtype=np.int64)
    elif same_tbsave != True:
      raise RuntimeError("Invalid value of same_tbsave, must be Boolean")

    for i in range(0, self.n_files):
      # The file object can be discarded after converting it to a
      # dcd_file, as the dcd_file duplicates the underlying file
      # descriptor.
      file = open("%s%d.dcd" %(name, n_start + i), "r")
      dcdfiles.append(pydcd.dcdfile(file))
      file.close()

      # Make sure each trajectory file has the same time step and number
      # of particles
      if i == 0:
        # Number of particles in files, may not be same as limited
        # number in analysis
        fparticles = dcdfiles[i].N

        timestep = dcdfiles[i].timestep

        if same_tbsave == True:
          tbsave = dcdfiles[i].tbsave
        else:
          tbsaves[i] = dcdfiles[i].tbsave
      else:
        if dcdfiles[i].N != fparticles:
          raise RuntimeError("Not the same number of particles in each file")
        if dcdfiles[i].timestep != timestep:
          raise RuntimeError("Not the same time step in each file")

        if same_tbsave == True:
          if dcdfiles[i].tbsave != tbsave:
            raise RuntimeError("Not the same frame difference between saves in each file")
        else:
          tbsaves[i] = dcdfiles[i].tbsave

      fileframes[i + 1] = dcdfiles[i].nset

    # Change to hold total index of last frame in each file
    fileframes = np.cumsum(fileframes)

    if same_tbsave == True:
      return dcdfiles, fileframes, fparticles, timestep, tbsave
    else:
      return dcdfiles, fileframes, fparticles, timestep, tbsaves

  def opentraj(self, name, n_start, same_tbsave):
    """
    Open a set of trajectory files, running consistency checks. Sets
    attributes in trajset class according to properties read from
    files.

    Arguments:
      name: str - Prefix of name of each DCD file
      n_start: int - Index on which trajectory file numbering starts
      same_tbsave: bool - If tbsave is the same for each file in series
    """
    if same_tbsave == False:
      self.dcdfiles, self.fileframes, self.fparticles, self.timestep, self.tbsaves = self.do_opentraj(name, n_start, same_tbsave)
    elif same_tbsave == True:
      self.dcdfiles, self.fileframes, self.fparticles, self.timestep, self.tbsave = self.do_opentraj(name, n_start, same_tbsave)
    else:
      raise RuntimeError("Invalid value of same_tbsave, must be Boolean")

  def opentraj_multirun(self, runname, name, n_start, same_tbsave):
    """
    Open a set of trajectory files in multiple runs, running
    consistency checks. Sets attributes in trajset class according to
    properties read from files.

    Arguments:
      runname: str - Prefix of name of each run directory
      name: str - Prefix of name of each DCD file
      n_start: int - Index on which trajectory file numbering starts
      same_tbsave: bool - If tbsave is supposed to be the same for each
        file in series
    """

    # 2D list of files, first dimension across runs, second across
    # files within each run
    self.dcdfiles = list()

    for i in range(0, self.n_runs):
      if i == 0:
        if same_tbsave == True:
          dcdfile, self.fileframes, self.fparticles, self.timestep, self.tbsave = self.do_opentraj("%s%d/%s" %(runname, i + 1, name), n_start, same_tbsave)
        elif same_tbsave == False:
          dcdfile, self.fileframes, self.fparticles, self.timestep, self.tbsaves = self.do_opentraj("%s%d/%s" %(runname, i + 1, name), n_start, same_tbsave)
        else:
          raise RuntimeError("Invalid value of same_tbsave, must be Boolean")

      else:
        if same_tbsave == True:
          dcdfile, r_fileframes, r_fparticles, r_timestep, r_tbsave = self.do_opentraj("%s%d/%s" %(runname, i + 1, name), n_start, same_tbsave)
        else:
          dcdfile, r_fileframes, r_fparticles, r_timestep, r_tbsaves = self.do_opentraj("%s%d/%s" %(runname, i + 1, name), n_start, same_tbsave)

        # Consistency checks across runs

        if (r_fileframes == self.fileframes).all() != True:
          raise RuntimeError("Not the same number of frames in each run for corresponding files")
        if r_fparticles != self.fparticles:
          raise RuntimeError("Not the same number of particles in each file")
        if r_timestep != self.timestep:
          raise RuntimeError("Not the same time step in each file")

        if same_tbsave == True:
          if r_tbsave != self.tbsave:
            raise RuntimeError("Not the same frame difference between saves in each file")
        else:
          if (r_tbsaves == self.tbsaves).all() != True:
            raise RuntimeError("Not the same tbsave in each run for corresponding files")

      # Append to multi-run list of dcd file lists
      self.dcdfiles.append(dcdfile)

  def catch_opt(self, o, a):
    """
    Determine if option corresponds to the opentraj module and process
    it if so. Returns True if option matched and processed, False
    otherwise.

    Arguments:
      o: str - Name of option to process, from array produced by
               gnu_getopt().
      a: str - Value of option to process, from array produced by
               gnu_getopt().
    """
    if o == "-n":
      self.n_files = int(a)
    elif o == "-r":
      self.n_runs = int(a)
    else:
      # Option not matched
      return False

    return True
