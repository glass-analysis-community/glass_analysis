import numpy as np
import sys
import pydcd

global_options_printed = False
one_open = False

class dcdfile_w:
  """
  Wrapper for dcdfile class of pydcd library

  Attributes:
    dcdfile: dcdfile - Underlying dcdfile object from pydcd.
    filename: str - Name of path to DCD file, only used when opening
      files one at a time. Value of None if opening all files at once
      (default).
    (other attributes mirror dcdfile attributes)
  """

  def __init__(self, dcdfile, filename=None):
    """
    Create dcdfile_w object

    Arguments:
      dcdfile: dcdfile - dcdfile to wrap
      filename: str - Pathname to file, only specified if opening one
        at a time.
    """
    self.dcdfile = dcdfile
    self.filename = filename

    self.N = self.dcdfile.N
    self.nset = self.dcdfile.nset
    self.itstart = self.dcdfile.itstart
    self.tbsave = self.dcdfile.tbsave
    self.timestep = self.dcdfile.timestep
    self.wcell = self.dcdfile.wcell

    # If opening DCD files one at a time, close DCD file
    if filename != None:
      del self.dcdfile

  def prepare_from_filename(self):
    # The file object can be discarded after converting it to a
    # dcd_file, as the dcd_file duplicates the underlying file
    # descriptor.
    file = open(self.filename, "r")
    self.dcdfile = pydcd.dcdfile(file)
    file.close()

  def gdcdp(self, x_array, y_array, z_array, pos):
    if self.filename != None:
      self.prepare_from_filename()

    self.dcdfile.gdcdp(x_array, y_array, z_array, pos)

    if self.filename != None:
      del self.dcdfile

  def sdcdp(self, x_array, y_array, z_array, pos):
    if self.filename != None:
      self.prepare_from_filename()

    self.dcdfile.sdcdp(x_array, y_array, z_array, pos)

    if self.filename != None:
      del self.dcdfile

  def adcdp(self, x_array, y_array, z_array):
    if self.filename != None:
      self.prepare_from_filename()

    self.dcdfile.adcdp(x_array, y_array, z_array)

    if self.filename != None:
      del self.dcdfile

class runset():
  """
  Attributes:
    shortopts: str - List of short options processed by this module,
      used by gnu_getopt()
    n_runs: int - Number of runs and corresponding run folders.
    rundirs: bool -  If run folders are being used. Otherwise, script
      is being run in trajectory file directory.
  """
  n_runs = 1
  rundirs = False

  def __init__(self, runname="run", opt="r"):
    """
    Create runset object

    Argument:
      opt: str - Letter of option that this run set object will catch
        to determine number of run files (e.g. "r")
      name: str - Prefix of name of each run folder
    """
    self.opt = opt
    self.runname = runname
    self.shortopts = opt + ":"

  def catch_opt(self, o, a):
    """
    Determine if option corresponds to run parameters in the opentraj
    module and process it if so. Returns True if option matched and
    processed, False otherwise.

    Arguments:
      o: str - Name of option to process, from array produced by
               gnu_getopt().
      a: str - Value of option to process, from array produced by
               gnu_getopt().
    """
    if o == "-r":
      self.n_runs = int(a)
      self.rundirs = True
    else:
      # Option not matched
      return False

    return True

  def usage(self):
    """
    Print help documentation for options processed by the runset class.
    """
    print("-r Number of runs, numbered as folders", file=sys.stderr)

class trajset():
  """
  Class for opening arrays of trajectory files.

  Attributes:
    n_files: int - Number of trajectory files in each run
    dcdfiles: list(pydcd.dcdfile) or list(list(pydcd.dcdfile)) - Set
      of dcdfile objects for trajectory files. May be one or two
      dimensional, depending on if multiple runs are being analyzed.
    fileframes: np.array(dtype=np.int64) - Array of cumulative frames
      in and before each file number. First value is set to 0.
    fparticles: int - Number of particles in the simulation
    timestep: float - Duration of time corresponding to each simulation
      step.
    opt: str - Option on command line that specifies number of files
      for this trajectory set.
    shortopts: str - List of short options processed by this module,
      used by gnu_getopt()
    longopts: str - List of long options processed by this module, used
      by gnu_getopt()

    tbsave: int - Simulation step difference between each saved frame.
      Set if tbsave is constant across files.
    tbsaves: np.array(np.int64) - Simulation step differences between
      each saved frame for each file number. Set if tbsave is not
      constant across files.
  """
  longopts = ["one-open"]

  def __init__(self, runset, opt="n", name="traj", default_n_files=1):
    """
    Create trajset object

    Argument:
      runset: class runset - Run set object associated with this
        trajectory set
      opt: str - Letter of option that this trajectory file set object
        will catch to determine number of trajectory files (e.g. "n")
      name: str - Prefix of name of each DCD file in this trajectory
        set
      default_n_files: int - Default number of trajectory files in set
    """
    self.runset = runset
    self.opt = opt
    self.name = name
    self.shortopts = opt + ":"
    self.n_files = default_n_files

  def do_opentraj(self, path, n_start, same_tbsave):
    """
    Open a set of trajectory files for a given run and return array and
    attributes. Not meant to be called by the user. Instead, use
    opentraj() or opentraj_multirun().
    """
    global one_open

    # Returned list of dcd file objects
    dcdfiles = list()

    # Number of frames in each file, first element is 0
    fileframes = np.empty(self.n_files + 1, dtype=np.int64)
    fileframes[0] = 0

    if same_tbsave == False:
      tbsaves = np.empty(self.n_files, dtype=np.int64)
    elif same_tbsave != True:
      raise RuntimeError("Invalid value of same_tbsave, must be Boolean")

    for i in range(0, self.n_files):
      # The file object can be discarded after converting it to a
      # dcd_file, as the dcd_file duplicates the underlying file
      # descriptor.
      file = open("%s%d.dcd" %(path, n_start + i), "r")
      if one_open == True:
        dcdfiles.append(dcdfile_w(pydcd.dcdfile(file), filename="%s%d.dcd" %(path, n_start + i)))
      else:
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

  def opentraj(self, n_start, same_tbsave):
    """
    Open a set of trajectory files, running consistency checks. Sets
    attributes in trajset class according to properties read from
    files.

    Arguments:
      n_start: int - Index on which trajectory file numbering starts
      same_tbsave: bool - If tbsave is the same for each file in series
    """
    if self.n_files > 0:
      if same_tbsave == False:
        dcdfile, self.fileframes, self.fparticles, self.timestep, self.tbsaves = self.do_opentraj(self.name, n_start, same_tbsave)
      elif same_tbsave == True:
        dcdfile, self.fileframes, self.fparticles, self.timestep, self.tbsave = self.do_opentraj(self.name, n_start, same_tbsave)
      else:
        raise RuntimeError("Invalid value of same_tbsave, must be Boolean")

      self.dcdfiles = list((dcdfile, ))

  def opentraj_multirun(self, n_start, same_tbsave):
    """
    Open a set of trajectory files in multiple runs, running
    consistency checks. Sets attributes in trajset class according to
    properties read from files.

    Arguments:
      n_start: int - Index on which trajectory file numbering starts
      same_tbsave: bool - If tbsave is supposed to be the same for each
        file in series
    """

    # 2D list of files, first dimension across runs, second across
    # files within each run
    self.dcdfiles = list()

    if self.n_files > 0:
      for i in range(0, self.runset.n_runs):
        if i == 0:
          if same_tbsave == True:
            dcdfile, self.fileframes, self.fparticles, self.timestep, self.tbsave = self.do_opentraj("%s%d/%s" %(self.runset.runname, i + 1, self.name), n_start, same_tbsave)
          elif same_tbsave == False:
            dcdfile, self.fileframes, self.fparticles, self.timestep, self.tbsaves = self.do_opentraj("%s%d/%s" %(self.runset.runname, i + 1, self.name), n_start, same_tbsave)
          else:
            raise RuntimeError("Invalid value of same_tbsave, must be Boolean")

        else:
          if same_tbsave == True:
            dcdfile, r_fileframes, r_fparticles, r_timestep, r_tbsave = self.do_opentraj("%s%d/%s" %(self.runset.runname, i + 1, self.name), n_start, same_tbsave)
          else:
            dcdfile, r_fileframes, r_fparticles, r_timestep, r_tbsaves = self.do_opentraj("%s%d/%s" %(self.runset.runname, i + 1, self.name), n_start, same_tbsave)

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
    Determine if option corresponds to file parameters in the opentraj
    module and process it if so. Returns True if option matched and
    processed, False otherwise.

    Arguments:
      o: str - Name of option to process, from array produced by
               gnu_getopt().
      a: str - Value of option to process, from array produced by
               gnu_getopt().
    """
    global one_open

    if o == "-" + self.opt:
      self.n_files = int(a)
    elif o == "--one-open":
      one_open = True
    else:
      # Option not matched
      return False

    return True

  def usage(self):
    """
    Print help documentation for options processed by the opentraj
    module.
    """
    global global_options_printed

    print("-%s Number of %s<n>.dcd files" %(self.opt, self.name), file=sys.stderr)
    if global_options_printed == False:
      print("--one-open Open DCD files one at a time instead of simultaneously", file=sys.stderr)
      global_options_printed = True
