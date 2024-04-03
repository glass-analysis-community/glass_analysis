import numpy as np
import sys
import pydcd

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
    n_runs: int - Number of runs and corresponding run folders.
    rundirs: bool -  If run folders are being used. Otherwise, script
      is being run in trajectory file directory.
    shortopts: str - List of short options processed by this module,
      used by gnu_getopt()
    longopts: str - List of long options processed by this module, used
      by gnu_getopt()
    argtext: str - Description of arguments processed by this module
  """
  n_runs = 1
  rundirs = False
  one_open = False

  longopts = ["one-open"]

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
    self.argtext = "-" + opt + " Number of runs, numbered as folders\n" \
                   + "--one-open Open DCD files one at a time instead of simultaneously"

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
    if o == "-" + self.opt:
      self.n_runs = int(a)
      self.rundirs = True
    elif o == "--one-open":
      self.one_open = True
    else:
      # Option not matched
      return False

    return True

class trajset():
  """
  Class for opening arrays of trajectory files.

  Attributes:
    n_files: int - Number of trajectory files in each run
    dcdfiles: list(list(pydcd.dcdfile)) - Two dimensional set of
      dcdfile objects for trajectory files.
    fileframes: np.array(dtype=np.int64) - Array of cumulative frames
      in and before each file number. First value is set to 0.
    fparticles: int - Number of particles in the simulation
    timestep: float - Duration of time corresponding to each simulation
      step.
    opt: str - Option on command line that specifies number of files
      for this trajectory set.
    shortopts: str - List of short options processed by this module,
      used by gnu_getopt()
    argtext: str - Description of arguments processed by this module

    tbsave: int - Simulation step difference between each saved frame.
      Set if tbsave is constant across files.
    tbsaves: np.array(np.int64) - Simulation step differences between
      each saved frame for each file number. Set if tbsave is not
      constant across files.
  """

  def __init__(self, runset, opt="n", name="traj", default_n_files=1, n_start=1, same_tbsave=True,):
    """
    Create trajset object

    Arguments:
      runset: class runset - Run set object associated with this
        trajectory set
      n_start: int - Index on which trajectory file numbering starts
      same_tbsave: bool - If tbsave is the same for each file in series
      opt: str - Letter of option that this trajectory file set object
        will catch to determine number of trajectory files (e.g. "n")
      name: str - Prefix of name of each DCD file in this trajectory
        set
      default_n_files: int - Default number of trajectory files in set
    """
    self.runset = runset
    self.opt = opt
    self.name = name
    self.n_files = default_n_files
    self.n_start = n_start
    self.same_tbsave = same_tbsave
    self.shortopts = opt + ":"
    self.argtext = "-" + opt + " Number of " + name + " <n>.dcd files"

  def do_opentraj(self, path):
    """
    Open a set of trajectory files for a given run and return array and
    attributes. Not meant to be called by the user. Instead, use
    opentraj().
    """
    # Returned list of dcd file objects
    dcdfiles = list()

    # Number of frames in each file, first element is 0
    fileframes = np.empty(self.n_files + 1, dtype=np.int64)
    fileframes[0] = 0

    if self.same_tbsave == False:
      tbsaves = np.empty(self.n_files, dtype=np.int64)

    for i in range(0, self.n_files):
      # The file object can be discarded after converting it to a
      # dcd_file, as the dcd_file duplicates the underlying file
      # descriptor.
      file = open("%s%d.dcd" %(path, self.n_start + i), "r")
      if self.runset.one_open == True:
        dcdfiles.append(dcdfile_w(pydcd.dcdfile(file), filename="%s%d.dcd" %(path, self.n_start + i)))
      else:
        dcdfiles.append(pydcd.dcdfile(file))
      file.close()

      # Make sure each trajectory file has the same time step and
      # number of particles
      if i == 0:
        # Number of particles in files, may not be same as limited
        # number in analysis
        fparticles = dcdfiles[i].N

        timestep = dcdfiles[i].timestep

        if self.same_tbsave == True:
          tbsave = dcdfiles[i].tbsave
        else:
          tbsaves[i] = dcdfiles[i].tbsave
      else:
        if dcdfiles[i].N != fparticles:
          raise RuntimeError("Not the same number of particles in each file")
        if dcdfiles[i].timestep != timestep:
          raise RuntimeError("Not the same time step in each file")

        if self.same_tbsave == True:
          if dcdfiles[i].tbsave != tbsave:
            raise RuntimeError("Not the same frame difference between saves in each file")
        else:
          tbsaves[i] = dcdfiles[i].tbsave

      fileframes[i + 1] = dcdfiles[i].nset

    # Change to hold total index of last frame in each file
    fileframes = np.cumsum(fileframes)

    if self.same_tbsave == True:
      return dcdfiles, fileframes, fparticles, timestep, tbsave
    else:
      return dcdfiles, fileframes, fparticles, timestep, tbsaves

  def opentraj(self):
    """
    Open a set of trajectory files, running consistency checks. Sets
    attributes in trajset class according to properties read from
    files.
    """
    # 2D list of files, first dimension across runs, second across
    # files within each run
    self.dcdfiles = list()

    if self.n_files > 0:
      if self.runset.rundirs == False:
        if self.same_tbsave == False:
          dcdfiles, self.fileframes, self.fparticles, self.timestep, self.tbsaves = self.do_opentraj(self.name)
        else:
          dcdfiles, self.fileframes, self.fparticles, self.timestep, self.tbsave = self.do_opentraj(self.name)

        self.dcdfiles.append(dcdfiles)

      else:
        for i in range(0, self.runset.n_runs):
          if i == 0:
            if self.same_tbsave == True:
              dcdfiles, self.fileframes, self.fparticles, self.timestep, self.tbsave = self.do_opentraj("%s%d/%s" %(self.runset.runname, i + 1, self.name))
            else:
              dcdfiles, self.fileframes, self.fparticles, self.timestep, self.tbsaves = self.do_opentraj("%s%d/%s" %(self.runset.runname, i + 1, self.name))

          else:
            if self.same_tbsave == True:
              dcdfiles, r_fileframes, r_fparticles, r_timestep, r_tbsave = self.do_opentraj("%s%d/%s" %(self.runset.runname, i + 1, self.name))
            else:
              dcdfiles, r_fileframes, r_fparticles, r_timestep, r_tbsaves = self.do_opentraj("%s%d/%s" %(self.runset.runname, i + 1, self.name))

            # Consistency checks across runs

            if (r_fileframes == self.fileframes).all() != True:
              raise RuntimeError("Not the same number of frames in each run for corresponding files")
            if r_fparticles != self.fparticles:
              raise RuntimeError("Not the same number of particles in each file")
            if r_timestep != self.timestep:
              raise RuntimeError("Not the same time step in each file")

            if self.same_tbsave == True:
              if r_tbsave != self.tbsave:
                raise RuntimeError("Not the same frame difference between saves in each file")
            else:
              if (r_tbsaves == self.tbsaves).all() != True:
                raise RuntimeError("Not the same tbsave in each run for corresponding files")

          # Append to multi-run list of dcd file lists
          self.dcdfiles.append(dcdfiles)

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
    if o == "-" + self.opt:
      self.n_files = int(a)
    else:
      # Option not matched
      return False

    return True
