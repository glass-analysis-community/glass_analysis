import numpy as np
import pydcd

def opentraj(n, name, n_start, same_tbsave):
  """
  Open a set of trajectory files, running consistency checks. Returns
  a list of dcd file objects, an array with the number of frames in
  each trajectory, number of particles in file, timestep, and tbsave.
  If same_tbsave is false, a tbsave array is returned instead of a
  single tbsave value.
  n - Number of trajectory files
  name - Prefix of name of each DCD file
  n_start -- Index on which trajectory file numbering starts
  same_tbsave - If tbsave is supposed to be the same for each file in
    series
  """
  # Returned list of dcd file objects
  dcdfiles = list()

  # Number of frames in each file, first element is 0
  fileframes = np.empty(n + 1, dtype=np.int64)
  fileframes[0] = 0

  if same_tbsave == False:
    tbsaves = np.empty(n, dtype=np.int64)
  elif same_tbsave != True:
    raise RuntimeError("Invalid value of same_tbsave, must be Boolean")

  for i in range(0, n):
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

  if same_tbsave == True:
    return dcdfiles, fileframes, fparticles, timestep, tbsave
  else:
    return dcdfiles, fileframes, fparticles, timestep, tbsaves

def opentraj_multirun(r, runname, n, name, n_start, same_tbsave):
  """
  Open a set of trajectory files in multiple runs, running consistency
  checks. Returns a 2D list of dcd file objects, an array with the
  number of frames in each trajectory, number of particles in file,
  timestep, and tbsave. If same_tbsave is false, a tbsave array is
  returned instead of a single tbsave value.
  r - Number of runs
  runname - Prefix of name of each run directory
  n - Number of trajectory files
  name - Prefix of name of each DCD file
  n_start -- Index on which trajectory file numbering starts
  same_tbsave - If tbsave is supposed to be the same for each file in
    series
  """

  # 2D list of files, first dimension across runs, second across files
  # within each run
  dcdfiles = list()

  for i in range(0, r):
    if i == 0:
      if same_tbsave == True:
        dcdfile, fileframes, fparticles, timestep, tbsave = opentraj(n, "%s%d/%s" %(runname, i + 1, name), n_start, same_tbsave)
      elif same_tbsave == False:
        dcdfile, fileframes, fparticles, timestep, tbsaves = opentraj(n, "%s%d/%s" %(runname, i + 1, name), n_start, same_tbsave)
      else:
        raise RuntimeError("Invalid value of same_tbsave, must be Boolean")

    else:
      if same_tbsave == True:
        dcdfile, r_fileframes, r_fparticles, r_timestep, r_tbsave = opentraj(n, "%s%d/%s" %(runname, i + 1, name), n_start, same_tbsave)
      else:
        dcdfile, r_fileframes, r_fparticles, r_timestep, r_tbsaves = opentraj(n, "%s%d/%s" %(runname, i + 1, name), n_start, same_tbsave)

      # Consistency checks across runs

      if (r_fileframes == fileframes).all() != True:
        raise RuntimeError("Not the same number of frames in each run for corresponding files")
      if r_fparticles != fparticles:
        raise RuntimeError("Not the same number of particles in each file")
      if r_timestep != timestep:
        raise RuntimeError("Not the same time step in each file")

      if same_tbsave == True:
        if r_tbsave != tbsave:
          raise RuntimeError("Not the same frame difference between saves in each file")
      else:
        if (r_tbsaves == tbsaves).all() != True:
          raise RuntimeError("Not the same tbsave in each run for corresponding files")

    # Append to multi-run list of dcd file lists
    dcdfiles.append(dcdfile)

  if same_tbsave == True:
    return dcdfiles, fileframes, fparticles, timestep, tbsave
  else:
    return dcdfiles, fileframes, fparticles, timestep, tbsaves
