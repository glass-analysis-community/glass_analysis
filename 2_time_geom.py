import numpy as np
import pydcd
import sys

# Import functionality from local library directory
import lib.script
import lib.opentraj
import lib.progression
import lib.frame

# Script-specific variables altered by arguments
class svars:
  # Last frame number to use for initial times
  initend = None
  # Start trajectory file index in filenames for second region
  m_start = 1
  # Length in frames of cycle of offsets
  set_len = None
  # Overlap radius for theta function
  radius = 0.25
  # Scattering vector constant
  q = 7.25
  # Threshold for overlap between orientations
  thetab = None
  # Number of Legendre polynomial degrees to use in orientation
  # calculations
  legendre = None

# Progression specification/generation object for lags
prog = lib.progression.prog()
# Run set opening object
runset = lib.opentraj.runset()
# Trajectory set opening object
trajset = lib.opentraj.trajset(runset, opt="t", name="short", same_tbsave=False)
# Frame reading object
frames = lib.frame.frames(trajset)

# Script-specific short options
shortopts = "k:z:a:q:c:"

# Script-specific long options
longopts = ["polyatomic-thetab=", "polyatomic-legendre="]

# Description of script-specific arguments
argtext = "-k Last frame number in range to use for initial times (index starts at 1)\n" \
          + "-z short(m).dcd file index to start on (default: 1)\n" \
          + "-a Overlap radius for theta function (default: 0.25)\n" \
          + "-q Scattering vector constant (default: 7.25)\n" \
          + "-c Number of frames in trajectory offset cycle of files\n" \
          + "--polyatomic-legendre Largest degree of Legendre polynomial to calculate orientational correlation functions with (default: 1)\n" \
          + "--polyatomic-thetab Threshold for theta function of dot product of molecule orientations (default: 0.9)\n" \
          + "Intervals are adjusted to fit intervals present in files"

def catch_opt(o, a, svars):
  if o == "-k":
    svars.initend = int(a)
  elif o == "-z":
    svars.m_start = int(a)
  elif o == "-a":
    svars.radius = float(a)
  elif o == "-q":
    svars.q = float(a)
  elif o == "-c":
    svars.set_len = int(a)
  elif o == "--polyatomic-legendre":
    svars.legendre = int(a)
  elif o == "--polyatomic-thetab":
    svars.thetab = float(a)
  else:
    return False

  return True

def main_func(svars, prog, runset, trajset, frames):
  if svars.set_len == None:
    raise RuntimeError("Must specify a set length")

  # Adjust start of trajectory file numbering to match specified index
  trajset.n_start = svars.m_start

  # Open trajectory files
  trajset.opentraj()

  # Prepare frames object for calculation
  frames.prepare()

  # Ensure frame set is long enough to work with chosen cycle
  if frames.n_frames < 2 * svars.set_len:
    raise RuntimeError("Trajectory set not long enough for averaging "
                       "cycle, one may use non-averaging script instead.")

  if frames.n_atoms == None and (svars.legendre != None or svars.thetab != None):
    raise RuntimeError("Legendre polynomials degree or theta function threshold of orientation correlations specified without being in polyatomic mode")

  if frames.n_atoms != None:
    if svars.legendre == None:
      svars.legendre = 1

    if svars.thetab == None:
      svars.thetab = 0.9

  # Real time of first frame of analysis
  zero_time = frames.frame_time(0)

  # Offset of times in cycle from first time in cycle
  lag_cycle_sum = np.array([frames.frame_time(i) for i in range(0, svars.set_len + 1)]) - zero_time

  # Incremental offsets of times in cycle from each other
  lag_cycle = np.diff(lag_cycle_sum)

  # Total offset of full cycle
  lag_sum = lag_cycle_sum[-1]

  # Verify that iterations do indeed follow cycle
  for i in range(0, frames.n_frames):
    if frames.frame_time(i) != lag_cycle_sum[i % svars.set_len] + (i // svars.set_len) * lag_sum + zero_time:
      offset, which_file = frames.lookup_frame(i)
      raise RuntimeError("Frame %d in file %d does not seem to follow "
                         "specified cycle." %(offset, which_file + 1))

  # Shift array to put smallest step first in sequence
  shift_index = np.argmin(lag_cycle)
  frames.shift_start(shift_index)
  lag_cycle = np.roll(lag_cycle, -shift_index)
  lag_cycle_sum = np.insert(np.cumsum(lag_cycle), 0, 0)

  # End of set of frames to use for initial times
  if svars.initend == None:
    svars.initend = frames.final
  else:
    if svars.initend > frames.final:
      raise RuntimeError("End initial time frame beyond set of analyzed frames")

  # Largest possible positive and negative lags
  prog.max_val = lag_sum * ((frames.n_frames - 1) // svars.set_len) + lag_cycle_sum[(frames.n_frames - 1) % svars.set_len]
  prog.min_val = -prog.max_val

  # Construct array of permitted lag values, to which the values of the
  # progression will be adjusted to the logarithmically closest of
  prog.adj_seq = np.insert(np.cumsum(np.resize(lag_cycle, frames.n_frames - 1)), 0, 0.0)
  if prog.neg_vals == True:
    prog.adj_seq = np.concatenate((prog.adj_seq[:-1] - prog.adj_seq[-1], prog.adj_seq))
  prog.adj_log = True

  # Construct progression of interval values using previously-specified
  # parameters
  lags = prog.construct()

  # Print basic properties shared across the files
  print("#nset: %d" %frames.fileframes[-1])
  print("#N: %d" %frames.fparticles)
  print("#timestep: %f" %trajset.timestep)
  print("#q = %f" %svars.q)
  print("#a = %f" %svars.radius)

  # Stores coordinates of all particles in a frame
  x0 = np.empty(frames.particles, dtype=np.single)
  y0 = np.empty(frames.particles, dtype=np.single)
  z0 = np.empty(frames.particles, dtype=np.single)
  x1 = np.empty(frames.particles, dtype=np.single)
  y1 = np.empty(frames.particles, dtype=np.single)
  z1 = np.empty(frames.particles, dtype=np.single)

  if frames.n_atoms != None:
    # Stores unit orientation vectors of all particles in a frame
    xo0 = np.empty(frames.particles, dtype=np.single)
    yo0 = np.empty(frames.particles, dtype=np.single)
    zo0 = np.empty(frames.particles, dtype=np.single)
    xo1 = np.empty(frames.particles, dtype=np.single)
    yo1 = np.empty(frames.particles, dtype=np.single)
    zo1 = np.empty(frames.particles, dtype=np.single)

  # Accumulated msd value for each lag
  msd = np.zeros(lags.size, dtype=np.float64)

  # Accumulated overlap value for each lag
  overlap = np.zeros(lags.size, dtype=np.float64)

  # Result of scattering function for lag. In last dimension, first
  # three indexes are x, y, and z, and last index is average between
  # them.
  fc = np.zeros((lags.size, 4), dtype=np.float64)

  if frames.n_atoms != None:
    # Orientational correlation functions for each lag and each
    # Legendre polynomial degree. In first dimension, first index is
    # total part, second index is self part, and third index is
    # distinct part.
    ocorr = np.zeros((3, lags.size, svars.legendre), dtype=np.float64)

    # Fraction of Legendre polynomials of dot products between particle
    # orientations below given threshold for each lag.
    otheta = np.zeros(lags.size, dtype=np.float64)

    # Matrix of Legendre polynomial coefficients
    legmat = np.zeros((svars.legendre, svars.legendre + 1), dtype=np.float64)
    for i in range(0, svars.legendre):
      legmat[i,:i + 2] = np.polynomial.legendre.leg2poly([0] * (i + 1) + [1])

    # Array for computed power terms for total correlation functions
    powtotal = np.empty(svars.legendre + 1, dtype=np.float64)
    powtotal[0] = frames.particles

    # Create array of precomputed factorials
    facts = np.cumprod(np.insert(np.arange(1, svars.legendre + 1), 0, 1))

  # Corresponding quantities for individual runs
  run_msd = np.empty(lags.size, dtype=np.float64)
  run_overlap = np.empty(lags.size, dtype=np.float64)
  run_fc = np.empty((lags.size, 4), dtype=np.float64)
  if frames.n_atoms != None:
    run_otheta = np.empty(lags.size, dtype=np.float64)
    run_ocorr = np.empty((3, lags.size, svars.legendre), dtype=np.float64)

  if runset.rundirs == True:
    # Corresponding arrays used for calculating standard deviations
    # across runs
    std_msd = np.zeros(lags.size, dtype=np.float64)
    std_overlap = np.zeros(lags.size, dtype=np.float64)
    std_fc = np.zeros((lags.size, 4), dtype=np.float64)
    if frames.n_atoms != None:
      std_otheta = np.zeros(lags.size, dtype=np.float64)
      std_ocorr = np.zeros((3, lags.size, svars.legendre), dtype=np.float64)

  # Normalization factor for scattering indices
  norm = np.zeros(lags.size, dtype=np.int64)

  # Iterate over runs
  for i in range(0, runset.n_runs):
    # Clear individual-run accumulators
    run_msd[:] = 0.0
    run_overlap[:] = 0.0
    run_fc[:] = 0.0
    if frames.n_atoms != None:
      run_otheta[:] = 0.0
      run_ocorr[:, :, :] = 0.0

    # Iterate over starting points for functions
    for j in np.arange(0, svars.initend - frames.start, svars.set_len):
      # Get interval start frame
      frames.get_frame(j, x0, y0, z0, i)
      if frames.n_atoms != None:
        frames.get_orientations(j, xo0, yo0, zo0, i)

      # Iterate over ending points for functions and add to
      # accumulated values, making sure to only use indices
      # which are within the range of the files.
      for index, k in enumerate(lags):
        if k >= (frames.n_frames - j) or k < -j:
          continue

        # Get interval end frame
        frames.get_frame(j + k, x1, y1, z1, i)
        if frames.n_atoms != None:
          frames.get_orientations(j + k, xo1, yo1, zo1, i)

        # Add msd value to accumulated value
        run_msd[index] += np.mean((x1 - x0)**2 + (y1 - y0)**2 + (z1 - z0)**2)

        # Add overlap value to accumulated value
        run_overlap[index] += np.mean(np.less((x1 - x0)**2 +
                                              (y1 - y0)**2 +
                                              (z1 - z0)**2, svars.radius**2).astype(int))

        # Get means of scattering functions of all the particles for
        # each coordinate
        run_fc[index][0] += np.mean(np.cos(svars.q * (x1 - x0)))
        run_fc[index][1] += np.mean(np.cos(svars.q * (y1 - y0)))
        run_fc[index][2] += np.mean(np.cos(svars.q * (z1 - z0)))

        if frames.n_atoms != None:
          # Compute orientation theta functions and self correlation

          # Find dot product of orientation vectors and evaluate
          # Legendre polynomials
          legself = np.polynomial.legendre.legval(((xo0*xo1)+(yo0*yo1)+(zo0*zo1)), np.eye(N=svars.legendre + 1, M=svars.legendre, k=-1), tensor=True)

          # Calculate and accumulate theta functions
          run_otheta[index] += np.mean(np.greater_equal(legself[0], svars.thetab).astype(int))

          # Accumulate to self part
          run_ocorr[1][index] += np.mean(legself, axis=1)

          # Compute orientation total correlation

          # Compute dot product total parts for required powers
          for l in range(1, svars.legendre + 1):
            powtotal[l] = 0.0
            # Compute each part of trinomial expansion of 3-component
            # dot product for given power
            for m in range(0, l + 1):
              for n in range(0, l + 1 - m):
                powtotal[l] += (facts[l]/(facts[m]*facts[n]*facts[l-m-n])) * np.sum(xo0**m * yo0**n * zo0**(l+1-m-n)) * np.sum(xo1**m * yo1**n * zo1**(l-m-n)) / frames.particles

          # Compute Legendre polynomial from powers and accumulate
          run_ocorr[0][index] += np.sum(legmat * powtotal, axis=1)

        if i == 0:
          # Accumulate the normalization value for this lag, which we
          # will use later in computing the mean scattering value for
          # each lag
          norm[index] += 1

      print("Processed frame %d in run %d" %(j + frames.start + 1, i + 1), file=sys.stderr)

    # Normalize the accumulated scattering values, thereby obtaining
    # averages over each pair of frames
    run_fc[:, 0:3] /= norm.reshape((lags.size, 1))

    # Calculate directional average for scattering function
    run_fc[:, 3] = np.mean(run_fc[:, 0:3], axis=1)

    # Normalize the overlap, thereby obtaining an average over each
    # pair of frames
    run_overlap /= norm

    # Normalize the msd, thereby obtaining an average over each pair of
    # frames
    run_msd /= norm

    if frames.n_atoms != None:
      # Compute distinct part of orientation correlation from self and
      # total parts
      run_ocorr[2] = run_ocorr[0] - run_ocorr[1]

      # Normalize orientation correlation quantities
      run_ocorr[:] /= norm.reshape((lags.size, 1))
      run_otheta /= norm

    # Accumulate individual-run quantities to total accumulators
    fc += run_fc
    msd += run_msd
    overlap += run_overlap
    if frames.n_atoms != None:
      ocorr += run_ocorr
      otheta += run_otheta

    if runset.rundirs == True:
      # Accumulate squares, to be later used for standard deviation
      # calculation
      std_fc += run_fc**2
      std_msd += run_msd**2
      std_overlap += run_overlap**2
      if frames.n_atoms != None:
        std_ocorr += run_ocorr**2
        std_otheta += run_otheta**2

  if runset.rundirs == True:
    # Normalize calculated values across runs
    fc /= runset.n_runs
    msd /= runset.n_runs
    overlap /= runset.n_runs
    std_fc /= runset.n_runs
    std_msd /= runset.n_runs
    std_overlap /= runset.n_runs
    if frames.n_atoms != None:
      ocorr /= runset.n_runs
      otheta /= runset.n_runs
      std_ocorr /= runset.n_runs
      std_otheta /= runset.n_runs

    # Calculate standard deviation with means and means of squares of
    # values
    std_fc = np.sqrt(np.maximum(0.0, std_fc - fc**2) / (runset.n_runs - 1))
    std_msd = np.sqrt(np.maximum(0.0, std_msd - msd**2) / (runset.n_runs - 1))
    std_overlap = np.sqrt(np.maximum(0.0, std_overlap - overlap**2) / (runset.n_runs - 1))
    if frames.n_atoms != None:
      std_ocorr = np.sqrt(np.maximum(0.0, std_ocorr - ocorr**2) / (runset.n_runs - 1))
      std_otheta = np.sqrt(np.maximum(0.0, std_otheta - otheta**2) / (runset.n_runs - 1))

  # Print description of output columns
  print("#",
        "#Output Columns:",
        sep="\n")
  if runset.rundirs == True:
    print("#  1 - Time difference constituting interval",
          "#  2 - Run average of mean squared displacement",
          "#  3 - Run average of average overlap",
          "#  4 - Run average of x scattering function",
          "#  5 - Run average of y scattering function",
          "#  6 - Run average of z scattering function",
          "#  7 - Run average of directional average scattering function",
          "#  8 - Standard deviation across runs of mean squared displacement",
          "#  9 - Standard deviation across runs of average overlap",
          "#  10 - Standard deviation across runs of x scattering function",
          "#  11 - Standard deviation across runs of y scattering function",
          "#  12 - Standard deviation across runs of z scattering function",
          "#  13 - Standard deviation across runs of directional average scattering function",
          sep="\n")
    if frames.n_atoms != None:
      print("#  14 - Run average of orientational correlation function theta function average",
            "#  15 - Standard deviation across runs of orientational correlation function theta function average",
            sep="\n")
      for i in range(1, svars.legendre + 1):
        print("#  %d - Run average of total part of Legendre degree %d of orientational correlation function" %(6 * i + 10, i),
              "#  %d - Run average of self part of Legendre degree %d of orientational correlation function" %(6 * i + 11, i),
              "#  %d - Run average of distinct part of Legendre degree %d of orientational correlation function" %(6 * i + 12, i),
              "#  %d - Standard deviation across runs of total part of Legendre degree %d of orientational correlation function" %(6 * i + 13, i),
              "#  %d - Standard deviation across runs of self part of Legendre degree %d of orientational correlation function" %(6 * i + 14, i),
              "#  %d - Standard deviation across runs of distinct part of Legendre degree %d of orientational correlation function" %(6 * i + 15, i),
              sep="\n")
      print("#  %d - Number of frame pairs in each run with interval" %(6 * svars.legendre + 16),
            "#  %d - Number of frame pairs in each run with interval" %(6 * svars.legendre + 17),
            sep="\n")
    else:
      print("#  14 - Number of frame pairs in each run with interval",
            "#  15 - Number of frame pairs in each run with interval",
            sep="\n")
  else:
    print("#  1 - Time difference constituting interval",
          "#  2 - Mean squared displacement",
          "#  3 - Average overlap",
          "#  4 - x scattering function",
          "#  5 - y scattering function",
          "#  6 - z scattering function",
          "#  7 - Directional average scattering function",
          sep="\n")
    if frames.n_atoms != None:
      print("#  8 - Orientational correlation theta function average")
      for i in range(1, svars.legendre + 1):
        print("#  %d - Total part of Legendre degree %d of orientational correlation function" %(3 * i + 6, i),
              "#  %d - Self part of Legendre degree %d of orientational correlation function" %(3 * i + 7, i),
              "#  %d - Distinct part of Legendre degree %d of orientational correlation function" %(3 * i + 8, i),
              sep="\n")
      print("#  %d - Number of frame pairs with interval" %(3 * svars.legendre + 9),
            "#  %d - Frame difference corresponding to interval time" %(3 * svars.legendre + 10),
            sep="\n")
    else:
      print("#  8 - Number of frame pairs with interval",
            "#  9 - Frame difference corresponding to interval time",
            sep="\n")
  print("#")

  for i in range(0, lags.size):
    time = trajset.timestep * ((lags[i]//svars.set_len) * lag_sum + lag_cycle_sum[lags[i]%svars.set_len])
    if runset.rundirs == True:
      # Print output columns
      sys.stdout.write("%f %f %f %f %f %f %f %f %f %f %f %f %f "
                       %(time,
                         msd[i],
                         overlap[i],
                         fc[i][0],
                         fc[i][1],
                         fc[i][2],
                         fc[i][3],
                         std_msd[i],
                         std_overlap[i],
                         std_fc[i][0],
                         std_fc[i][1],
                         std_fc[i][2],
                         std_fc[i][3]))
      if frames.n_atoms != None:
        sys.stdout.write("%f %f "
                         %(otheta[i],
                           std_otheta[i]))
        for j in range(0, svars.legendre):
          sys.stdout.write("%f %f %f %f %f %f "
                           %(ocorr[0][i][j],
                             ocorr[1][i][j],
                             ocorr[2][i][j],
                             std_ocorr[0][i][j],
                             std_ocorr[1][i][j],
                             std_ocorr[2][i][j]))
      sys.stdout.write("%d %d\n"
                       %(norm[i],
                         lags[i]))
    else:
      # Print output columns
      sys.stdout.write("%f %f %f %f %f %f %f "
                       %(time,
                         msd[i],
                         overlap[i],
                         fc[i][0],
                         fc[i][1],
                         fc[i][2],
                         fc[i][3]))
      if frames.n_atoms != None:
        sys.stdout.write("%f "
                         %(otheta[i]))
        for j in range(0, svars.legendre):
          sys.stdout.write("%f %f %f "
                           %(ocorr[0][i][j],
                             ocorr[1][i][j],
                             ocorr[2][i][j]))
      sys.stdout.write("%d %d\n"
                       %(norm[i],
                         lags[i]))

# Run full program
lib.script.run(main_func,
               argtext=argtext,
               svars=svars,
               shortopts=shortopts,
               longopts=longopts,
               catch_opt=catch_opt,
               modules=[prog, runset, trajset, frames])
