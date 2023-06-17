import numpy as np
import pydcd
import sys
import math
import getopt
import enum

# Import functionality from local library directory
import lib.opentraj
import lib.progression
import lib.frame

# Last frame number to use for initial times
initend = None
# Difference between frame pair starts
framediff = 10
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
trajset = lib.opentraj.trajset(runset)
# Frame reading object
frames = lib.frame.frames(trajset)

def usage():
  print("Arguments:", file=sys.stderr)
  runset.usage()
  trajset.usage()
  frames.usage()
  print("-k Last frame number in range to use for initial times (index starts at 1)",
        "-d Number of frames between starts of pairs to average (dt)",
        "-a Overlap radius for theta function (default: 0.25)",
        "-q Scattering vector constant (default: 7.25)",
        "--polyatomic-legendre Largest degree of Legendre polynomial to calculate orientational correlation functions with (default: 1)",
        "--polyatomic-thetab Threshold for theta function of dot product of molecule orientations (default: 0.9)",
        "-h Print usage",
        sep="\n", file=sys.stderr)
  prog.usage()

try:
  opts, args = getopt.gnu_getopt(sys.argv[1:], "k:d:a:q:h" +
                                               runset.shortopts +
                                               trajset.shortopts +
                                               frames.shortopts +
                                               prog.shortopts,
                                               ["polyatomic-thetab=", "polyatomic-legendre="] +
                                               prog.longopts +
                                               frames.longopts)
except getopt.GetoptError as err:
  print(err, file=sys.stderr)
  usage()
  sys.exit(1)

for o, a in opts:
  if o == "-h":
    usage()
    sys.exit(0)
  elif o == "-k":
    initend = int(a)
  elif o == "-d":
    framediff = int(a)
  elif o == "-a":
    radius = float(a)
  elif o == "-q":
    q = float(a)
  elif o == "--polyatomic-legendre":
    legendre = int(a)
  elif o == "--polyatomic-thetab":
    thetab = float(a)
  elif runset.catch_opt(o, a) == True:
    pass
  elif trajset.catch_opt(o, a) == True:
    pass
  elif prog.catch_opt(o, a) == True:
    pass
  elif frames.catch_opt(o, a) == True:
    pass

# Open trajectory files
if runset.rundirs == True:
  trajset.opentraj_multirun(1, True)
else:
  trajset.opentraj(1, True)

# Prepare frames object for calculation
frames.prepare()

# Print basic properties of files and analysis
print("#nset: %d" %frames.fileframes[-1])
print("#N: %d" %frames.fparticles)
print("#timestep: %f" %trajset.timestep)
print("#tbsave: %f" %trajset.tbsave)
print("#dt = %f" %framediff)
print("#q = %f" %q)
print("#a = %f" %radius)

# End of set of frames to use for initial times
if initend == None:
  initend = trajset.fileframes[-1]
else:
  if initend > trajset.fileframes[-1]:
    raise RuntimeError("End initial time frame beyond set of frames")

if frames.n_atoms == None and (legendre != None or thetab != None):
  raise RuntimeError("Legendre polynomials degree or theta function threshold of orientation correlations specified without being in polyatomic mode")

if frames.n_atoms != None:
  if legendre == None:
    legendre = 1

  if thetab == None:
    thetab = 0.9

# Largest possible positive and negative lags
prog.max_val = frames.n_frames - 1
prog.min_val = -frames.n_frames + 1

# Construct progression of interval values using previously-specified
# parameters
lags = prog.construct()

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

# Result of scattering function for lag. In last dimension, first three
# indexes are x, y, and z, and last index is average between them.
fc = np.zeros((lags.size, 4), dtype=np.float64)

if frames.n_atoms != None:
  # Orientational correlation functions for each lag and each Legendre
  # polynomial degree. In first dimension, first index is total part,
  # second index is self part, and third index is distinct part.
  ocorr = np.zeros((3, lags.size, legendre), dtype=np.float64)

  # Fraction of Legendre polynomials of dot products between particle
  # orientations below given threshold for each lag.
  otheta = np.zeros(lags.size, dtype=np.float64)

  # Matrix of Legendre polynomial coefficients
  legmat = np.zeros((legendre, legendre + 1), dtype=np.float64)
  for i in range(0, legendre):
    legmat[i,:i + 2] = np.polynomial.legendre.leg2poly([0] * (i + 1) + [1])

  # Array for computed power terms for total correlation functions
  powtotal = np.empty(legendre + 1, dtype=np.float64)
  powtotal[0] = frames.particles

# Corresponding quantities for individual runs
run_msd = np.empty(lags.size, dtype=np.float64)
run_overlap = np.empty(lags.size, dtype=np.float64)
run_fc = np.empty((lags.size, 4), dtype=np.float64)
if frames.n_atoms != None:
  run_otheta = np.empty(lags.size, dtype=np.float64)
  run_ocorr = np.empty((3, lags.size, legendre), dtype=np.float64)

if runset.rundirs == True:
  # Corresponding arrays used for calculating standard deviations
  # across runs
  std_msd = np.zeros(lags.size, dtype=np.float64)
  std_overlap = np.zeros(lags.size, dtype=np.float64)
  std_fc = np.zeros((lags.size, 4), dtype=np.float64)
  if frames.n_atoms != None:
    std_otheta = np.zeros(lags.size, dtype=np.float64)
    std_ocorr = np.zeros((3, lags.size, legendre), dtype=np.float64)

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
  for j in np.arange(0, initend - frames.start, framediff):
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
                                            (z1 - z0)**2, radius**2).astype(int))

      # Get means of scattering functions of all the particles for each
      # coordinate
      run_fc[index][0] += np.mean(np.cos(q * (x1 - x0)))
      run_fc[index][1] += np.mean(np.cos(q * (y1 - y0)))
      run_fc[index][2] += np.mean(np.cos(q * (z1 - z0)))

      if frames.n_atoms != None:
        # Compute orientation theta functions and self correlation

        # Find dot product of orientation vectors and evaluate Legendre
        # polynomials
        legself = np.polynomial.legendre.legval(((xo0*xo1)+(yo0*yo1)+(zo0*zo1)), np.eye(N=legendre + 1, M=legendre, k=-1), tensor=True)

        # Calculate and accumulate theta functions
        run_otheta[index] += np.mean(np.greater_equal(legself[0], thetab).astype(int))

        # Accumulate to self part
        run_ocorr[1][index] += np.mean(legself, axis=1)

        # Compute orientation total correlation

        # Compute dot product total parts for required powers
        for l in range(1, legendre + 1):
          powtotal[l] = 0.0
          # Compute each part of trinomial expansion of 3-component dot
          # product for given power
          for m in range(0, l + 1):
            for n in range(0, l + 1 - m):
              powtotal[l] += (math.factorial(l)/(math.factorial(m)*math.factorial(n)*math.factorial(l-m-n))) * np.sum(xo0**m * yo0**n * zo0**(l+1-m-n)) * np.sum(xo1**m * yo1**n * zo1**(l-m-n)) / frames.particles

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

  # Normalize the overlap, thereby obtaining an average over each pair
  # of frames
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

for i in range(0, lags.size):
  time = lags[i] * trajset.timestep * trajset.tbsave
  if runset.rundirs == True:
    # Print output columns:
    # (n - current Legendre degree)
    # (m - total number of Legendre degrees)
    # (l - 1 if polyatomic, 0 otherwise)
    # 1 - time difference constituting interval
    # 2 - mean squared displacement run average
    # 3 - average overlap run average
    # 4 - x scattering function run average
    # 5 - y scattering function run average
    # 6 - z scattering function run average
    # 7 - directional average scattering function run average
    # 8 - mean squared displacement standard deviation
    # 9 - average overlap standard deviation
    # 10 - x scattering function standard deviation
    # 11 - y scattering function standard deviation
    # 12 - z scattering function standard deviation
    # 13 - directional average scattering function
    # 14 - orientational correlation theta function average run average
    #      (if polyatomic)
    # 15 - orientational correlation theta function average standard
    #      deviation (if polyatomic)
    # 10 + 6n - orientational correlation function total part run
    #           average
    # 11 + 6n - orientational correlation function self part run
    #           average
    # 12 + 6n - orientational correlation function distinct part run
    #           average
    # 13 + 6n - orientational correlation function total part standard
    #           deviation
    # 14 + 6n - orientational correlation function self part standard
    #           deviation
    # 15 + 6n - orientational correlation function distinct part
    #           standard deviation
    # 14 + 6m + 2l - number of frame pairs in each run with interval
    # 15 + 6m + 2l - frame difference corresponding to interval time
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
      for j in range(0, legendre):
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
    # Print output columns:
    # (n - current Legendre degree)
    # (m - total number of Legendre degrees)
    # (l - 1 if polyatomic, 0 otherwise)
    # 1 - time difference constituting interval
    # 2 - mean squared displacement
    # 3 - average overlap
    # 4 - x scattering function
    # 5 - y scattering function
    # 6 - z scattering function
    # 7 - directional average scattering function
    # 8 - orientational correlation theta function average (if
    #     polyatomic)
    # 5 + 3n - orientational correlation function total part
    # 6 + 3n - orientational correlation function self part
    # 7 + 3n - orientational correlation function distinct part
    # 8 + 3m + l - number of frame pairs with interval
    # 9 + 3m + l - frame difference corresponding to interval time
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
      for j in range(0, legendre):
        sys.stdout.write("%f %f %f "
                         %(ocorr[0][i][j],
                           ocorr[1][i][j],
                           ocorr[2][i][j]))
    sys.stdout.write("%d %d\n"
                     %(norm[i],
                       lags[i]))
