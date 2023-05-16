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
        "-h Print usage",
        sep="\n", file=sys.stderr)
  prog.usage()

try:
  opts, args = getopt.gnu_getopt(sys.argv[1:], "k:d:a:q:h" +
                                               runset.shortopts +
                                               trajset.shortopts +
                                               frames.shortopts +
                                               prog.shortopts,
                                               prog.longopts)
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

# Accumulated msd value for each difference in times
msd = np.zeros(lags.size, dtype=np.float64)

# Accumulated overlap value for each difference in times
overlap = np.zeros(lags.size, dtype=np.float64)

# Result of scattering function for each difference in times. In last
# dimension, first three indexes are x, y, and z, and last index is
# average between them.
fc = np.zeros((lags.size, 4), dtype=np.float64)

# Corresponding quantities for individual runs
run_msd = np.empty(lags.size, dtype=np.float64)
run_overlap = np.empty(lags.size, dtype=np.float64)
run_fc = np.empty((lags.size, 4), dtype=np.float64)

if runset.rundirs == True:
  # Corresponding arrays used for calculating standard deviations
  # across runs
  std_msd = np.zeros(lags.size, dtype=np.float64)
  std_overlap = np.zeros(lags.size, dtype=np.float64)
  std_fc = np.zeros((lags.size, 4), dtype=np.float64)

# Normalization factor for scattering indices
norm = np.zeros(lags.size, dtype=np.int64)

# Find center of mass of each frame
print("Finding centers of mass for frames", file=sys.stderr)
frames.generate_cm()

# Iterate over runs
for i in range(0, runset.n_runs):
  # Clear individual-run accumulators
  run_msd[:] = 0.0
  run_overlap[:] = 0.0
  run_fc[:] = 0.0

  # Iterate over starting points for functions
  for j in np.arange(0, initend - frames.start, framediff):
    # Get interval start frame
    frames.get_frame(j, x0, y0, z0, i)

    # Iterate over ending points for functions and add to
    # accumulated values, making sure to only use indices
    # which are within the range of the files.
    for index, k in enumerate(lags):
      if k >= (frames.n_frames - j) or k < -j:
        continue

      # Get interval end frame
      frames.get_frame(j + k, x1, y1, z1, i)

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

  # Accumulate individual-run quantities to total accumulators
  fc += run_fc
  msd += run_msd
  overlap += run_overlap

  if runset.rundirs == True:
    # Accumulate squares, to be later used for standard deviation
    # calculation
    std_fc += run_fc**2
    std_msd += run_msd**2
    std_overlap += run_overlap**2

if runset.rundirs == True:
  # Normalize calculated values across runs
  fc /= runset.n_runs
  msd /= runset.n_runs
  overlap /= runset.n_runs
  std_fc /= runset.n_runs
  std_msd /= runset.n_runs
  std_overlap /= runset.n_runs

  # Calculate standard deviation with means and means of squares of
  # values
  std_fc = np.sqrt(np.maximum(0.0, std_fc - fc**2) / (runset.n_runs - 1))
  std_msd = np.sqrt(np.maximum(0.0, std_msd - msd**2) / (runset.n_runs - 1))
  std_overlap = np.sqrt(np.maximum(0.0, std_overlap - overlap**2) / (runset.n_runs - 1))

for i in range(0, lags.size):
  time = lags[i] * trajset.timestep * trajset.tbsave
  if runset.rundirs == True:
    # Print output columns:
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
    # 14 - number of frame pairs in each run with interval
    # 15 - frame difference corresponding to interval time
    print("%f %f %f %f %f %f %f %f %f %f %f %f %f %d %d" %(time,
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
                                                           std_fc[i][3],
                                                           norm[i],
                                                           lags[i]))
  else:
    # Print output columns:
    # 1 - time difference constituting interval
    # 2 - mean squared displacement
    # 3 - average overlap
    # 4 - x scattering function
    # 5 - y scattering function
    # 6 - z scattering function
    # 7 - directional average scattering function
    # 8 - number of frame pairs with interval
    # 9 - frame difference corresponding to interval time
    print("%f %f %f %f %f %f %f %d %d" %(time,
                                         msd[i],
                                         overlap[i],
                                         fc[i][0],
                                         fc[i][1],
                                         fc[i][2],
                                         fc[i][3],
                                         norm[i],
                                         lags[i]))
