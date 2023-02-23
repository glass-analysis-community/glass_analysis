import numpy as np
import pydcd
import sys
import math
import getopt
import enum

# Import functionality from local library directory
import lib.opentraj

def usage():
  print("Arguments:",
        "-n Number of files",
        "-r Number of runs, numbered as folders",
        "-s Frame number to start on (index starts at 1)",
        "-d Number of frames between starts of pairs to average (dt)",
        "-a Overlap radius for theta function (default: 0.25)",
        "-q Scattering vector constant (default: 7.25)",
        "-h Print usage",
        "Interval increase progression (last specified is used):"
        "-f Flenner-style periodic-exponential-increasing increment (iterations: 50, power: 5)",
        "-g Geometric spacing progression, selectively dropped to fit on integer frames (argument is geometric base)",
        sep="\n", file=sys.stderr)

try:
  opts, args = getopt.gnu_getopt(sys.argv[1:], "n:r:s:d:a:q:hfg:")
except getopt.GetoptError as err:
  print(err, file=sys.stderr)
  usage()
  sys.exit(1)

class progtypes(enum.Enum):
  flenner = 1
  geometric = 2

# Total number of trajectory files
n_files = 1
# Total number of run folders. 0 means not specified.
n_runs = 0
# What frame number to start on
start = 0
# Difference between frame pair starts
framediff = 10
# Overlap radius for theta function
radius = 0.25
# Scattering vector constant
q = 7.25
# Type of progression to increase time interval by
progtype = progtypes.flenner

for o, a in opts:
  if o == "-h":
    usage()
    sys.exit(0)
  elif o == "-n":
    n_files = int(a)
  elif o == "-r":
    n_runs = int(a)
  elif o == "-s":
    start = int(a) - 1
  elif o == "-d":
    framediff = int(a)
  elif o == "-a":
    radius = float(a)
  elif o == "-q":
    q = float(a)
  elif o == "-f":
    progtype = progtypes.flenner
  elif o == "-g":
    progtype = progtypes.geometric
    geom_base = float(a)

if n_runs <= 1:
  raise RuntimeError("Must have at least 2 runs")

# Open trajectory files
dcdfiles, fileframes, particles, timestep, tbsave = lib.opentraj.opentraj_multirun(n_runs, "run", n_files, "traj", 1, True)

# Print basic properties shared across the files
print("#nset: %d" %fileframes[-1])
print("#N: %d" %particles)
print("#timestep: %f" %timestep)
print("#tbsave: %f" %tbsave)

# Number of frames to analyze
n_frames = fileframes[-1] - start

# Largest possible lag
max_lag = n_frames - 1

if progtype == progtypes.flenner:
  # Construct list of lags according to a method of increasing spacing
  magnitude = -1
  frames_beyond_magnitude = max_lag
  while frames_beyond_magnitude >= 50 * 5**(magnitude + 1):
    magnitude += 1
    frames_beyond_magnitude -= 50 * 5**magnitude

  lags_beyond_magnitude = frames_beyond_magnitude // 5**(magnitude + 1)

  n_lags = 1 + (50 * (magnitude + 1)) + lags_beyond_magnitude

  # Allocate that array
  lag = np.empty(n_lags, dtype=np.int64)

  # Efficiently fill the array
  lags[0] = 0
  last_lag = 0
  for i in range(0, magnitude + 1):
    lags[1 + 50 * i : 1 + 50 * (i + 1)] = last_lag + np.arange(5**i , 51 * 5**i, 5**i)
    last_lag += 50 * 5**i
  lags[1 + 50 * (magnitude + 1) : n_lags] = last_lag + np.arange(5**(magnitude + 1), (lags_beyond_magnitude + 1) * 5**(magnitude + 1), 5**(magnitude + 1))

elif progtype == progtypes.geometric:
  # Largest power of geom_base that will be able to be used as a lag
  # value
  end_power = math.floor(math.log(max_lag, geom_base))

  # Create array of lags following geometric progression, with flooring
  # to have lags adhere to integer boundaries, removing duplicate
  # numbers, and prepending 0
  lags = np.insert(np.unique(np.floor(np.logspace(0, end_power, num=end_power + 1, base=geom_base)).astype(np.int64)), 0, 0)

  n_lags = lags.size

# Stores coordinates of all particles in a frame
x = np.empty(particles, dtype=np.single)
y = np.empty(particles, dtype=np.single)
z = np.empty(particles, dtype=np.single)
x1 = np.empty(particles, dtype=np.single)
y1 = np.empty(particles, dtype=np.single)
z1 = np.empty(particles, dtype=np.single)

# Center of mass of each frame
cm = [np.empty((n_frames, 3), dtype=np.float64)] * n_runs

# Accumulated msd variance value for each difference in times
msd = np.zeros(n_lags, dtype=np.float64)

# Accumulated overlap variance value for each difference in times
overlap = np.zeros(n_lags, dtype=np.float64)

# Result of scattering function variance for each difference in times
fc = np.zeros((n_lags, 3), dtype=np.float64)

# Normalization factor for scattering indices
norm = np.zeros(n_lags, dtype=np.int64)

# Find center of mass of each frame
print("Finding centers of mass for frames", file=sys.stderr)
for i in range(0, n_frames):
  which_file = np.searchsorted(fileframes, start + i, side="right") - 1
  offset = start + i - fileframes[which_file]
  for j in range(0, n_runs):
    dcdfiles[j][which_file].gdcdp(x, y, z, offset)
    cm[j][i][0] = np.mean(x)
    cm[j][i][1] = np.mean(y)
    cm[j][i][2] = np.mean(z)

# Accumulates squared values of given quantity across runs.
msd_a2_accum = 0.0
overlap_a2_acccum = 0.0
fc_a2_accum = np.empty(3, dtype=np.float64)

# Accumulates values of given quantity across runs.
msd_a_accum = 0.0
overlap_a_acccum = 0.0
fc_a_accum = np.empty(3, dtype=np.float64)

# Iterate over starting points for functions
for i in np.arange(0, n_frames, framediff):
  # Iterate over ending points for functions and add to
  # accumulated values, making sure to only use indices
  # which are within the range of the files.
  for index, j in enumerate(lags):
    if j >= (n_frames - i):
      continue

    # Clear accumulator values
    msd_a_accum = 0.0
    overlap_a_accum = 0.0
    fc_a_accum[:] = 0.0
    msd_a2_accum = 0.0
    overlap_a2_accum = 0.0
    fc_a2_accum[:] = 0.0

    for k in range(0, n_runs):
      which_file = np.searchsorted(fileframes, start + i, side="right") - 1
      offset = start + i - fileframes[which_file]
      dcdfiles[k][which_file].gdcdp(x, y, z, offset)

      which_file = np.searchsorted(fileframes, start + i + j, side="right") - 1
      offset = start + i + j - fileframes[which_file]
      dcdfiles[k][which_file].gdcdp(x1, y1, z1, offset)

      # Get means of scattering functions of all the particles for each
      # coordinate
      fcx_run = np.mean(np.cos(q * ((x1 - cm[k][i + j][0]) - (x - cm[k][i][0]))))
      fcy_run = np.mean(np.cos(q * ((y1 - cm[k][i + j][1]) - (y - cm[k][i][1]))))
      fcz_run = np.mean(np.cos(q * ((z1 - cm[k][i + j][2]) - (z - cm[k][i][2]))))

      fc_a_accum[0] += fcx_run
      fc_a_accum[1] += fcy_run
      fc_a_accum[2] += fcz_run
      fc_a2_accum[0] += fcx_run**2
      fc_a2_accum[1] += fcy_run**2
      fc_a2_accum[2] += fcz_run**2

      # Add msd value to accumulated value
      msd_run = np.mean(((x1 - cm[k][i + j][0]) - (x - cm[k][i][0]))**2 +
                        ((y1 - cm[k][i + j][1]) - (y - cm[k][i][1]))**2 +
                        ((z1 - cm[k][i + j][2]) - (z - cm[k][i][2]))**2)

      msd_a_accum += msd_run
      msd_a2_accum += msd_run**2

      # Add overlap value to accumulated value
      overlap_run = np.mean(np.less(((x1 - cm[k][i + j][0]) - (x - cm[k][i][0]))**2 +
                                    ((y1 - cm[k][i + j][1]) - (y - cm[k][i][1]))**2 +
                                    ((z1 - cm[k][i + j][2]) - (z - cm[k][i][2]))**2, radius**2).astype(np.int8, copy=False))

      overlap_a_accum += overlap_run
      overlap_a2_accum += overlap_run**2

    fc_a_accum /= n_runs
    fc_a2_accum /= n_runs
    msd_a_accum /= n_runs
    msd_a2_accum /= n_runs
    overlap_a_accum /= n_runs
    overlap_a2_accum /= n_runs

    # Calculate variances for lag index
    fc[index] += particles * (fc_a2_accum - fc_a_accum**2)
    msd[index] += particles * (msd_a2_accum - msd_a_accum**2)
    overlap[index] += particles * (overlap_a2_accum - overlap_a_accum**2)

    # Accumulate the normalization value for this lag, which we will
    # use later in computing the mean scattering value for each lag
    norm[index] += 1

  print("Processed frame %d" %(i + start + 1), file=sys.stderr)

print("#dt = %f" %framediff)
print("#q = %f" %q)
print("#a = %f" %radius)

# Normalize the accumulated scattering values, thereby obtaining
# averages over each pair of frames
fc /= norm.reshape((n_lags, 1))

# Normalize the overlap, thereby obtaining an average over each pair of frames
overlap /= norm

# Normalize the msd, thereby obtaining an average over each pair of frames
msd /= norm

for i in range(0, n_lags):
  time = lags[i] * timestep * tbsave
  # Print time difference, msd, averarge overlap, x, y, and z
  # scattering function averages, number of frame sets contributing to
  # such averages, and frame difference
  print("%f %f %f %f %f %f %d %d" %(time, msd[i], overlap[i], fc[i][0], fc[i][1], fc[i][2], norm[i], lags[i]))
