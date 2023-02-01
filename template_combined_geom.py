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
        "-m Number of files",
        "-z short(m).dcd file index to start on (default: 1)",
        "-c Number of frames in trajectory offset cycle of files",
        "-s Frame number to start on (index starts at 1)",
        "-a Overlap radius for theta function (default: 0.25)",
        "-q Scattering vector constant (default: 7.25)",
        "-o Start index (from 1) of particles to limit analysis to",
        "-p End index (from 1) of particles to limit analysis to",
        "-h Print usage",
        "-g Number of interval length values in geometric sequence (may be less due to rounding)",
        sep="\n", file=sys.stderr)

try:
  opts, args = getopt.gnu_getopt(sys.argv[1:], "m:z:c:s:a:q:o:p:hg:")
except getopt.GetoptError as err:
  print(err, file=sys.stderr)
  usage()
  sys.exit(1)

# Total number of trajectory files
n_files = 1
# Start trajectory file index in filenames for second region
m_start = 1
# Length in frames of cycle of offsets
set_len = None
# What frame number to start on
start = 0
# Overlap radius for theta function
radius = 0.25
# Scattering vector constant
q = 7.25
# Whether to limit analysis to subset of particles, and upper and lower
# indices for limit.
limit_particles = False
upper_limit = None
lower_limit = None
# Number of final times to use per initial time, used in geometric
# sequence
geom_num = None

for o, a in opts:
  if o == "-h":
    usage()
    sys.exit(0)
  elif o == "-m":
    n_files = int(a)
  elif o == "-z":
    m_start = int(a)
  elif o == "-c":
    set_len = int(a)
  elif o == "-s":
    start = int(a) - 1
  elif o == "-a":
    radius = float(a)
  elif o == "-q":
    q = float(a)
  elif o == "-o":
    limit_particles = True
    lower_limit = int(a) - 1
  elif o == "-p":
    limit_particles = True
    upper_limit = int(a)
  elif o == "-g":
    geom_num = int(a)

if set_len == None:
  raise RuntimeError("Must specify a set length")

if geom_num == None:
  raise RuntimeError("Must specify number of elements in geometric sampling sequence")

# Open trajectory files
dcdfiles, fileframes, fparticles, timestep, tbsaves = lib.opentraj.opentraj(n_files, "short", m_start, False)

# Now holds total index of last frame in each file
fileframes = np.cumsum(fileframes)

# Limit particles if necessary
if limit_particles == False:
  particles = fparticles
else:
  if lower_limit == None:
    lower_limit = 0
  if upper_limit == None:
    upper_limit = fparticles

  if lower_limit != 0 or upper_limit < fparticles:
    particles = upper_limit - lower_limit
  else:
    particles = fparticles
    limit_particles = False

# Print basic properties shared across the files
print("#nset: %d" %fileframes[-1])
print("#N: %d" %particles)
print("#timestep: %f" %timestep)

# Number of frames to analyze
n_frames = fileframes[-1] - start

# Largest possible offset between samples
max_offset = n_frames - 1

# Ensure frame set is long enough to work with chosen cycle
if n_frames < 2 * set_len:
  raise RuntimeError("Trajectory set not long enough for averaging "
                     "cycle, one may use non-averaging script instead.")

# Cycle of offset times
samp_cycle = np.empty(set_len, dtype=np.int64)
which_file = np.searchsorted(fileframes, start, side="right") - 1
offset = start - fileframes[which_file]
t1 = dcdfiles[which_file].itstart + offset * dcdfiles[which_file].tbsave
for i in range(0, set_len):
  t0 = t1
  which_file = np.searchsorted(fileframes, start + i + 1, side="right") - 1
  offset = start + i + 1 - fileframes[which_file]
  t1 = dcdfiles[which_file].itstart + offset * dcdfiles[which_file].tbsave

  # Store differences in iteration increments
  samp_cycle[i] = t1 - t0

# Total offset of full cycle
samp_sum = np.sum(samp_cycle)

# Get time of frame 0
which_file = np.searchsorted(fileframes, start, side="right") - 1
offset = start - fileframes[which_file]
zero_time = dcdfiles[which_file].itstart + offset * dcdfiles[which_file].tbsave

# Cumulative sum of sample cycle
samp_cycle_sum = np.insert(np.cumsum(samp_cycle), 0, 0)

# Verify that iterations do indeed follow cycle
for i in range(0, n_frames):
  which_file = np.searchsorted(fileframes, start + i, side="right") - 1
  offset = start + i - fileframes[which_file]
  t = dcdfiles[which_file].itstart + offset * dcdfiles[which_file].tbsave

  if t != samp_cycle_sum[i % set_len] + (i // set_len) * samp_sum + zero_time:
    raise RuntimeError("Frame %d in file %d does not seem to follow "
                       "specified cycle." %(offset, which_file + 1))

# Shift array to put smallest step first in sequence
shift_index = np.argmin(samp_cycle)
start += shift_index
n_frames -= shift_index
samp_cycle = np.roll(samp_cycle, -shift_index)
samp_cycle_sum = np.insert(np.cumsum(samp_cycle), 0, 0)

# Holds frame number offsets from initial time to sample
samples = np.empty(geom_num, dtype=np.int64)

# Base to use for geometric sequence to approximately fit full sample size
geom_base = (samp_cycle_sum[(n_frames - 1) % set_len] + ((n_frames - 1) // set_len) * samp_sum)**(1 / geom_num)

# Create sample array to approximate geometric sequence
for i in range(0, geom_num):
  # Geometric sequence value to find closest sample value to
  target = geom_base**(i + 1)

  # Array of cycled values adjusted to range that will contain target,
  # taking advantage of the fact that the samp_cycle_sum array includes
  # a representation of the smallest value of the next sequence.
  # Clamp values to minimum of 1 so that logarithm will work correctly.
  adjusted_array = np.maximum(1, samp_sum * (target // samp_sum) + samp_cycle_sum)

  # Calculate logarithmically closest sample, clamping to allowed
  # values
  samples[i] = min(n_frames - 1, max(1, set_len * (target // samp_sum) + np.argmin(np.absolute(np.log(adjusted_array) - math.log(target)))))

# Eliminate duplicate samples and prepend 0 for 0-length interval
samples = np.insert(np.unique(samples), 0, 0)

# Maximum number of samples per initial time. Likely less samples used
# for most initial times due to limited remaining space in trajectory
# set for offset
n_samples = samples.size

# If particles limited, must be read into different array
if limit_particles == True:
  x = np.empty(fparticles, dtype=np.single)
  y = np.empty(fparticles, dtype=np.single)
  z = np.empty(fparticles, dtype=np.single)

# Stores coordinates of all particles in a frame
x0 = np.empty(particles, dtype=np.single)
y0 = np.empty(particles, dtype=np.single)
z0 = np.empty(particles, dtype=np.single)
x1 = np.empty(particles, dtype=np.single)
y1 = np.empty(particles, dtype=np.single)
z1 = np.empty(particles, dtype=np.single)

# Center of mass of each frame
cm = np.empty((n_frames, 3), dtype=np.float64)

# Accumulated msd value for each difference in times
msd = np.zeros(n_samples, dtype=np.float64)

# Accumulated overlap value for each difference in times
overlap = np.zeros(n_samples, dtype=np.float64)

# Result of scattering function for each difference in times
fc = np.zeros((n_samples, 3), dtype=np.float64)

# Normalization factor for scattering indices
norm = np.zeros(n_samples, dtype=np.int64)

# Find center of mass of each frame
print("Finding centers of mass for frames", file=sys.stderr)
for i in range(0, n_frames):
  which_file = np.searchsorted(fileframes, start + i, side="right") - 1
  offset = start + i - fileframes[which_file]
  if limit_particles == True:
    dcdfiles[which_file].gdcdp(x, y, z, offset)
    x0[:] = x[lower_limit:upper_limit]
    y0[:] = y[lower_limit:upper_limit]
    z0[:] = z[lower_limit:upper_limit]
  else:
    dcdfiles[which_file].gdcdp(x0, y0, z0, offset)

  cm[i][0] = np.mean(x0)
  cm[i][1] = np.mean(y0)
  cm[i][2] = np.mean(z0)

# Iterate over starting points for functions
for i in np.arange(0, n_frames, set_len):
  which_file = np.searchsorted(fileframes, start + i, side="right") - 1
  offset = start + i - fileframes[which_file]
  if limit_particles == True:
    dcdfiles[which_file].gdcdp(x, y, z, offset)
    x0[:] = x[lower_limit:upper_limit]
    y0[:] = y[lower_limit:upper_limit]
    z0[:] = z[lower_limit:upper_limit]
  else:
    dcdfiles[which_file].gdcdp(x0, y0, z0, offset)

  # Iterate over ending points for functions and add to
  # accumulated values, making sure to only use indices
  # which are within the range of the files.
  for index, j in enumerate(samples):
    if j >= (n_frames - i):
      continue

    which_file = np.searchsorted(fileframes, start + i + j, side="right") - 1
    offset = start + i + j - fileframes[which_file]
    if limit_particles == True:
      dcdfiles[which_file].gdcdp(x, y, z, offset)
      x1[:] = x[lower_limit:upper_limit]
      y1[:] = y[lower_limit:upper_limit]
      z1[:] = z[lower_limit:upper_limit]
    else:
      dcdfiles[which_file].gdcdp(x1, y1, z1, offset)

    # Get means of scattering functions of all the particles for each
    # coordinate
    fc[index][0] += np.mean(np.cos(q * ((x1 - cm[i + j][0]) - (x0 - cm[i][0]))))
    fc[index][1] += np.mean(np.cos(q * ((y1 - cm[i + j][1]) - (y0 - cm[i][1]))))
    fc[index][2] += np.mean(np.cos(q * ((z1 - cm[i + j][2]) - (z0 - cm[i][2]))))

    # Add msd value to accumulated value
    msd[index] += np.mean(((x1 - cm[i + j][0]) - (x0 - cm[i][0]))**2 +
                          ((y1 - cm[i + j][1]) - (y0 - cm[i][1]))**2 +
                          ((z1 - cm[i + j][2]) - (z0 - cm[i][2]))**2)

    # Add overlap value to accumulated value
    overlap[index] += np.mean(np.less(np.sqrt(((x1 - cm[i + j][0]) - (x0 - cm[i][0]))**2 +
                                              ((y1 - cm[i + j][1]) - (y0 - cm[i][1]))**2 +
                                              ((z1 - cm[i + j][2]) - (z0 - cm[i][2]))**2), radius).astype(np.int8, copy=False))

    # Accumulate the normalization value for this sample offset, which
    # we will use later in computing the mean scattering value for each
    # offset
    norm[index] += 1

  print("Processed frame %d" %(i + start + 1), file=sys.stderr)

print("#q = %f" %q)
print("#a = %f" %radius)

# Normalize the accumulated scattering values, thereby obtaining
# averages over each pair of frames
fc /= norm.reshape((n_samples, 1))

# Normalize the overlap, thereby obtaining an average over each pair of
# frames
overlap /= norm

# Normalize the msd, thereby obtaining an average over each pair of
# frames
msd /= norm

for i in range(0, n_samples):
  time = timestep * ((samples[i]//set_len) * samp_sum + samp_cycle_sum[samples[i]%set_len])
  # Print time difference, msd, averarge overlap, x, y, and z
  # scattering function averages, average of directional scattering
  # function number of frame sets contributing to such averages, and
  # frame difference
  print("%f %f %f %f %f %f %f %d %d" %(time, msd[i], overlap[i], fc[i][0], fc[i][1], fc[i][2], (fc[i][0]+fc[i][1]+fc[i][2])/3, norm[i], samples[i]))
