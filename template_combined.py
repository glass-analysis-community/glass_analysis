import numpy as np
import pydcd
import sys
import math
import getopt
import enum

def usage():
  print("Arguments:",
        "-n Number of files",
        "-s Frame number to start on (index starts at 1)",
        "-d Number of frames between starts of pairs to average (dt)",
        "-a Overlap radius for theta function (default: 0.25)",
        "-q Scattering vector constant (default: 7.25)",
        "-o Start index (from 1) of particles to limit analysis to"
        "-p End index (from 1) of particles to limit analysis to",
        "-h Print usage",
        "Interval increase progression (last specified is used):"
        "-f Flenner-style periodic-exponential-increasing increment (iterations: 50, power: 5)",
        "-g Geometric spacing progression, selectively dropped to fit on integer frames (argument is geometric base)",
        sep="\n", file=sys.stderr)

try:
  opts, args = getopt.gnu_getopt(sys.argv[1:], "n:s:d:a:q:o:p:hfg:")
except getopt.GetoptError as err:
  print(err, file=sys.stderr)
  usage()
  sys.exit(1)

class progtypes(enum.Enum):
  flenner = 1
  geometric = 2

# Total number of trajectory files
n_files = 1
# What frame number to start on
start = 0
# Difference between frame pair starts
framediff = 10
# Overlap radius for theta function
radius = 0.25
# Scattering vector constant
q = 7.25
# Whether to limit analysis to subset of particles, and upper and lower
# indices for limit.
limit_particles = False
upper_limit = None
lower_limit = None
# Type of progression to increase time interval by
progtype = progtypes.flenner

for o, a in opts:
  if o == "-h":
    usage()
    sys.exit(0)
  elif o == "-n":
    n_files = int(a)
  elif o == "-s":
    start = int(a) - 1
  elif o == "-d":
    framediff = int(a)
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
  elif o == "-f":
    progtype = progtypes.flenner
  elif o == "-g":
    progtype = progtypes.geometric
    geom_base = float(a)

# Holds number of frames per file
fileframes = np.empty(n_files + 1, dtype=int)
fileframes[0] = 0

# Open each trajectory file
total_frames = 0
dcdfiles = list()
for i in range(0, n_files):
  # The file object can be discarded after converting it to a dcd_file,
  # as the dcd_file duplicates the underlying file descriptor.
  file = open("traj%d.dcd" %(i + 1), "r")
  dcdfiles.append(pydcd.dcdfile(file))
  file.close()

  # Make sure each trajectory file has the same  time step and number
  # of particles
  if i == 0:
    # Number of particles in files, may not be same as limited
    # number in analysis
    fparticles = dcdfiles[i].N

    timestep = dcdfiles[i].timestep
    tbsave = dcdfiles[i].tbsave
  else:
    if dcdfiles[i].N != fparticles:
      raise RuntimeError("Not the same number of particles in each file")
    if dcdfiles[i].timestep != timestep:
      raise RuntimeError("Not the same time step in each file")
    if dcdfiles[i].tbsave != tbsave:
      raise RuntimeError("Not the same frame difference between saves in each file")

  fileframes[i + 1] = dcdfiles[i].nset
  total_frames += dcdfiles[i].nset

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

# Now holds total index of last frame in each file
fileframes = np.cumsum(fileframes)

# Print basic properties shared across the files
print("%d %d" %(total_frames, particles), file=sys.stderr)

# Number of frames to analyze
n_frames = total_frames - start

# Largest possible offset between samples
max_offset = n_frames - 1

if progtype == progtypes.flenner:
  # Construct list of frame difference numbers for sampling according
  # to a method of increasing spacing
  magnitude = -1
  frames_beyond_magnitude = max_offset
  while frames_beyond_magnitude >= 50 * 5**(magnitude + 1):
    magnitude += 1
    frames_beyond_magnitude -= 50 * 5**magnitude

  samples_beyond_magnitude = frames_beyond_magnitude // 5**(magnitude + 1)

  n_samples = 1 + (50 * (magnitude + 1)) + samples_beyond_magnitude

  # Allocate that array
  samples = np.empty(n_samples, dtype=int)

  # Efficiently fill the array
  samples[0] = 0
  last_sample_number = 0
  for i in range(0, magnitude + 1):
    samples[1 + 50 * i : 1 + 50 * (i + 1)] = last_sample_number + np.arange(5**i , 51 * 5**i, 5**i)
    last_sample_number += 50 * 5**i
  samples[1 + 50 * (magnitude + 1) : n_samples] = last_sample_number + np.arange(5**(magnitude + 1), (samples_beyond_magnitude + 1) * 5**(magnitude + 1), 5**(magnitude + 1))

elif progtype == progtypes.geometric:
  # Largest power of geom_base that will be able to be sampled
  end_power = math.floor(math.log(max_offset, geom_base))

  # Create array of sample numbers following geometric progression,
  # with flooring to have samples adhere to integer boundaries,
  # removing duplicate numbers, and prepending 0
  samples = np.insert(np.unique(np.floor(np.logspace(0, end_power, num=end_power + 1, base=geom_base)).astype(int)), 0, 0)

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
cm = np.empty((n_frames, 3), dtype=float)

# Accumulated msd value for each difference in times
msd = np.zeros(n_samples, dtype=float)

# Accumulated overlap value for each difference in times
overlap = np.zeros(n_samples, dtype=float)

# Result of scattering function for each difference in times
fc = np.zeros((n_samples, 3), dtype=float)

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
for i in np.arange(0, n_frames, framediff):
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
                                              ((z1 - cm[i + j][2]) - (z0 - cm[i][2]))**2), radius).astype(int))

    # Accumulate the normalization value for this sample offset, which
    # we will use later in computing the mean scattering value for each
    # offset
    norm[index] += 1

  print("Processed frame %d" %(i + start + 1), file=sys.stderr)

print("#dt = %f" %framediff)
print("#q = %f" %q)
print("#a = %f" %radius)

# Normalize the accumulated scattering values, thereby obtaining
# averages over each pair of frames
fc /= norm.reshape((n_samples, 1))

# Normalize the overlap, thereby obtaining an average over each pair of frames
overlap /= norm

# Normalize the msd, thereby obtaining an average over each pair of frames
msd /= norm

for i in range(0, n_samples):
  time = samples[i] * timestep * tbsave
  # Print time difference, msd, averarge overlap, x, y, and z
  # scattering function averages, average of directional scattering
  # function number of frame sets contributing to such averages, and
  # frame difference
  print("%f %f %f %f %f %f %d %d" %(time, msd[i], overlap[i], fc[i][0], fc[i][1], fc[i][2], (fc[i][0]+fc[i][1]+fc[i][2])/3, norm[i], samples[i]))
