import numpy as np
import pydcd
import sys
import math
import getopt

def usage():
  print("Arguments:",
        "-n Number of files",
        "-s Number of file to start on",
        "-d Number of frames between starts of pairs to average (dt)",
        "-a Overlap radius for theta function (default: 0.25)",
        "-q Scattering vector constant (default: 7.25)",
        "-h Print usage",
        sep="\n", file=sys.stderr)

try:
  opts, args = getopt.gnu_getopt(sys.argv[1:], "n:s:d:a:q:h")
except getopt.GetoptError as err:
  print(err, file=sys.stderr)
  usage()
  sys.exit(1)

# Total number of trajectory files
nfiles = 1
# Which file to start on
start_file = 0
# Difference between frame pair starts
framediff = 10
# Overlap radius for theta function
radius = 0.25
# Scattering vector constant
q = 7.25

for o, a in opts:
  if o == "-h":
    usage()
    sys.exit(0)
  elif o == "-n":
    nfiles = int(a)
  elif o == "-s":
    start_file = int(a) - 1
  elif o == "-d":
    framediff = int(a)
  elif o == "-a":
    radius = float(a)
  elif o == "-q":
    q = float(a)

# Holds number of frames per file
fileframes = np.empty(nfiles + 1, dtype=int)
fileframes[0] = 0

# Open each trajectory file
total_frames = 0
dcdfiles = list()
for i in range(0, nfiles):
  # The file object can be discarded after converting it to a dcd_file,
  # as the dcd_file duplicates the underlying file descriptor.
  file = open("traj%d.dcd" %(i + 1), "r")
  dcdfiles.append(pydcd.dcdfile(file))
  file.close()

  # Make sure each trajectory file has the same number of frames
  if i == 0:
    particles = dcdfiles[i].N
    timestep = dcdfiles[i].tbsave
  else:
    if dcdfiles[i].N != particles:
      raise RuntimeError("Not the same number of particles in each file")
    if dcdfiles[i].tbsave != timestep:
      raise RuntimeError("Not the same time step in each file")

  fileframes[i + 1] = dcdfiles[i].nset
  total_frames += dcdfiles[i].nset

# Now holds total index of last frame in each file
fileframes = np.cumsum(fileframes)

# Print basic properties shared across the files
print("%d %d" %(total_frames, particles), file=sys.stderr)

# What frame number to start on
start = fileframes[start_file]

# Number of frames to analyze
n_frames = total_frames - start

# Construct list of frame difference numbers for sampling according to
# a method of increasing spacing
magnitude = math.floor(math.log(n_frames / 50, 5))

frames_beyond_magnitude = n_frames - 1
for i in range(0, magnitude + 1):
  frames_beyond_magnitude -= 50 * 5**i

samples_beyond_magnitude = frames_beyond_magnitude // 5**(magnitude + 1)

n_samples = 1 + (50 * (magnitude + 1)) + samples_beyond_magnitude

# Allocate that array
samples = np.empty(n_samples, dtype=int)

# Efficiently fill the array
samples[0] = 0.0
last_sample_number = 0
for i in range(0, magnitude + 1):
  samples[1 + 50 * i : 1 + 50 * (i + 1)] = last_sample_number + np.arange(5**i , 51 * 5**i, 5**i)
  last_sample_number += 50 * 5**i
samples[1 + 50 * (magnitude + 1) : n_samples] = last_sample_number + np.arange(5**(magnitude + 1), (samples_beyond_magnitude + 1) * 5**(magnitude + 1), 5**(magnitude + 1))

# Stores coordinates of all particles in a frame
r = np.empty((3, particles), dtype=np.single)
r1 = np.empty((3, particles), dtype=np.single)

# Center of mass of each frame
cm = np.empty((n_frames, 3), dtype=float)

# Accumulated msd value for each difference in times
msd = np.empty(n_samples, dtype=float)

# Accumulated overlap value for each difference in times
overlap = np.empty(n_samples, dtype=float)

# Result of scattering function for each difference in times
fcx = np.empty(n_samples, dtype=float)
fcy = np.empty(n_samples, dtype=float)
fcz = np.empty(n_samples, dtype=float)
# Normalization factor for scattering indices
norm = np.zeros(n_samples, dtype=np.int64)

# Find center of mass of each frame
print("Finding centers of mass for frames", file=sys.stderr)
for i in range(0, n_frames):
  which_file = np.searchsorted(fileframes, i + start, side="right") - 1
  offset = i - (fileframes[which_file] - start)
  dcdfiles[which_file].gdcdp(r[0], r[1], r[2], offset)
  cm[i] = np.mean(r, axis=1)

# Reshape for broadcast to coordinates
cm = cm.reshape((n_frames, 3, 1))

# Iterate over starting points for functions
for i in np.arange(0, n_frames, framediff):
  which_file = np.searchsorted(fileframes, i + start, side="right") - 1
  offset = i - (fileframes[which_file] - start)
  dcdfiles[which_file].gdcdp(r[0], r[1], r[2], offset)

  # Iterate over ending points for functions and add to
  # accumulated values, making sure to only use indices
  # which are within the range of the files.
  for index, j in enumerate(samples[samples < (n_frames - i)]):
    which_file = np.searchsorted(fileframes, i + j + start, side="right") - 1
    offset = i + j - (fileframes[which_file] - start)
    dcdfiles[which_file].gdcdp(r1[0], r1[1], r1[2], offset)

    # Get means of scattering functions of all the particles for each
    # coordinate
    fcx[index] += np.mean(np.cos(q * ((r1[0] - cm[i + j][0][0]) - (r[0] - cm[i][0][0]))))
    fcy[index] += np.mean(np.cos(q * ((r1[1] - cm[i + j][1][0]) - (r[1] - cm[i][1][0]))))
    fcz[index] += np.mean(np.cos(q * ((r1[2] - cm[i + j][2][0]) - (r[2] - cm[i][2][0]))))

    # Add msd value to accumulated value
    msd[index] += 3 * np.mean(((r1 - cm[i + j]) - (r - cm[i]))**2)

    # Add overlap value to accumulated value
    overlap[index] += np.mean(np.less(np.linalg.norm(((r1 - cm[i + j]) - (r - cm[i])), axis=0), radius).astype(int))

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
fcx /= norm
fcy /= norm
fcz /= norm

# Normalize the overlap, thereby obtaining an average over each pair of frames
overlap /= norm

# Normalize the msd, thereby obtaining an average over each pair of frames
msd /= norm

for i in range(0, n_samples):
  time = samples[i] * timestep
  # Print time difference, msd, averarge overlap, x, y, and z
  # scattering function averages, number of frame sets contributing to
  # such averages, and frame difference
  print("%f %f %f %f %f %f %d %d" %(time, msd[i], overlap[i], fcx[i], fcy[i], fcz[i], norm[i], samples[i]))
