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
        "-h Print usage",
        sep="\n")

try:
  opts, args = getopt.gnu_getopt(sys.argv[1:], "n:s:d:h")
except getopt.GetoptError as err:
  print(err)
  usage()
  sys.exit(1)

# Total number of trajectory files
nfiles = 1
# Which file to start on
start_file = 0
# Difference between frame pair starts
framediff = 10

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

# Accumulated msd value for each difference in times
msd = np.empty(n_samples, dtype=float)
# Normalization factor for accumulated msd values
norm = np.zeros(n_samples, dtype=np.int64)

# Iterate over starting points for msd
for i in np.arange(0, n_frames, framediff):
  which_file = np.searchsorted(fileframes, i + start, side="right") - 1
  offset = i - (fileframes[which_file] - start)
  dcdfiles[which_file].gdcdp(r[0], r[1], r[2], offset)

  # Iterate over ending points for msd and add to accumulated
  # scattering values, making sure to only use indices which are within
  # the range of the files.
  for index, j in enumerate(samples[samples < (n_frames - i)]):
    which_file = np.searchsorted(fileframes, i + j + start, side="right") - 1
    offset = i + j - (fileframes[which_file] - start)
    dcdfiles[which_file].gdcdp(r1[0], r1[1], r1[2], offset)

    # Add msd value to accumulated value
    msd[index] += 3 * np.mean((r1 - r)**2)

    # Accumulate the normalization value for this sample offset, which
    # we will use later in computing the mean msd for each offset
    norm[index] += 1

  print("Processed frame %d" %(i + start + 1), file=sys.stderr)

print("#dt = %f" %framediff)

# Normalize the msd, thereby obtaining an average over each pair of frames
msd /= norm

for i in range(0, n_samples):
  time = samples[i] * timestep
  # Print time difference, msd average, number of frame sets
  # contributing to such average, and frame difference
  print("%f %f %f %f" %(time, msd[i], norm[i], samples[i]))
