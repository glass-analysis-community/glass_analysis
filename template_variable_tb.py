import numpy as np
import pydcd
import sys
import math
import getopt
import enum

def usage():
  print("Arguments:",
        "-n Number of files",
        "-s Number of file to start on",
        "-d Number of frames between starts of pairs to average (dt)",
        "-q Fourier transform vector constant",
        "-h Print usage",
        "w function types (last specified is used, must be specified):",
        "-t Theta function threshold (argument is threshold radius)",
        "-f Double negative exponential/Gaussian (argument is exponential length)",
        "-e Single negative exponential (argument is exponential length)",
        "-i Imaginary negative exponential with directional dot product (argument is scattering vector constant)",
        sep="\n")

try:
  opts, args = getopt.gnu_getopt(sys.argv[1:], "n:s:d:q:ht:f:e:i:")
except getopt.GetoptError as err:
  print(err)
  usage()
  sys.exit(1)

class wtypes(enum.Enum):
  none = 1
  theta = 2
  gauss = 3
  exp = 4
  cos = 5

# Total number of trajectory files
nfiles = 1
# Which file to start on
start_file = 0
# Difference between frame set starts
framediff = 10
# Fourier transform vector constant
q = 6.1
# Type of w function to use
wtype = wtypes.none

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
  elif o == "-q":
    q = float(a)
  elif o == "-t":
    wtype = wtypes.theta
    radius = float(a)
  elif o == "-f":
    wtype = wtypes.gauss
    sscale = float(a)
  elif o == "-e":
    wtype = wtypes.exp
    gscale = float(a)
  elif o == "-i":
    wtype = wtypes.cos
    k = float(a)

if wtype == wtypes.none:
  raise RuntimeError("No w function type specified")

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
# a method of increasing spacing. These are used as t_b values
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

# Stores given coordinate of all particles in a frame
r = np.empty((3, particles), dtype=np.single)
r1 = np.empty((3, particles), dtype=np.single)

# Result of scattering function for each difference in times
fcx = np.empty(n_samples, dtype=float)
fcy = np.empty(n_samples, dtype=float)
fcz = np.empty(n_samples, dtype=float)
# Normalization factor for scattering indices
norm = np.zeros(n_samples, dtype=np.int64)

# W function values for each particle (if wtypes.cos, component-wise)
if wtype == wtypes.cos:
  w = np.empty((3, particles), dtype=float)
elif wtype == wtypes.theta:
  w = np.empty(particles, dtype=int)
else:
  w = np.empty(particles, dtype=float)

# Iterate over starting points for scattering function
for i in np.arange(0, n_frames, framediff):
  which_file = np.searchsorted(fileframes, i + start, side="right") - 1
  offset = i - (fileframes[which_file] - start)
  dcdfiles[which_file].gdcdp(r[0], r[1], r[2], offset)

  # Iterate over ending points for scattering function and add to
  # accumulated scattering values, making sure to only use indices
  # which are within the range of the files. j is used as t_b.
  for index, j in enumerate(samples[samples < (n_frames - i)]):
    which_file = np.searchsorted(fileframes, i + j + start, side="right") - 1
    offset = i + j - (fileframes[which_file] - start)
    dcdfiles[which_file].gdcdp(r1[0], r1[1], r1[2], offset)

    # Calculate w function for each particle
    if wtype == wtypes.theta:
      w = np.less(np.linalg.norm((r1 - r), axis=0), radius).astype(int)
    elif wtype == wtypes.gauss:
      w = np.exp(-(np.linalg.norm((r1 - r), axis=0)**2)/(2 * gscale**2))
    elif wtype == wtypes.exp:
      w = np.exp(-np.linalg.norm((r1 - r), axis=0)/sscale)
    elif wtype == wtypes.cos:
      # This may be a vector in the future, for now just take
      # component-wise with same value
      w = np.cos(k * (r1 - r))

    # Calculate normalized product of two sums. j in these expressions
    # is used to indicate python complex numbers, not the variable j.
    if wtype == wtypes.cos:
      fcx[index] += np.real(np.sum(w[0] * np.exp(-1j * q * r[0])) ** 2) / particles
      fcy[index] += np.real(np.sum(w[1] * np.exp(-1j * q * r[1])) ** 2) / particles
      fcz[index] += np.real(np.sum(w[2] * np.exp(-1j * q * r[2])) ** 2) / particles
    else:
      fcx[index] += np.real(np.sum(w * np.exp(-1j * q * r[0])) ** 2) / particles
      fcy[index] += np.real(np.sum(w * np.exp(-1j * q * r[1])) ** 2) / particles
      fcz[index] += np.real(np.sum(w * np.exp(-1j * q * r[2])) ** 2) / particles

    # Accumulate the normalization value for this sample offset, which
    # we will use later in computing the mean value for each t_b
    norm[index] += 1

  print("Processed frame %d" %(i + start + 1), file=sys.stderr)

print("#dt = %f" %framediff)
print("#q = %f" %q)

if wtype == wtypes.theta:
  print("#w function type: Threshold")
  print("#a = %f" %radius)
elif wtype == wtypes.gauss:
  print("#w function type: Gaussian")
  print("#a = %f" %gscale)
elif wtype == wtypes.exp:
  print("#w function type: Single Exponential")
  print("#a = %f" %sscale)
elif wtype == wtypes.cos:
  print("#w function type: Imaginary Exponential")
  print("#k = %f" %q)

# Normalize the accumulated values, thereby obtaining averages over
# each pair of frames
fcx /= norm
fcy /= norm
fcz /= norm

for i in range(0, n_samples):
  time_tb = samples[i] * timestep
  # Print t_b, x, y, and z averages, number of frame sets contributing
  # to such average, and frame difference corresponding to t_b
  print("%f %f %f %f %d %d" %(time_tb, fcx[i], fcy[i], fcz[i], norm[i], samples[i]))
