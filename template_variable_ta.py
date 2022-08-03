import numpy as np
import pydcd
import sys
import math
import getopt
import enum

def usage():
  print("Arguments:",
        "-n Number of files",
        "-r Number of runs, numbered as folders",
        "-s Frame number to start on (index starts at 1)",
        "-d Number of frames between starts of pairs to average (dt)",
        "-x Number of Fourier transform vector constants to used in addition to q=0",
        "-y Box size in each dimension (assumed to be cubic, required)"
        "-b Average interval in frames (t_b)",
        "-c Difference between intervals in frames (t_c)",
        "-h Print usage",
        "Interval increase progression (last specified is used):"
        "-f Flenner-style periodic-exponential-increasing increment (iterations: 50, power: 5)",
        "-g Geometric spacing progression, selectively dropped to fit on integer frames (argument is geometric base)",
        "w function types (last specified is used, must be specified):",
        "-t Theta function threshold (argument is threshold radius)",
        "-u Double negative exponential/Gaussian (argument is exponential length)",
        "-e Single negative exponential (argument is exponential length)",
        "-i Imaginary negative exponential with directional dot product (argument is scattering vector constant)",
        sep="\n", file=sys.stderr)

try:
  opts, args = getopt.gnu_getopt(sys.argv[1:], "n:r:s:d:x:y:b:c:hfg:t:u:e:i:")
except getopt.GetoptError as err:
  print(err, file=sys.stderr)
  usage()
  sys.exit(1)

class progtypes(enum.Enum):
  flenner = 1
  geometric = 2

class wtypes(enum.Enum):
  none = 1
  theta = 2
  gauss = 3
  exp = 4
  ima = 5

class stypes(enum.Enum):
  total = 0
  self = 1
  distinct = 2
n_stypes = len(stypes)

# Total number of trajectory files
n_files = 1
# Total number of run folders. If not specified, this is being run in
# directory with trajectory files.
n_runs = 1
rundirs = False
# What frame number to start on
start = 0
# Difference between frame set starts
framediff = 10
# Number of Fourier transform vector constants (including q=0)
n_q = 1
# If box size was defined
box_size_defined = False
# Average length of intervals (t_b)
tb = 1
# Half difference between length of initial and end intervals (t_c)
tc = 0
# Type of progression to increase time interval by
progtype = progtypes.flenner
# Type of w function to use
wtype = wtypes.none

for o, a in opts:
  if o == "-h":
    usage()
    sys.exit(0)
  elif o == "-n":
    n_files = int(a)
  elif o == "-r":
    n_runs = int(a)
    rundirs = True
  elif o == "-s":
    start = int(a) - 1
  elif o == "-d":
    framediff = int(a)
  elif o == "-x":
    n_q = int(a) + 1
  elif o == "-y":
    box_size = float(a)
    box_size_defined = True
  elif o == "-b":
    tb = int(a)
  elif o == "-c":
    tc = int(a)
  elif o == "-f":
    progtype = progtypes.flenner
  elif o == "-g":
    progtype = progtypes.geometric
    geom_base = float(a)
  elif o == "-t":
    wtype = wtypes.theta
    radius = float(a)
  elif o == "-u":
    wtype = wtypes.gauss
    sscale = float(a)
  elif o == "-e":
    wtype = wtypes.exp
    gscale = float(a)
  elif o == "-i":
    wtype = wtypes.ima
    kvec = float(a)

if wtype == wtypes.none:
  raise RuntimeError("No w function type specified")

if box_size_defined == False:
  raise RuntimeError("Must define box size dimensions")

# Holds number of frames per file
fileframes = np.empty(n_files + 1, dtype=int)
fileframes[0] = 0

# 2D list of files, first dimension across runs, second across files
# within each run
dcdfiles = list()

# Open each trajectory file
total_frames = 0
for i in range(0, n_runs):
  dcdfiles.append(list())
  for j in range(0, n_files):
    # The file object can be discarded after converting it to a dcd_file,
    # as the dcd_file duplicates the underlying file descriptor.
    if rundirs == True:
      file = open("run%d/traj%d.dcd" %(i + 1, j + 1), "r")
    else:
      file = open("traj%d.dcd" %(j + 1), "r")
    dcdfiles[i].append(pydcd.dcdfile(file))
    file.close()

    # Make sure each trajectory file in each run mirrors the files in
    # other runs and has the same time step and number of particles
    if i == 0:
      fileframes[j + 1] = dcdfiles[i][j].nset
      total_frames += dcdfiles[i][j].nset

      if j == 0:
        particles = dcdfiles[i][j].N
        timestep = dcdfiles[i][j].timestep
      else:
        if dcdfiles[i][j].N != particles:
          raise RuntimeError("Not the same number of particles in each file")
        if dcdfiles[i][j].timestep != timestep:
          raise RuntimeError("Not the same time step in each file")

    else:
      if dcdfiles[i][j].nset != fileframes[j + 1]:
        raise RuntimeError("Not the same number of frames in each run for corresponding files")

      if dcdfiles[i][j].N != particles:
        raise RuntimeError("Not the same number of particles in each file")
      if dcdfiles[i][j].tbsave != timestep:
        raise RuntimeError("Not the same time step in each file")

# Now holds total index of last frame in each file
fileframes = np.cumsum(fileframes)

# Print basic properties shared across the files
print("%d %d" %(total_frames, particles), file=sys.stderr)

# Number of frames in each run to analyze
n_frames = total_frames - start

# Largest possible offset between samples
max_offset = n_frames - tb - 1

if progtype == progtypes.flenner:
  # Construct list of frame difference numbers for sampling according
  # to a method of increasing spacing
  magnitude = math.floor(math.log(max_offset / 50, 5))

  frames_beyond_magnitude = max_offset
  for i in range(0, magnitude + 1):
    frames_beyond_magnitude -= 50 * 5**i

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

# Array with progression of q values to use, with 0.0 always at index 0.
# All of these create integral number of wave periods inside the box.
qs = np.linspace(0.0, (n_q - 1) * 2 * math.pi / box_size, num=n_q)

# Stores coordinates of all particles in a frame
x0 = np.empty(particles, dtype=np.single)
y0 = np.empty(particles, dtype=np.single)
z0 = np.empty(particles, dtype=np.single)
x1 = np.empty(particles, dtype=np.single)
y1 = np.empty(particles, dtype=np.single)
z1 = np.empty(particles, dtype=np.single)
x2 = np.empty(particles, dtype=np.single)
y2 = np.empty(particles, dtype=np.single)
z2 = np.empty(particles, dtype=np.single)
x3 = np.empty(particles, dtype=np.single)
y3 = np.empty(particles, dtype=np.single)
z3 = np.empty(particles, dtype=np.single)

# Center of mass of each frame
cm = np.empty((n_runs, n_frames, 3), dtype=float)

# Structure factor variance for each difference in times
variance = np.zeros((n_stypes, n_samples, n_q, 3), dtype=float)

# Normalization factor for structure factor variance indices
norm = np.zeros(n_samples, dtype=np.int64)

# W function values for each particle and for both initial and end
# values (if wtypes.ima, component-wise)
if wtype == wtypes.ima:
  w = np.empty((2, 3, particles), dtype=float)
elif wtype == wtypes.theta:
  w = np.empty((2, particles), dtype=int)
else:
  w = np.empty((2, particles), dtype=float)

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

# Accumulates squared values of structure factor component across runs.
# First dimension is stype (total or self), second is q value index,
# third is spatial dimension.
a2_accum = np.empty((2, n_q, 3), dtype=complex)

# Accumulates values of structure factor component across runs, only
# used for q=0.0 and for total structure factor.
a_accum = np.empty(3, dtype=complex)
b_accum = np.empty(3, dtype=complex)

# Special case accumulator of w values across runs, needed for
# combination of q=0.0 and self part.
if wtype == wtypes.ima:
  a_accum_s = np.empty(3, particles, dtype=complex)
  b_accum_s = np.empty(3, particles, dtype=complex)
else:
  a_accum_s = np.empty(particles, dtype=complex)
  b_accum_s = np.empty(particles, dtype=complex)

def calculate_w(wa, run, xa0, ya0, za0, xa1, ya1, za1, index1, index2):
  # Get values for start of w function
  which_file = np.searchsorted(fileframes, index1, side="right") - 1
  offset = index1 + fileframes[which_file]
  dcdfiles[run][which_file].gdcdp(x, y, z, offset)

  # Get values for end of w function
  which_file = np.searchsorted(fileframes, index2, side="right") - 1
  offset = index2 - fileframes[which_file]
  dcdfiles[run][which_file].gdcdp(x1, y1, z1, offset)

  # Correct for center of mass
  xa0 -= cm[run][index1][0]
  ya0 -= cm[run][index1][1]
  za0 -= cm[run][index1][2]
  xa1 -= cm[run][index2][0]
  ya1 -= cm[run][index2][1]
  za1 -= cm[run][index2][2]

  # Calculate w function for each particle
  if wtype == wtypes.theta:
    wa = np.less((xa1 - xa0)**2 +
                 (ya1 - ya0)**2 +
                 (za1 - za0)**2, radius**2).astype(int)
  elif wtype == wtypes.gauss:
    wa = np.exp(-((xa1 - xa0)**2 +
                  (ya1 - ya0)**2 +
                  (za1 - za0)**2)/(2 * gscale**2))
  elif wtype == wtypes.exp:
    wa = np.exp(-np.sqrt((xa1 - xa0)**2 +
                         (ya1 - ya0)**2 +
                         (za1 - za0)**2)/sscale)
  elif wtype == wtypes.ima:
    # This may be a vector in the future, for now just take
    # component-wise with same value
    wa[0] = np.cos(kvec * (xa1 - xa0))
    wa[1] = np.cos(kvec * (ya1 - ya0))
    wa[2] = np.cos(kvec * (za1 - za0))

# Iterate over starting points for structure factor
for i in np.arange(0, n_frames, framediff):
  # Iterate over ending points for structure factor and add to
  # accumulated structure factor, making sure to only use indices
  # which are within the range of the files. j is used as t_b.
  for index, j in enumerate(samples):
    if j < (tc - i) or j >= (n_frames - i - tb):
      continue

    # Clear run accumulators. j is used to indicate a python complex
    # number, not the variable j.
    a2_accum[:] = 0.0+0.0j
    a_accum[:] = 0.0+0.0j
    a_accum_s[:] = 0.0+0.0j

    # Iterate over files
    for k in range(0, n_runs):
      # Calculate w values for t3 and t4
      calculate_w(w[0], k, x0, y0, z0, x1, y1, z1, i, i + tb - tc)

      # Calculate w values for t1 and t2
      calculate_w(w[1], k, x2, y2, z2, x3, y3, z3, i + j - tc, i + j + tb)

      for qindex, q in enumerate(qs):
        # Calculate sums of w functions multiplied by imaginary
        # exponentials. j in these expressions is used to indicate
        # python complex numbers, not the variable j. This is used for
        # the total s calculation, not the self part.
        if wtype == wtypes.ima:
          ab_accum[stypes.total.value][qindex][0] += np.sum(w[0][0] * np.exp(-1j * q * x0)) * np.sum(w[1][0] * np.exp(-1j * q * x2))
          ab_accum[stypes.total.value][qindex][1] += np.sum(w[0][1] * np.exp(-1j * q * y0)) * np.sum(w[1][1] * np.exp(-1j * q * y2))
          ab_accum[stypes.total.value][qindex][2] += np.sum(w[0][2] * np.exp(-1j * q * z0)) * np.sum(w[1][2] * np.exp(-1j * q * z2))
        else:
          ab_accum[stypes.total.value][qindex][0] += np.sum(w[0] * np.exp(-1j * q * x0)) * np.sum(w[1] * np.exp(-1j * q * x2))
          ab_accum[stypes.total.value][qindex][1] += np.sum(w[0] * np.exp(-1j * q * y0)) * np.sum(w[1] * np.exp(-1j * q * y2))
          ab_accum[stypes.total.value][qindex][2] += np.sum(w[0] * np.exp(-1j * q * z0)) * np.sum(w[1] * np.exp(-1j * q * z2))

        # Broadcasts correctly regardless of dimensionality of w, as
        # dimensions of a_accum_s and w match.
        if qindex == 0:
          a_accum_s += w[0]
          b_accum_s += w[1]

    # Normalize accumulators by number of runs to obtain expectation
    # values
    ab_accum /= n_runs
    a_accum_s /= n_runs
    b_accum_s /= n_runs

    # Calculate the variance for the current index and add it to the
    # accumulator entry corresponding to the value of t_b
    variance[stypes.total.value][index] += ab_accum[stypes.total.value].real / particles
    variance[stypes.self.value][index] += ab_accum[stypes.self.value].real / particles

    # Case for q=0.0, where w value of each particle (stored in
    # a_accum_s) must be used in order to find the term to subtract to
    # find the variance. The summing cannot be done inside the run
    # loop, as averaging over runs must be done before multiplication
    # of terms.
    if wtype == wtypes.ima:
      variance[stypes.total.value][index][0] -= (np.sum(a_accum_s, axis=1)*np.sum(b_accum_s, axis=1)).real / particles
      variance[stypes.self.value][index][0] -= np.sum(a_accum_s * b_accum_s, axis=1).real / particles
    else:
      variance[stypes.total.value][index][0] -= (np.sum(a_accum_s)*np.sum(b_accum_s)).real / particles
      variance[stypes.self.value][index][0] -= np.sum(a_accum_s * b_accum_s).real / particles

    # Accumulate the normalization value for this sample offset, which
    # we will use later in computing the mean value for each t_b
    norm[index] += 1

  print("Processed frame %d" %(i + start + 1), file=sys.stderr)

print("#dt = %f" %framediff)

if wtype == wtypes.theta:
  print("#w function type: Threshold")
  print("#a = %f" %radius)
elif wtype == wtypes.gauss:
  print("#w function type: Gaussian")
  print("#a = %f" %gscale)
elif wtype == wtypes.exp:
  print("#w function type: Single Exponential")
  print("#a = %f" %sscale)
elif wtype == wtypes.ima:
  print("#w function type: Imaginary Exponential")
  print("#k = %f" %kvec)

# Find the distinct component of the variance by subtracting the self
# part from the total.
variance[stypes.distinct.value] = variance[stypes.total.value] - variance[stypes.self.value]

# Normalize the accumulated values, thereby obtaining averages over
# each pair of frames
variance /= norm.reshape((n_samples, 1, 1))

for stype in stypes:
  if stype == stypes.total:
    label = "total"
  elif stype == stypes.self:
    label = "self"
  elif stype == stypes.distinct:
    label = "distinct"
  for qindex, q in enumerate(qs):
    for i in range(0, n_samples):
      time_tb = samples[i] * timestep
      var = variance[stype.value][i][qindex]
      # Print stype, t_a, q value, x, y, and z averages, number of
      # frame sets contributing to such average, and frame difference
      # corresponding to t_a
      print("%s %f %f %f %f %f %d %d" %(label, q, time_tb, var[0], var[1], var[2], norm[i], samples[i]))
