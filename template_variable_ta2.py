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
        "-d Spacing between initial times (dt)",
#        "-x Number of Fourier transform vector constants to used in addition to q=0",
        "-y Box size in each dimension (assumed to be cubic, required)"
        "-b Average interval in frames (t_b)",
        "-c Difference between intervals in frames (t_c)",
        "-p Limit number of particles to analyze",
        "-h Print usage",
        "Interval increase progression (last specified is used):"
        "-f Flenner-style periodic-exponential-increasing increment (iterations: 50, power: 5)",
        "-g Geometric spacing progression, selectively dropped to fit on integer frames (argument is number of lags)",
        "-l Linear spacing progression, uses same spacing as initial times (no argument)",
        "-m Mirror lags to have negative values",
        "w function types (last specified is used, must be specified):",
        "-t Theta function threshold (argument is threshold radius)",
        "-u Double negative exponential/Gaussian (argument is exponential length)",
        "-e Single negative exponential (argument is exponential length)",
        sep="\n", file=sys.stderr)

try:
  opts, args = getopt.gnu_getopt(sys.argv[1:], "n:r:s:d:x:y:b:c:p:jhfg:lmt:u:e:")
except getopt.GetoptError as err:
  print(err, file=sys.stderr)
  usage()
  sys.exit(1)

class progtypes(enum.Enum):
  flenner = 1
  geometric = 2
  linear = 3

class wtypes(enum.Enum):
  none = 1
  theta = 2
  gauss = 3
  exp = 4

class stypes(enum.Enum):
  total = 0
  self = 1
  distinct = 2
n_stypes = len(stypes)

# Total number of trajectory files
n_files = 1
# Total number of run folders. 0 means not specified.
n_runs = 0
# What frame number to start on
start = 0
# Spacing between initial times (dt)
framediff = 10
# Number of Fourier transform vector constants (including q=0)
n_q = 1
# User-defined value of dimension of box, assumed to be cubic
box_size = None
# Average length of intervals (t_b)
tb = 1
# Half difference between length of initial and end intervals (t_c)
tc = 0
# Number of particles to limit analysis to
particle_limit = None
# Type of progression to increase time interval by
progtype = progtypes.flenner
# Whether to use mirrored negative lags
negative_lags = False
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
  elif o == "-s":
    start = int(a) - 1
  elif o == "-d":
    framediff = int(a)
  elif o == "-x":
    n_q = int(a) + 1
  elif o == "-y":
    box_size = float(a)
  elif o == "-b":
    tb = int(a)
  elif o == "-c":
    tc = int(a)
  elif o == "-p":
    particle_limit = int(a)
  elif o == "-f":
    progtype = progtypes.flenner
  elif o == "-g":
    progtype = progtypes.geometric
    geom_num = int(a)
  elif o == "-l":
    progtype = progtypes.linear
  elif o == "-m":
    negative_lags = True
  elif o == "-j":
    print("-j is default, no need to specify", file=sys.stderr)
  elif o == "-t":
    wtype = wtypes.theta
    radius = float(a)
  elif o == "-u":
    wtype = wtypes.gauss
    sscale = float(a)
  elif o == "-e":
    wtype = wtypes.exp
    gscale = float(a)

if wtype == wtypes.none:
  raise RuntimeError("No w function type specified")

if box_size == None:
  raise RuntimeError("Must define box size dimensions")

if n_runs <= 1:
  raise RuntimeError("Must have at least 2 runs")

# Open trajectory files
dcdfiles, fileframes, fparticles, timestep, tbsave = lib.opentraj.opentraj_multirun(n_runs, "run", n_files, "traj", 1, True)

# Limit particles if necessary
if particle_limit == None:
  particles = fparticles
else:
  if particle_limit < fparticles:
    particles = particle_limit
  else:
    particles = fparticles
    particle_limit = None

# Print basic properties shared across the files
print("#nset: %d" %fileframes[-1])
print("#N: %d" %particles)
print("#timestep: %f" %timestep)
print("#tbsave: %f" %tbsave)

# Number of frames in each run to analyze
n_frames = fileframes[-1] - start

# Largest possible lag
max_lag = n_frames - tb - 1

# Largest possible negative lag
if negative_lags == False:
  max_neg_lag = 0
else:
  max_neg_lag = -max_lag

if progtype == progtypes.flenner:
  # Construct list of frame difference numbers for sampling according
  # to a method of increasing spacing
  magnitude = -1
  frames_beyond_magnitude = max_lag
  while frames_beyond_magnitude >= 50 * 5**(magnitude + 1):
    magnitude += 1
    frames_beyond_magnitude -= 50 * 5**magnitude

  lags_beyond_magnitude = frames_beyond_magnitude // 5**(magnitude + 1)

  n_lags = 1 + (50 * (magnitude + 1)) + lags_beyond_magnitude

  # Allocate that array
  lags = np.empty(n_lags, dtype=np.int64)

  # Efficiently fill the array
  lags[0] = 0
  last_lag_number = 0
  for i in range(0, magnitude + 1):
    lags[1 + 50 * i : 1 + 50 * (i + 1)] = last_lag_number + np.arange(5**i , 51 * 5**i, 5**i)
    last_lag_number += 50 * 5**i
  lags[1 + 50 * (magnitude + 1) : n_lags] = last_lag_number + np.arange(5**(magnitude + 1), (lags_beyond_magnitude + 1) * 5**(magnitude + 1), 5**(magnitude + 1))

elif progtype == progtypes.geometric:
  # Largest power of geom_base that will be able to be read
  geom_base = max_lag**(1.0 / geom_num)

  # Create array of lags following geometric progression, with flooring
  # to have lags adhere to integer boundaries, removing duplicate
  # numbers, and prepending 0
  lags = np.insert(np.unique(np.floor(np.logspace(0, geom_num, num=(geom_num + 1), base=geom_base)).astype(np.int64)), 0, 0)

elif progtype == progtypes.linear:
  # Create evenly spaced array of lag values with same spacing as
  # initial times (framediff)
  lags = np.arange(0, max_lag + 1, step=framediff)

if negative_lags == True:
  # Mirror lags array, making sure not to duplicate the 0 value at
  # index 0
  lags = np.concatenate((np.flip(-lags[1:]), lags))

# Number of lag values
n_lags = lags.size

# Array with progression of q values to use, with 0.0 always at index 0.
# All of these create integral number of wave periods inside the box.
qs = np.linspace(0.0, (n_q - 1) * 2 * math.pi / box_size, num=n_q)

# If particles limited, must be read into different array
if particle_limit != None:
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
x2 = np.empty(particles, dtype=np.single)
y2 = np.empty(particles, dtype=np.single)
z2 = np.empty(particles, dtype=np.single)
x3 = np.empty(particles, dtype=np.single)
y3 = np.empty(particles, dtype=np.single)
z3 = np.empty(particles, dtype=np.single)

# Center of mass of each frame
cm = [np.empty((n_frames, 3), dtype=np.float64)] * n_runs

# Structure factor variance for each difference in times
s4 = np.zeros((n_stypes, n_lags, n_q, 3), dtype=np.float64)

# Normalization factor for structure factor variance indices
norm = np.zeros(n_lags, dtype=np.int64)

# W function values for each particle and for both initial and end
# values
if wtype == wtypes.theta:
  w = np.empty((2, particles), dtype=np.int8)
else:
  w = np.empty((2, particles), dtype=np.float8)

# Find center of mass of each frame
print("Finding centers of mass for frames", file=sys.stderr)
for i in range(0, n_frames):
  which_file = np.searchsorted(fileframes, start + i, side="right") - 1
  offset = start + i - fileframes[which_file]
  for j in range(0, n_runs):
    if particle_limit == None:
      dcdfiles[j][which_file].gdcdp(x0, y0, z0, offset)
      cm[j][i][0] = np.mean(x0)
      cm[j][i][1] = np.mean(y0)
      cm[j][i][2] = np.mean(z0)
    else:
      dcdfiles[j][which_file].gdcdp(x, y, z, offset)
      cm[j][i][0] = np.mean(x[:particles])
      cm[j][i][1] = np.mean(y[:particles])
      cm[j][i][2] = np.mean(z[:particles])

# Accumulates squared values of structure factor component across runs.
# First dimension is stype (total or self), second is q value index,
# third is spatial dimension.
ab_accum = np.empty((2, n_q, 3), dtype=np.float64)

# Special case accumulator of w values across runs, needed for
# combination of q=0.0 and self part. Only real values are ever
# accumulated to these, so they can be float.
a_accum_s = 0.0
b_accum_s = 0.0

def calculate_w(wa, run, xa0, ya0, za0, xa1, ya1, za1, index1, index2):
  # Get values for start of w function
  which_file = np.searchsorted(fileframes, index1, side="right") - 1
  offset = index1 - fileframes[which_file]
  if particle_limit == None:
    dcdfiles[run][which_file].gdcdp(xa0, ya0, za0, offset)
  else:
    dcdfiles[run][which_file].gdcdp(x, y, z, offset)
    xa0[:] = x[:particles]
    ya0[:] = y[:particles]
    za0[:] = z[:particles]

  # Get values for end of w function
  which_file = np.searchsorted(fileframes, index2, side="right") - 1
  offset = index2 - fileframes[which_file]
  if particle_limit == None:
    dcdfiles[run][which_file].gdcdp(xa1, ya1, za1, offset)
  else:
    dcdfiles[run][which_file].gdcdp(x, y, z, offset)
    xa1[:] = x[:particles]
    ya1[:] = y[:particles]
    za1[:] = z[:particles]

  # Correct for center of mass
  xa0 -= cm[run][index1][0]
  ya0 -= cm[run][index1][1]
  za0 -= cm[run][index1][2]
  xa1 -= cm[run][index2][0]
  ya1 -= cm[run][index2][1]
  za1 -= cm[run][index2][2]

  # Calculate w function for each particle
  if wtype == wtypes.theta:
    wa[:] = np.less((xa1 - xa0)**2 +
                    (ya1 - ya0)**2 +
                    (za1 - za0)**2, radius**2).astype(np.int8, copy=False)
  elif wtype == wtypes.gauss:
    wa[:] = np.exp(-((xa1 - xa0)**2 +
                     (ya1 - ya0)**2 +
                     (za1 - za0)**2)/(2 * gscale**2))
  elif wtype == wtypes.exp:
    wa[:] = np.exp(-np.sqrt((xa1 - xa0)**2 +
                            (ya1 - ya0)**2 +
                            (za1 - za0)**2)/sscale)

# Iterate over starting points for structure factor
for i in np.arange(0, n_frames - (tb - tc), framediff):
  # Iterate over ending points for structure factor and add to
  # accumulated structure factor, making sure to only use indices
  # which are within the range of the files. j is used as t_a.
  for index, ta in enumerate(lags):
    if ta < (tc - i) or ta >= (n_frames - i - tb):
      continue

    # Clear run accumulators.
    ab_accum[:] = 0.0
    a_accum_s = 0.0
    b_accum_s = 0.0

    # Iterate over files
    for k in range(0, n_runs):
      # Calculate w values for t3 and t4
      calculate_w(w[0], k, x0, y0, z0, x1, y1, z1, i, i + tb - tc)

      # Calculate w values for t1 and t2
      calculate_w(w[1], k, x2, y2, z2, x3, y3, z3, i + ta - tc, i + ta + tb)

      for qindex, q in enumerate(qs):
        # Calculate and accumulate values for total s4. Simulate
        # complex multiplication, since using complex numbers with
        # numpy imposes a very large performance penalty.
        ab_accum[stypes.total.value][qindex][0] += (np.sum(w[0] * np.cos(q * x0)) * np.sum(w[1] * np.cos(q * x2)) -
                                                    np.sum(w[0] * np.sin(-q * x0)) * np.sum(w[1] * np.sin(q * x2)))
        ab_accum[stypes.total.value][qindex][1] += (np.sum(w[0] * np.cos(q * y0)) * np.sum(w[1] * np.cos(q * y2)) -
                                                    np.sum(w[0] * np.sin(-q * y0)) * np.sum(w[1] * np.sin(q * y2)))
        ab_accum[stypes.total.value][qindex][2] += (np.sum(w[0] * np.cos(q * z0)) * np.sum(w[1] * np.cos(q * z2)) -
                                                    np.sum(w[0] * np.sin(-q * z0)) * np.sum(w[1] * np.sin(q * z2)))

        # Calculate and accumulate values for self part of s2
        ab_accum[stypes.self.value][qindex][0] += np.sum(w[0] * w[1] * np.cos(q * (x0 - x2)))
        ab_accum[stypes.self.value][qindex][1] += np.sum(w[0] * w[1] * np.cos(q * (y0 - y2)))
        ab_accum[stypes.self.value][qindex][2] += np.sum(w[0] * w[1] * np.cos(q * (z0 - z2)))

        if qindex == 0:
          a_accum_s += np.sum(w[0])
          b_accum_s += np.sum(w[1])

    # Normalize accumulators by number of runs to obtain expectation
    # values
    ab_accum /= n_runs
    a_accum_s /= n_runs
    b_accum_s /= n_runs

    # Calculate the variance for the current index and add it to the
    # accumulator entry corresponding to the value of t_b
    s4[stypes.total.value][index] += ab_accum[stypes.total.value] / particles
    s4[stypes.self.value][index] += ab_accum[stypes.self.value] / particles

    # Case for q=0.0, where w value of each particle (stored in
    # a_accum_s) must be used in order to find the term to subtract to
    # find the variance.
    s4[stypes.total.value][index][0] -= (a_accum_s * b_accum_s) / particles
    s4[stypes.self.value][index][0] -= (a_accum_s * b_accum_s) / particles**2

    # Accumulate the normalization value for this lag value, which
    # we will use later in computing the mean value for each t_b
    norm[index] += 1

  print("Processed frame %d" %(i + start + 1), file=sys.stderr)

print("#dt = %d" %framediff)
print("#t_b = %d" %tb)
print("#t_c = %d" %tc)

if wtype == wtypes.theta:
  print("#w function type: Threshold")
  print("#a = %f" %radius)
elif wtype == wtypes.gauss:
  print("#w function type: Gaussian")
  print("#a = %f" %gscale)
elif wtype == wtypes.exp:
  print("#w function type: Single Exponential")
  print("#a = %f" %sscale)

# Find the distinct component of the variance by subtracting the self
# part from the total.
s4[stypes.distinct.value] = s4[stypes.total.value] - s4[stypes.self.value]

# Normalize the accumulated values, thereby obtaining averages over
# each pair of frames
s4 /= norm.reshape((n_lags, 1, 1))

for stype in stypes:
  if stype == stypes.total:
    label = "total"
  elif stype == stypes.self:
    label = "self"
  elif stype == stypes.distinct:
    label = "distinct"

  for qindex, q in enumerate(qs):
    for i in range(0, n_lags):
      time_ta = lags[i] * timestep * tbsave
      s4i = s4[stype.value][i][qindex]
      # Print stype, t_a, q value, x, y, and z averages, number of
      # frame sets contributing to such average, and frame difference
      # corresponding to t_a
      print("%s %f %f %f %f %f %d %d" %(label, q, time_ta, s4i[0], s4i[1], s4i[2], norm[i], lags[i]))
