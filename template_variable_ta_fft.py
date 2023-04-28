import numpy as np
from numpy import fft
import pydcd
import sys
import math
import getopt
import enum

# Import functionality from local library directory
import lib.opentraj
import lib.progression
import lib.frame
import lib.wcalc

def usage():
  print("Arguments:", file=sys.stderr)
  lib.opentraj.usage()
  print("-d Spacing between initial times as well as lag values (dt)",
        "-x Dimensionality of FFT matrix, length in each dimension in addition to 0",
        "-y Box size in each dimension (assumed to be cubic, required)",
        "-b Average interval in frames (t_b)",
        "-c Difference between intervals in frames (t_c)",
        "-h Print usage",
        sep="\n", file=sys.stderr)
  lib.frame.usage()
  lib.progression.usage()
  lib.wcalc.usage()

try:
  opts, args = getopt.gnu_getopt(sys.argv[1:], "s:d:x:y:b:c:jh" +
                                               lib.opentraj.shortopts +
                                               lib.progression.shortopts +
                                               lib.frame.shortopts +
                                               lib.wcalc.shortopts,
                                               lib.progression.longopts)
except getopt.GetoptError as err:
  print(err, file=sys.stderr)
  usage()
  sys.exit(1)

class stypes(enum.Enum):
  total = 0
  self = 1
  distinct = 2
  totalstd = 3
  selfstd = 4
  distinctstd = 5
n_stypes = len(stypes)

# Last frame number to use for initial times
initend = None
# Spacing between initial times (dt)
framediff = 10
# Limit of number of Fourier transform vector constants (including q=0)
size_fft = None
# User-defined value of dimension of box, assumed to be cubic
box_size = None
# Average length of intervals (t_b)
tb = 1
# Half difference between length of initial and end intervals (t_c)
tc = 0
# Progression specification/generation object for lags
prog = lib.progression.prog()
# Trajectory set opening object
trajset = lib.opentraj.trajset()
# Frame reading object
frames = lib.frame.frames(trajset)
# w function calculation object
wcalc = lib.wcalc.wcalc(frames)

for o, a in opts:
  if o == "-h":
    usage()
    sys.exit(0)
  elif o == "-s":
    start = int(a) - 1
  elif o == "-d":
    framediff = int(a)
  elif o == "-x":
    size_fft = int(a) + 1
  elif o == "-y":
    box_size = float(a)
  elif o == "-b":
    tb = int(a)
  elif o == "-c":
    tc = int(a)
  elif o == "-j":
    print("-j is default, no need to specify", file=sys.stderr)
  elif trajset.catch_opt(o, a) == True:
    pass
  elif prog.catch_opt(o, a) == True:
    pass
  elif frames.catch_opt(o, a) == True:
    pass
  elif wcalc.catch_opt(o, a) == True:
    pass

# Verify correctness of parameters
wcalc.verify()

if box_size == None:
  raise RuntimeError("Must define box size dimensions")

if trajset.n_runs <= 1:
  raise RuntimeError("Must have at least 2 runs")

if size_fft == None:
  raise RuntimeError("Must specify size for FFT matrix")

if prog.progtype == None:
  raise RuntimeError("Must specify interval increase progression type")

# Open trajectory files
trajset.opentraj_multirun("run", "traj", 1, True)

# Print basic properties shared across the files
print("#nset: %d" %trajset.fileframes[-1])
print("#N: %d" %trajset.fparticles)
print("#timestep: %f" %trajset.timestep)
print("#tbsave: %f" %trajset.tbsave)

# Spatial size of individual cell for FFT
cell = box_size / size_fft

# End of set of frames to use for initial times
if initend == None:
  initend = trajset.fileframes[-1]
else:
  if initend > trajset.fileframes[-1]:
    raise RuntimeError("End initial time frame beyond set of frames")

# Prepare frames object for calculation
frames.prepare()

# Largest possible positive and negative lags
prog.max_val = frames.n_frames - 1 - max(tb, tb - tc)
prog.min_val = -((framediff * ((frames.n_frames - 1 - (tb - 2 * tc)) // framediff)) - tc)

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
x2 = np.empty(frames.particles, dtype=np.single)
y2 = np.empty(frames.particles, dtype=np.single)
z2 = np.empty(frames.particles, dtype=np.single)

# Bins for total calculation.
a_bins = np.zeros((size_fft, size_fft, size_fft), dtype=np.float64)
b_bins = np.zeros((size_fft, size_fft, size_fft), dtype=np.float64)
self_bins = np.empty((size_fft, size_fft, size_fft), dtype=np.float64)

# Temporary value for each run to allow for calculation of each run's
# self component of s4
run_self_s4 = np.empty((size_fft, size_fft, (size_fft // 2) + 1), dtype=np.float64)

# Temporary value for each run to allow for calculation of each run's
# total component of s4
run_total_s4 = np.empty((size_fft, size_fft, (size_fft // 2) + 1), dtype=np.float64)

# Structure factor variance for each difference in times. The second
# and third fft dimensions hold values for negative vectors. Since all
# inputs are real, this is not required for the first fft dimension.
s4 = np.zeros((n_stypes, size_fft, size_fft, (size_fft // 2) + 1), dtype=np.float64)

# W function values for each particle and for both initial and end
# values
if wcalc.wtype == lib.wcalc.wtypes.theta:
  w = np.empty((2, frames.particles), dtype=np.int8)
else:
  w = np.empty((2, frames.particles), dtype=np.float64)

print("#dt = %d" %framediff)
print("#n_lags = %d" %lags.size)
print("#t_b = %d" %tb)
print("#t_c = %d" %tc)

if wcalc.wtype == lib.wcalc.wtypes.theta:
  print("#w function type: Threshold")
  print("#a = %f" %wcalc.radius)
elif wcalc.wtype == lib.wcalc.wtypes.gauss:
  print("#w function type: Gaussian")
  print("#a = %f" %wcalc.gscale)
elif wcalc.wtype == lib.wcalc.wtypes.exp:
  print("#w function type: Single Exponential")
  print("#a = %f" %wcalc.sscale)

# Find center of mass for each frame
print("Finding centers of mass for frames", file=sys.stderr)
frames.generate_cm(x0, y0, z0)

# S4 calcuation

print("Entering S4 calculation", file=sys.stderr)

# Iterate over lags (t_a)
for index, ta in enumerate(lags):
  # Clear lag accumulator
  s4[:, :, :, :] = 0.0

  # Normalization factor for number of sets contributing to lag value
  norm = 0

  # Iterate over runs (FFT will be averaged over runs)
  for i in np.arange(0, trajset.n_runs):
    # Clear run accumulators
    run_self_s4[:, :, :] = 0.0
    run_total_s4[:, :, :] = 0.0

    # Iterate over starting points for structure factor
    for j in np.arange(0, initend - frames.start, framediff):
      # Use only indices that are within range
      if (ta < (tc - j) or
          ta - frames.n_frames >= (tc - j) or
          ta < (-tb - j) or
          ta - frames.n_frames >= (-tb - j) or
          j < (tc - tb) or
          j - frames.n_frames >= (tc - tb)):
        continue

      # Get particle coordinates and calculate w values for first and
      # second intervals
      wcalc.calculate_w(w[0], j, x0, y0, z0, j + tb - tc, x1, y1, z1, i)
      wcalc.calculate_w(w[1], j + ta - tc, x2, y2, z2, j + ta + tb, x1, y1, z1, i)

      # Convert particle positions into bin numbers and wrap for
      # binning
      x0i = (x0 // cell).astype(np.int64) % size_fft
      y0i = (y0 // cell).astype(np.int64) % size_fft
      z0i = (z0 // cell).astype(np.int64) % size_fft
      x2i = (x2 // cell).astype(np.int64) % size_fft
      y2i = (y2 // cell).astype(np.int64) % size_fft
      z2i = (z2 // cell).astype(np.int64) % size_fft

      # Sort first interval w values into bins for FFT
      a_bins, dummy = np.histogramdd((x0i, y0i, z0i), bins=size_fft, range=((-0.5, size_fft - 0.5), ) * 3, weights=w[0])

      # Sort second interval w values into bins for FFT
      b_bins, dummy = np.histogramdd((x2i, y2i, z2i), bins=size_fft, range=((-0.5, size_fft - 0.5), ) * 3, weights=w[1])

      # Calculate total part of S4
      run_total_s4 += fft.fftshift((fft.rfftn(a_bins) * np.conjugate(fft.rfftn(b_bins))).real, axes=(0, 1)) / frames.particles

      # Convert particle bin numbers into bin number differences
      # between starting frames of first and second intervals for self
      # part calculation and wrap for binning. Since total exponential
      # is negative, must use reverse difference with
      # positive-exponential FFT.
      x0i = (x0i - x2i) % size_fft
      y0i = (y0i - y2i) % size_fft
      z0i = (z0i - z2i) % size_fft

      # Multiply w values for different intervals together for self
      # bins
      w[0] *= w[1]

      # Bin multiplied w values according to coordinate differences
      self_bins, dummy = np.histogramdd((x0i, y0i, z0i), bins=size_fft, range=((-0.5, size_fft - 0.5), ) * 3, weights=w[0])

      # Perform FFT, thereby calculating self S4 for current index
      run_self_s4 += fft.fftshift(fft.rfftn(self_bins).real, axes=(0, 1)) / frames.particles

      # Accumulate the normalization value for this lag, which will be
      # used later in computing the mean S4 quantities for each lag
      if i == 0:
        norm += 1

    # Calculate distinct part of S4 for current run
    run_distinct_s4 = run_total_s4 - run_self_s4

    # Normalize the accumulated values, thereby obtaining averages over
    # each pair of frames
    run_total_s4 /= norm
    run_self_s4 /= norm
    run_distinct_s4 /= norm

    # Accumulate total, self, and distinct averages for run
    s4[stypes.total.value] += run_total_s4
    s4[stypes.self.value] += run_self_s4
    s4[stypes.distinct.value] += run_distinct_s4

    # Accumulate squares of total, self, and distinct averages for run,
    # holding variances for eventual calculation of standard deviation
    s4[stypes.totalstd.value] += run_total_s4**2
    s4[stypes.selfstd.value] += run_self_s4**2
    s4[stypes.distinctstd.value] += run_distinct_s4**2

  # Normalize S4 values across runs
  s4 /= trajset.n_runs

  # Calculate standard deviations from normalized variances over runs
  s4[stypes.totalstd.value] = np.sqrt(np.maximum(0.0, s4[stypes.totalstd.value] - s4[stypes.total.value]**2) / (trajset.n_runs - 1))
  s4[stypes.selfstd.value] = np.sqrt(np.maximum(0.0, s4[stypes.selfstd.value] - s4[stypes.self.value]**2) / (trajset.n_runs - 1))
  s4[stypes.distinctstd.value] = np.sqrt(np.maximum(0.0, s4[stypes.distinctstd.value] - s4[stypes.distinct.value]**2) / (trajset.n_runs - 1))

  # Print results for current lag
  time_ta = lags[index] * trajset.timestep * trajset.tbsave
  for i in range(0, size_fft):
    for j in range(0, size_fft):
      for k in range(0, (size_fft // 2) + 1):
        # Print t_a, x, y, and z components of fft frequency, total,
        # self, and distinct averages, standard deviations across runs
        # of total, self, and distinct averages, number of frame sets
        # contributing to such average, and frame difference
        # corresponding to t_a
        print("%f %f %f %f %f %f %f %f %f %f %d %d" %(time_ta,
                                                      (i-size_fft//2)*2*math.pi/box_size,
                                                      (j-size_fft//2)*2*math.pi/box_size,
                                                      k*2*math.pi/box_size,
                                                      s4[stypes.total.value][i][j][k],
                                                      s4[stypes.self.value][i][j][k],
                                                      s4[stypes.distinct.value][i][j][k],
                                                      s4[stypes.totalstd.value][i][j][k],
                                                      s4[stypes.selfstd.value][i][j][k],
                                                      s4[stypes.distinctstd.value][i][j][k],
                                                      norm,
                                                      lags[index]))
