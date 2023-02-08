import numpy as np
from numpy import fft
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
        "-d Spacing between initial times as well as lag values (dt)",
        "-x Dimensionality of FFT matrix, length in each dimension in addition to 0",
        "-y Box size in each dimension (assumed to be cubic, required)"
        "-b Average interval in frames (t_b)",
        "-c Difference between intervals in frames (t_c)",
        "-p Limit number of particles to analyze",
        "-h Print usage",
        "w function types (last specified is used, must be specified):",
        "-t Theta function threshold (argument is threshold radius)",
        "-u Double negative exponential/Gaussian (argument is exponential length)",
        "-e Single negative exponential (argument is exponential length)",
        sep="\n", file=sys.stderr)

try:
  opts, args = getopt.gnu_getopt(sys.argv[1:], "n:r:s:d:x:y:b:c:p:jht:u:e:")
except getopt.GetoptError as err:
  print(err, file=sys.stderr)
  usage()
  sys.exit(1)

class ttypes(enum.Enum):
  t1 = 0
  t2 = 1
  t3 = 2
  t4 = 3
n_ttypes = len(ttypes)

class wtypes(enum.Enum):
  theta = 1
  gauss = 2
  exp = 3

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
# Limit of number of Fourier transform vector constants (including q=0)
size_fft = None
# User-defined value of dimension of box, assumed to be cubic
box_size = None
# Average length of intervals (t_b)
tb = 1
# Half difference between length of initial and end intervals (t_c)
tc = 0
# Number of particles to limit analysis to
particle_limit = None
# Type of w function to use
wtype = None

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
    size_fft = int(a) + 1
  elif o == "-y":
    box_size = float(a)
  elif o == "-b":
    tb = int(a)
  elif o == "-c":
    tc = int(a)
  elif o == "-p":
    particle_limit = int(a)
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

if wtype == None:
  raise RuntimeError("No w function type specified")

if box_size == None:
  raise RuntimeError("Must define box size dimensions")

if n_runs <= 1:
  raise RuntimeError("Must have at least 2 runs")

if size_fft == None:
  raise RuntimeError("Must specify size for FFT matrix")

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

# Spatial size of individual cell for FFT
cell = box_size / size_fft

# Number of frames in each run to analyze
n_frames = fileframes[-1] - start

# Number of initial times
n_init = (n_frames - 1 - (tb - tc)) // framediff + 1

# Largest possible positive and negative lags
max_pos_lag = ((n_frames - tb - 1) // framediff) * framediff
max_neg_lag = -((n_frames - tb - 1 - ((n_frames - 1 - (tb - tc)) % framediff)) // framediff) * framediff

# Create evenly spaced array of lag values with same spacing as
# initial times (framediff)
lags = np.arange(max_neg_lag, max_pos_lag + 1, step=framediff)

# Number of lag values
n_lags = lags.size

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

# Center of mass of each frame
cm = [np.empty((n_init, n_ttypes, 3), dtype=np.float64)] * n_runs

# Bins for total calculation. Bounds of frame numbers extend beyond
# what is required for storage in order for autocorrelation calculation
# to work correctly and not wrap improperly.
a_bins = np.zeros((2 * n_init - 1, size_fft, size_fft, size_fft), dtype=np.float64)
b_bins = np.zeros((2 * n_init - 1, size_fft, size_fft, size_fft), dtype=np.float64)
self_bins = np.empty((size_fft, size_fft, size_fft), dtype=np.float64)

# Accumulator of summed w values for each frame, used for computing
# second 0 vector term of s4 (term_0_2).
a_accum = np.zeros(2 * n_init - 1, dtype=np.float64)
b_accum = np.zeros(2 * n_init - 1, dtype=np.float64)

# Accumulates squared values of structure factor component across runs.
ab_accum = np.empty((size_fft, size_fft, (size_fft // 2) + 1), dtype=np.float64)

# Structure factor variance for each difference in times. The second
# and third fft dimensions hold values for negative vectors. Since all
# inputs are real, this is not required for the first fft dimension.
s4 = np.zeros((n_stypes, n_lags, size_fft, size_fft, (size_fft // 2) + 1), dtype=np.float64)

# Normalization factor for structure factor variance indices
norm = np.zeros(n_lags, dtype=np.int64)

# W function values for each particle and for both initial and end
# values
if wtype == wtypes.theta:
  w = np.empty((2, particles), dtype=np.int8)
else:
  w = np.empty((2, particles), dtype=np.float64)

# Find center of mass of each frame
tlist = ((ttypes.t1.value, 0),
         (ttypes.t2.value, tb - tc),
         (ttypes.t3.value, -tc),
         (ttypes.t4.value, tb))
print("Finding centers of mass for frames", file=sys.stderr)
for i in range(0, n_init):
  for j in range(0, n_runs):
    # Iterate over possible ways this frame index may be used, skipping
    # if index would be outside bounds.
    for ttype, tadd in tlist:
      index = framediff * i + tadd
      if index < 0 or index >= n_frames:
        continue

      which_file = np.searchsorted(fileframes, start + index, side="right") - 1
      offset = start + index - fileframes[which_file]

      if particle_limit == None:
        dcdfiles[j][which_file].gdcdp(x0, y0, z0, offset)
        cm[j][i][ttype][0] = np.mean(x0)
        cm[j][i][ttype][1] = np.mean(y0)
        cm[j][i][ttype][2] = np.mean(z0)
      else:
        dcdfiles[j][which_file].gdcdp(x, y, z, offset)
        cm[j][i][ttype][0] = np.mean(x[:particles])
        cm[j][i][ttype][1] = np.mean(y[:particles])
        cm[j][i][ttype][2] = np.mean(z[:particles])

def calculate_w(wa, run, xa0, ya0, za0, xa1, ya1, za1, i, ttype1, ttype2):
  if ttype1 == ttypes.t1:
    index1 = framediff * i
  elif ttype1 == ttypes.t2:
    index1 = framediff * i + tb - tc
  elif ttype1 == ttypes.t3:
    index1 = framediff * i - tc
  elif ttype1 == ttypes.t4:
    index1 = framediff * i + tb

  if ttype2 == ttypes.t1:
    index2 = framediff * i
  elif ttype2 == ttypes.t2:
    index2 = framediff * i + tb - tc
  elif ttype2 == ttypes.t3:
    index2 = framediff * i - tc
  elif ttype2 == ttypes.t4:
    index2 = framediff * i + tb

  # Get values for start of w function
  which_file = np.searchsorted(fileframes, start + index1, side="right") - 1
  offset = start + index1 - fileframes[which_file]
  if particle_limit == None:
    dcdfiles[run][which_file].gdcdp(xa0, ya0, za0, offset)
  else:
    dcdfiles[run][which_file].gdcdp(x, y, z, offset)
    xa0[:] = x[:particles]
    ya0[:] = y[:particles]
    za0[:] = z[:particles]

  # Get values for end of w function
  which_file = np.searchsorted(fileframes, start + index2, side="right") - 1
  offset = start + index2 - fileframes[which_file]
  if particle_limit == None:
    dcdfiles[run][which_file].gdcdp(xa1, ya1, za1, offset)
  else:
    dcdfiles[run][which_file].gdcdp(x, y, z, offset)
    xa1[:] = x[:particles]
    ya1[:] = y[:particles]
    za1[:] = z[:particles]

  # Correct for center of mass
  xa0 -= cm[run][i][ttype1.value][0]
  ya0 -= cm[run][i][ttype1.value][1]
  za0 -= cm[run][i][ttype1.value][2]
  xa1 -= cm[run][i][ttype2.value][0]
  ya1 -= cm[run][i][ttype2.value][1]
  za1 -= cm[run][i][ttype2.value][2]

  # Calculate w function for each particle
  if wtype == wtypes.theta:
    np.less((xa1 - xa0)**2 +
            (ya1 - ya0)**2 +
            (za1 - za0)**2, radius**2, out=wa)
  elif wtype == wtypes.gauss:
    np.exp(-((xa1 - xa0)**2 +
             (ya1 - ya0)**2 +
             (za1 - za0)**2)/(2 * gscale**2), out=wa)
  elif wtype == wtypes.exp:
    np.exp(-np.sqrt((xa1 - xa0)**2 +
                    (ya1 - ya0)**2 +
                    (za1 - za0)**2)/sscale, out=wa)

# Total S4 calcuation

print("Calculating total part of S4", file=sys.stderr)

# Iterate over runs (FFT will be averaged over runs)
for i in range(0, n_runs):
  for j in range(0, n_init):
    root = framediff * j

    if root + tb - tc < n_frames:
      # Calculate w values for t1 and t2
      calculate_w(w[0], i, x0, y0, z0, x2, y2, z2, j, ttypes.t1, ttypes.t2)

      # Wrap particle coordinates to box. If not done, particles will
      # not be distributed across bins correctly.
      x0 %= box_size
      y0 %= box_size
      z0 %= box_size

      # Bin values for FFT
      a_bins[j], dummy = np.histogramdd((x0, y0, z0), bins=size_fft, range=((0, box_size), ) * 3, weights=w[0])

      # Accumulate for second term of variance
      a_accum[j] += np.sum(w[0])

    if root - tc >= 0 and root + tb < n_frames:
      # Calculate w values for t3 and t4
      calculate_w(w[0], i, x0, y0, z0, x2, y2, z2, j, ttypes.t3, ttypes.t4)

      # Wrap particle coordinates to box. If not done, particles will
      # not be distributed across bins correctly.
      x0 %= box_size
      y0 %= box_size
      z0 %= box_size

      # Bin values for FFT
      b_bins[j], dummy = np.histogramdd((x0, y0, z0), bins=size_fft, range=((0, box_size), ) * 3, weights=w[0])

      # Accumulate for second term of variance
      b_accum[j] += np.sum(w[0])

  # Calculate correlations of first point sums with second point sums.
  # This will later be normalized for number of terms corresponding to
  # each correlation offset. Last spatial axis is halved due to use of
  # rfftn. b_bins must be conjugated for the FFT over the last 3
  # spatial dimensions, but unconjugated for the time dimension. The
  # FFT of a_bins must be conjugated for the time dimension only. The
  # roll followed by flip unconjugates the time axis of b_bins and
  # conjugates the time axis of a_bins.
  s4[stypes.total.value] += fft.fftshift(fft.ifft(np.flip(np.roll(fft.rfftn(a_bins) * np.conjugate(fft.rfftn(b_bins)), -1, axis=0), axis=0), axis=0).real, axes=(0, 1, 2))[n_init-1+(max_neg_lag//framediff):n_init+(max_pos_lag//framediff)] / particles

# Normalize total S4 values across runs
s4[stypes.total.value] /= n_runs

# Normalize second term of variance across runs
a_accum /= n_runs
b_accum /= n_runs

# Used with 0 vector for calculating second term of variance. This will
# later be normalized for number of terms corresponding to each
# correlation offset. a_accum must be conjugated for the correlation.
term_0_2 = fft.fftshift(fft.irfft(np.conjugate(fft.rfft(a_accum)) * fft.rfft(b_accum), n=a_accum.size).real)[n_init-1+(max_neg_lag//framediff):n_init+(max_pos_lag//framediff)] / particles

# Self S4 calculation

print("Calculating self part of S4", file=sys.stderr)

# Iterate over starting points for structure factor
for i in range(0, n_init):
  # Iterate over ending points for structure factor and add to
  # accumulated structure factor, making sure to only use indices
  # which are within the range of the files.
  for index, ta in enumerate(lags):
    if ta < (tc - framediff * i) or ta >= (n_frames - framediff * i - tb):
      continue

    # Clear run accumulators.
    ab_accum[:] = 0.0

    # Iterate over files
    for k in range(0, n_runs):
      # Calculate w values for t3 and t4
      calculate_w(w[0], k, x0, y0, z0, x2, y2, z2, i, ttypes.t1, ttypes.t2)

      # Calculate w values for t1 and t2
      calculate_w(w[1], k, x1, y1, z1, x2, y2, z2, i + ta // framediff, ttypes.t3, ttypes.t4)

      # Align particle coordinates to make sum equivalent to total
      # part and find difference in positions for binning. Since total
      # exponential is negative, must use reverse difference with
      # positive-exponential FFT.
      x0[:] = ((x0 // cell) - (x1 // cell) + 0.5) * cell
      y0[:] = ((y0 // cell) - (y1 // cell) + 0.5) * cell
      z0[:] = ((z0 // cell) - (z1 // cell) + 0.5) * cell

      # Wrap particle coordinates to box. If not done, particles will
      # not be distributed across bins correctly.
      x0 %= box_size
      y0 %= box_size
      z0 %= box_size

      # Multiply w values for calculation of pairs
      w[0] *= w[1]

      # Bin multiplied w values according to coordinate differences
      self_bins, dummy = np.histogramdd((x0, y0, z0), bins=size_fft, range=((0, box_size), ) * 3, weights=w[0])

      # Accumulate S4 FFT
      ab_accum += fft.fftshift(fft.rfftn(self_bins).real, axes=(0, 1))

    # Normalize accumulators by number of runs to obtain expectation
    # values
    ab_accum /= n_runs

    # Calculate the variance for the current index and add it to the
    # accumulator entry corresponding to the value of t_a
    s4[stypes.self.value][index] += ab_accum / particles

    # Accumulate the normalization value for this lag value, which
    # we will use later in computing the mean value for each t_b
    norm[index] += 1

  print("Processed frame %d" %(start + framediff * i + 1), file=sys.stderr)

# Subtract second term of variance from 0 vector terms
s4[stypes.total.value][:, size_fft // 2, size_fft // 2, 0] -= term_0_2
s4[stypes.self.value][:, size_fft // 2, size_fft // 2, 0] -= term_0_2 / particles

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
s4 /= norm.reshape((n_lags, 1, 1, 1))

for i in range(0, n_lags):
  for j in range(0, size_fft):
    for k in range(0, size_fft):
      for l in range(0, (size_fft // 2) + 1):
        time_ta = lags[i] * timestep * tbsave
        # Print t_a, x, y, and z components of fft frequency, total,
        # self, and distinct averages, number of frame sets
        # contributing to such average, and frame difference
        # corresponding to t_a
        print("%f %f %f %f %f %f %f %d %d" %(time_ta, (j-size_fft//2)*2*math.pi/box_size, (k-size_fft//2)*2*math.pi/box_size, l*2*math.pi/box_size, s4[stypes.total.value][i][j][k][l], s4[stypes.self.value][i][j][k][l], s4[stypes.distinct.value][i][j][k][l], norm[i], lags[i]))
