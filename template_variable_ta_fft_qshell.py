import numpy as np
from numpy import fft
import pydcd
import sys
import math
import getopt
import enum

def usage():
  print("Arguments:",
        "-n Number of files in each run",
        "-r Number of runs, numbered as folders",
        "-s Frame number to start on (index starts at 1)",
        "-d Spacing between initial times as well as lag values (dt)",
        "-x Dimensionality of FFT matrix, length in each dimension in addition to 0",
        "-y Box size in each dimension (assumed to be cubic, required)",
        "-b Average interval in frames (t_b)",
        "-c Difference between intervals in frames (t_c)",
        "-o Start index (from 1) of particles to limit analysis to",
        "-p End index (from 1) of particles to limit analysis to",
        "-q Upper boundary for first q region with discrete q values",
        "-v Upper boundary for second q region divided into onion shells",
        "-l Number of onion shells to use in second q region",
        "-i Write output to files, one for each lag time",
        "-h Print usage",
        "w function types (last specified is used, must be specified):",
        "-t Theta function threshold (argument is threshold radius)",
        "-u Double negative exponential/Gaussian (argument is exponential length)",
        "-e Single negative exponential (argument is exponential length)",
        sep="\n", file=sys.stderr)

try:
  opts, args = getopt.gnu_getopt(sys.argv[1:], "n:r:s:d:x:y:b:c:o:p:q:v:l:ijht:u:e:")
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
  totalstd = 3
  selfstd = 4
  distinctstd = 5
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
# Whether to limit analysis to subset of particles, and upper and lower
# indices for limit.
limit_particles = False
upper_limit = None
lower_limit = None
# Upper boundary for first q region
qb1 = None
# Upper boundary for second q region
qb2 = None
# Number of onion shells for second region
shells = None
# Whether to write output to files rather than stdout
dumpfiles = False
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
  elif o == "-o":
    limit_particles = True
    lower_limit = int(a) - 1
  elif o == "-p":
    limit_particles = True
    upper_limit = int(a)
  elif o == "-q":
    qb1 = float(a)
  elif o == "-v":
    qb2 = float(a)
  elif o == "-l":
    shells = int(a)
  elif o == "-i":
    dumpfiles = True
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

# If q is not only 0-vector
if size_fft > 1:
  if qb1 == None:
    raise RuntimeError("Must specify upper q boundary for first region if nonzero q values used")

  if qb2 == None:
    raise RuntimeError("Must specify upper q boundary for second region if nonzero q values used")

  if shells == None:
    raise RuntimeError("Must specify number of onion shells in second region if nonzero q values used")

  # Convert upper boundaries of regions to multipliers of smallest q
  # magnitude
  qb1 = qb1 * box_size / (2 * math.pi)
  qb2 = qb2 * box_size / (2 * math.pi)

else:
  # Only q=0 being calculated, q regions of 0 width
  qb1 = 0.0
  qb2 = 0.0

# Holds number of frames per file
fileframes = np.empty(n_files + 1, dtype=np.int64)
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
    file = open("run%d/traj%d.dcd" %(i + 1, j + 1), "r")
    dcdfiles[i].append(pydcd.dcdfile(file))
    file.close()

    # Make sure each trajectory file in each run mirrors the files in
    # other runs and has the same time step and number of particles
    if i == 0:
      fileframes[j + 1] = dcdfiles[i][j].nset
      total_frames += dcdfiles[i][j].nset

      if j == 0:
        # Number of particles in files, may not be same as limited
        # number in analysis
        fparticles = dcdfiles[i][j].N

        timestep = dcdfiles[i][j].timestep
        tbsave = dcdfiles[i][j].tbsave
      else:
        if dcdfiles[i][j].N != fparticles:
          raise RuntimeError("Not the same number of particles in each file")
        if dcdfiles[i][j].timestep != timestep:
          raise RuntimeError("Not the same time step in each file")
        if dcdfiles[i][j].tbsave != tbsave:
          raise RuntimeError("Not the same frame difference between saves in each file")

    else:
      if dcdfiles[i][j].nset != fileframes[j + 1]:
        raise RuntimeError("Not the same number of frames in each run for corresponding files")

      if dcdfiles[i][j].N != fparticles:
        raise RuntimeError("Not the same number of particles in each file")
      if dcdfiles[i][j].timestep != timestep:
        raise RuntimeError("Not the same time step in each file")
      if dcdfiles[i][j].tbsave != tbsave:
        raise RuntimeError("Not the same frame difference between saves in each file")

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
print("#nset: %d" %total_frames)
print("#N: %d" %particles)
print("#timestep: %f" %timestep)
print("#tbsave: %f" %tbsave)

# Spatial size of individual cell for FFT
cell = box_size / size_fft

# Number of frames in each run to analyze
n_frames = total_frames - start

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
# second 0 vector term of S4 (term_0_2).
a_accum = 0.0
b_accum = 0.0

# Normalization factors for A and B accumulators for second 0 vector
# term of S4.
a_norm = 0
b_norm = 0

# Temporary value for each run to allow for calculation of each run's
# self component of s4
run_self_s4 = np.empty((n_lags, size_fft, size_fft, (size_fft // 2) + 1), dtype=np.float64)

# Structure factor variance for each difference in times. The first
# and second fft dimensions hold values for negative vectors. Since all
# inputs are real, this is not required for the third fft dimension.
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

      if limit_particles == True:
        dcdfiles[j][which_file].gdcdp(x, y, z, offset)
        cm[j][i][ttype][0] = np.mean(x[lower_limit:upper_limit])
        cm[j][i][ttype][1] = np.mean(y[lower_limit:upper_limit])
        cm[j][i][ttype][2] = np.mean(z[lower_limit:upper_limit])
      else:
        dcdfiles[j][which_file].gdcdp(x0, y0, z0, offset)
        cm[j][i][ttype][0] = np.mean(x0)
        cm[j][i][ttype][1] = np.mean(y0)
        cm[j][i][ttype][2] = np.mean(z0)

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
  if limit_particles == True:
    dcdfiles[run][which_file].gdcdp(x, y, z, offset)
    xa0[:] = x[lower_limit:upper_limit]
    ya0[:] = y[lower_limit:upper_limit]
    za0[:] = z[lower_limit:upper_limit]
  else:
    dcdfiles[run][which_file].gdcdp(xa0, ya0, za0, offset)

  # Get values for end of w function
  which_file = np.searchsorted(fileframes, start + index2, side="right") - 1
  offset = start + index2 - fileframes[which_file]
  if limit_particles == True:
    dcdfiles[run][which_file].gdcdp(x, y, z, offset)
    xa1[:] = x[lower_limit:upper_limit]
    ya1[:] = y[lower_limit:upper_limit]
    za1[:] = z[lower_limit:upper_limit]
  else:
    dcdfiles[run][which_file].gdcdp(xa1, ya1, za1, offset)

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
            (za1 - za0)**2, radius**2, out=wa).astype(np.int8, copy=False)
  elif wtype == wtypes.gauss:
    np.exp(-((xa1 - xa0)**2 +
             (ya1 - ya0)**2 +
             (za1 - za0)**2)/(2 * gscale**2), out=wa)
  elif wtype == wtypes.exp:
    np.exp(-np.sqrt((xa1 - xa0)**2 +
                    (ya1 - ya0)**2 +
                    (za1 - za0)**2)/sscale, out=wa)

# S4 calcuation

print("Entering S4 calculation", file=sys.stderr)

# Iterate over runs (FFT will be averaged over runs)
for i in range(0, n_runs):
  # Clear self accumulator
  run_self_s4[:,:,:,:] = 0.0
  for j in range(0, n_init):
    root = framediff * j

    # Total S4 calculation, uses FFT-based autocorrelation

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
      a_accum += np.sum(w[0])

      # Normalization factor for second term of variance
      if i == 0:
        a_norm += 1

    if root - tc >= 0 and root + tb < n_frames:
      # Calculate w values for t3 and t4
      calculate_w(w[1], i, x1, y1, z1, x2, y2, z2, j, ttypes.t3, ttypes.t4)

      # Wrap particle coordinates to box. If not done, particles will
      # not be distributed across bins correctly.
      x1 %= box_size
      y1 %= box_size
      z1 %= box_size

      # Bin values for FFT
      b_bins[j], dummy = np.histogramdd((x1, y1, z1), bins=size_fft, range=((0, box_size), ) * 3, weights=w[1])

      # Accumulate for second term of variance
      b_accum += np.sum(w[1])

      # Normalization factor for second term of variance
      if i == 0:
        b_norm += 1

    # Self S4 calculation, iterates over each possible interval end
    # frame for current start frame

    for index, ta in enumerate(lags):
      if (ta < max(tc - root, -tb - root) or
          ta - n_frames >= min(tc - root, -tb - root) or
          root < (tc - tb) or
          root - n_frames >= (tc - tb)):
        continue

      # Calculate w values for t3 and t4 (again for each interval end)
      calculate_w(w[1], i, x1, y1, z1, x2, y2, z2, j + ta // framediff, ttypes.t3, ttypes.t4)

      # Align particle coordinates to make sum equivalent to total
      # part and find difference in positions for binning. Since total
      # exponential is negative, must use reverse difference with
      # positive-exponential FFT.
      x1[:] = ((x0 // cell) - (x1 // cell) + 0.5) * cell
      y1[:] = ((y0 // cell) - (y1 // cell) + 0.5) * cell
      z1[:] = ((z0 // cell) - (z1 // cell) + 0.5) * cell

      # Wrap particle coordinates to box. If not done, particles will
      # not be distributed across bins correctly.
      x1 %= box_size
      y1 %= box_size
      z1 %= box_size

      # Multiply w values for calculation of pairs
      w[1] *= w[0]

      # Bin multiplied w values according to coordinate differences
      self_bins, dummy = np.histogramdd((x1, y1, z1), bins=size_fft, range=((0, box_size), ) * 3, weights=w[1])

      # Calculate the variance for the current index
      run_self_s4[index] += fft.fftshift(fft.rfftn(self_bins).real, axes=(0, 1)) / particles

      # Accumulate the normalization value for this lag value, which
      # we will use later in computing the mean value for each t_b
      if i == 0:
        norm[index] += 1

    print("Processed frame %d in run %d" %(start + root + 1, i + 1), file=sys.stderr)

  # Calculate correlations of first point sums with second point sums.
  # This will later be normalized for number of terms corresponding to
  # each correlation offset. Last spatial axis is halved due to use of
  # rfftn. b_bins must be conjugated for the FFT over the last 3
  # spatial dimensions, but unconjugated for the time dimension. The
  # FFT of a_bins must be conjugated for the time dimension only. The
  # roll followed by flip unconjugates the time axis of b_bins and
  # conjugates the time axis of a_bins.
  run_total_s4 = fft.fftshift(fft.ifft(np.flip(np.roll(fft.rfftn(a_bins) * np.conjugate(fft.rfftn(b_bins)), -1, axis=0), axis=0), axis=0).real, axes=(0, 1, 2))[n_init-1+(max_neg_lag//framediff):n_init+(max_pos_lag//framediff)] / particles

  # Normalize the accumulated values, thereby obtaining averages over
  # each pair of frames
  run_total_s4 /= norm.reshape((n_lags, 1, 1, 1))
  run_self_s4 /= norm.reshape((n_lags, 1, 1, 1))

  # Calculate distinct part of S4 for current run
  run_distinct_s4 = run_total_s4 - run_self_s4

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
s4 /= n_runs

# Calculate standard deviations from normalized variances over runs
s4[stypes.totalstd.value] = np.sqrt((s4[stypes.totalstd.value] - s4[stypes.total.value]**2) / (n_runs - 1))
s4[stypes.selfstd.value] = np.sqrt((s4[stypes.selfstd.value] - s4[stypes.self.value]**2) / (n_runs - 1))
s4[stypes.distinctstd.value] = np.sqrt((s4[stypes.distinctstd.value] - s4[stypes.distinct.value]**2) / (n_runs - 1))

# Normalize second term of variance across runs and initial times
a_accum /= n_runs * a_norm
b_accum /= n_runs * b_norm

# Used with 0 vector for calculating second term of variance. This will
# later be normalized for number of terms corresponding to each
# correlation offset. a_accum must be conjugated for the correlation.
term_0_2 = a_accum * b_accum / particles

# Subtract second term of variance from 0 vector terms
s4[stypes.total.value][:, size_fft // 2, size_fft // 2, 0] -= term_0_2
s4[stypes.self.value][:, size_fft // 2, size_fft // 2, 0] -= term_0_2 / particles
s4[stypes.distinct.value][:, size_fft // 2, size_fft // 2, 0] -= term_0_2 * (particles - 1) / particles

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

# Upper and lower bounds for dimensions of q that fit within qb2, used
# for matrix dimensioning
qb2l = max(-int(qb2), -(size_fft // 2))
qb2u = min(int(qb2), (size_fft - 1) // 2)

# Shell width
if shells != 0:
  swidth = (qb2 - qb1) / shells

# List of q values to use, first region is for shell intervals, second,
# to be appended, is for specific q values. First region does not
# contain actual q values, as intervals are calculated later. Instead,
# the first region (index < shells) contains shell numbers.
qlist_shells = list(range(0, shells))
qlist_discrete = list()

# Norm for number of FFT matrix elements corresponding to each element
# of qlist, for first and second regions.
qnorm_shells = [0] * shells
qnorm_discrete = list()

# Array of indices in qlist matrix elements correspond to. The first
# number in the last dimension is whether the index is not in the q
# range (-1), is within the first region of discrete q values (0), or
# is within the second region of shells (1). The second number in the
# last dimension is the qlist index.
element_qs = np.empty((qb2u - qb2l + 1, qb2u - qb2l + 1, qb2u + 1, 2), dtype=np.int64)

# Initialize to default of no corresponding index
element_qs[:, :, :, 0] = -1

# Find q lengths corresponding to each set of q coordinates
for i in range(qb2l, qb2u + 1):
  for j in range(qb2l, qb2u + 1):
    for k in range(0, qb2u + 1):
      hyp = float(np.linalg.norm((i, j, k)))
      if hyp > qb1:
        # Index of onion shell that would include given q
        shell_index = shells - int((qb2 - hyp) // swidth) - 1
        if shell_index < shells:
          element_qs[i - qb2l][j - qb2l][k][0] = 1
          element_qs[i - qb2l][j - qb2l][k][1] = shell_index
          qnorm_shells[shell_index] += 1
      else:
        if not (hyp in qlist_discrete):
          qlist_discrete.append(hyp)
          qnorm_discrete.append(0)
        element_qs[i - qb2l][j - qb2l][k][0] = 0
        element_qs[i - qb2l][j - qb2l][k][1] = qlist_discrete.index(hyp)
        qnorm_discrete[qlist_discrete.index(hyp)] += 1

# Sorted copies of discrete qlist and qnorm
qlist_discrete_sorted, qnorm_discrete_sorted = zip(*sorted(zip(qlist_discrete, qnorm_discrete)))

# Delete q elements with 0 norm (possible for shells)
for i in reversed(range(0, len(qlist_shells))):
  if qnorm_shells[i] == 0:
    qlist_shells.pop(i)
    qnorm_shells.pop(i)
    # Shift element_qs values to take into account new ordering of
    # qlistsorted
    for j in range(qb2l, qb2u + 1):
      for k in range(qb2l, qb2u + 1):
        for l in range(0, qb2u + 1):
          # If within shell region and above deleted shell value
          if element_qs[j][k][l][0] == 1 and element_qs[j][k][l][1] >= i:
            element_qs[j][k][l][1] -= 1

# Modify element_qs values to point to indices in qlistsorted rather
# than qlist
for i in range(qb2l, qb2u + 1):
  for j in range(qb2l, qb2u + 1):
    for k in range(0, qb2u + 1):
      # Only sort discrete values, shell values already sorted
      if element_qs[i][j][k][0] == 0:
        element_qs[i][j][k][1] = qlist_discrete_sorted.index(qlist_discrete[element_qs[i][j][k][1]])

# Accumulated values of S4 for each q value. First dimension
# corresponds to S4 type
qaccum_discrete = np.empty((n_stypes, len(qlist_discrete_sorted)), dtype=np.float64)
qaccum_shells = np.empty((n_stypes, len(qlist_shells)), dtype=np.float64)

for i in range(0, n_lags):
  time_ta = lags[i] * timestep * tbsave

  # File to write data for time to
  if dumpfiles == True:
    file = open("lag_%f" %(lags[i]), "w")
  else:
    file = sys.stdout

  # Clear accumulators
  qaccum_discrete[:][:] = 0.0
  qaccum_shells[:][:] = 0.0

  for j in range(qb2l, qb2u + 1):
    for k in range(qb2l, qb2u + 1):
      for l in range(0, qb2u + 1):
        # Index of qlist we are to use
        qcurrent = element_qs[j - qb2l][k - qb2l][l]

        # If matrix element corresponds to used q value in either
        # qlist_discrete_sorted or qlist_shells
        if element_qs[j - qb2l][k - qb2l][l][0] == 0:
          # Accumulate values to corresponding q value
          qaccum_discrete[:, element_qs[j - qb2l][k - qb2l][l][1]] += s4[:, i, (size_fft//2)+j, (size_fft//2)+k, l]
        if element_qs[j - qb2l][k - qb2l][l][0] == 1:
          # Accumulate values to corresponding q value
          qaccum_shells[:, element_qs[j - qb2l][k - qb2l][l][1]] += s4[:, i, (size_fft//2)+j, (size_fft//2)+k, l]

  # Normalize q values for number of contributing elements
  qaccum_discrete /= qnorm_discrete_sorted
  qaccum_shells /= qnorm_shells

  # For each discrete q value, print t_a, q value, number of FFT matrix
  # elements contributing to q value, total, self, and distinct
  # averages, standard deviations of total, self, and distinct
  # averages, number of frame sets contributing to such averages, and
  # frame difference corresponding to t_a
  for j in range(0, len(qlist_discrete_sorted)):
    file.write("%f %f %d %f %f %f %f %f %f %d %d\n" %(time_ta,
                                                      qlist_discrete_sorted[j])*2*math.pi/box_size,
                                                      qnorm_discrete_sorted[j],
                                                      qaccum_discrete[stypes.total.value][j],
                                                      qaccum_discrete[stypes.self.value][j],
                                                      qaccum_discrete[stypes.distinct.value][j],
                                                      qaccum_discrete[stypes.totalstd.value][j],
                                                      qaccum_discrete[stypes.selfstd.value][j],
                                                      qaccum_discrete[stypes.distinctstd.value][j],
                                                      norm[i],
                                                      lags[i]))

  # For each shell, print t_a, midpoint of q value range of fft
  # frequency, number of FFT matrix elements contributing to q value,
  # total, self, and distinct averages, standard deviations of total,
  # self, and distinct averages, number of frame sets contributing to
  # such averages, and frame difference corresponding to t_a
  for j in range(0, len(qlist_shells)):
    file.write("%f %f %d %f %f %f %f %f %f %d %d\n" %(time_ta,
                                                      (qb1+(qlist_shells[j]+0.5)*swidth)*2*math.pi/box_size,
                                                      qnorm_shells[j],
                                                      qaccum_shells[stypes.total.value][j],
                                                      qaccum_shells[stypes.self.value][j],
                                                      qaccum_shells[stypes.distinct.value][j],
                                                      qaccum_shells[stypes.totalstd.value][j],
                                                      qaccum_shells[stypes.selfstd.value][j],
                                                      qaccum_shells[stypes.distinctstd.value][j],
                                                      norm[i],
                                                      lags[i]))

  # Close file if opened
  if dumpfiles == True:
    file.close()
