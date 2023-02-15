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
        "-n Number of files in each run",
        "-r Number of runs, numbered as folders",
        "-s Frame number to start on (index starts at 1)",
        "-k Last frame number in range to use for initial times (index starts at 1)",
        "-m Last frame number in range to use for analysis, either final or initial times (index starts at 1)",
        "-d Number of frames between starts of pairs to average (dt)",
        "-x Dimensionality of FFT matrix, length in each dimension in addition to 0",
        "-y Box size in each dimension (assumed to be cubic, required)"
        "-a Offset between centers of begginning and end intervals in frames (t_a)",
        "-c Difference between intervals in frames (t_c)",
        "-o Start index (from 1) of particles to limit analysis to",
        "-p End index (from 1) of particles to limit analysis to",
        "-q Upper boundary for first q region with discrete q values",
        "-v Upper boundary for second q region divided into onion shells",
        "-l Number of onion shells to use in second q region",
        "-i Write output to files, one for each t_b",
        "-h Print usage",
        "Interval increase progression (last specified is used):",
        "-f Flenner-style periodic-exponential-increasing increment (iterations: 50, power: 5)",
        "-g Geometric spacing progression, selectively dropped to fit on integer frames (argument is geometric base)",
        "w function types (last specified is used, must be specified):",
        "-t Theta function threshold (argument is threshold radius)",
        "-u Double negative exponential/Gaussian (argument is exponential length)",
        "-e Single negative exponential (argument is exponential length)",
        sep="\n", file=sys.stderr)

try:
  opts, args = getopt.gnu_getopt(sys.argv[1:], "n:r:s:k:m:d:x:y:a:c:o:p:q:v:l:ihfg:t:u:e:")
except getopt.GetoptError as err:
  print(err, file=sys.stderr)
  usage()
  sys.exit(1)

class progtypes(enum.Enum):
  flenner = 1
  geometric = 2

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
# Total number of run folders.
n_runs = 0
# What frame number to start on
start = 0
# Last frame number to use for initial times
initend = None
# Last frame number to use for either final or initial times
final = None
# Difference between frame set starts
framediff = 10
# Limit of number of Fourier transform vector constants (including q=0)
size_fft = None
# User-defined value of dimension of box, assumed to be cubic
box_size = None
# Offset between centers of beginning and end intervals (t_a)
ta = 0
# Half difference between length of first and second intervals (t_c)
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
# Type of progression to increase time interval by
progtype = progtypes.flenner
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
  elif o == "-k":
    initend = int(a)
  elif o == "-m":
    final = int(a)
  elif o == "-d":
    framediff = int(a)
  elif o == "-x":
    size_fft = int(a) + 1
  elif o == "-y":
    box_size = float(a)
  elif o == "-a":
    ta = int(a)
  elif o == "-c":
    tc = int(a)
  elif o == "-f":
    progtype = progtypes.flenner
  elif o == "-g":
    progtype = progtypes.geometric
    geom_base = float(a)
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

# Open trajectory files
dcdfiles, fileframes, fparticles, timestep, tbsave = lib.opentraj.opentraj_multirun(n_runs, "run", n_files, "traj", 1, True)

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
print("#tbsave: %f" %tbsave)

# Spatial size of individual cell for FFT
cell = box_size / size_fft

# End of set of frames to use for initial times
if initend == None:
  initend = fileframes[-1]
else:
  if initend > fileframes[-1]:
    raise RuntimeError("End initial time frame beyond set of frames")

# End of set of frames to used for both final and initial times
if final == None:
  final = fileframes[-1]
else:
  if final > fileframes[-1]:
    raise RuntimeError("End limit time frame beyond set of frames")

# Number of frames to analyze
n_frames = final - start

# Largest possible average interval width (t_b), adjusting for both
# space taken up by t_a and t_c and intervals at the beginning which
# may not be accessible
max_width = n_frames - 1 - (framediff * ((max(tc - ta, 0) + (framediff - 1)) // framediff)) - max(ta + tc, 0)

if progtype == progtypes.flenner:
  # Construct list of frame difference numbers for sampling according
  # to a method of increasing spacing
  magnitude = -1
  frames_beyond_magnitude = max_width
  while frames_beyond_magnitude >= 50 * 5**(magnitude + 1):
    magnitude += 1
    frames_beyond_magnitude -= 50 * 5**magnitude

  samples_beyond_magnitude = frames_beyond_magnitude // 5**(magnitude + 1)

  n_samples = 1 + (50 * (magnitude + 1)) + samples_beyond_magnitude

  # Allocate that array
  samples = np.empty(n_samples, dtype=np.int64)

  # Efficiently fill the array
  samples[0] = 0
  last_sample_number = 0
  for i in range(0, magnitude + 1):
    samples[1 + 50 * i : 1 + 50 * (i + 1)] = last_sample_number + np.arange(5**i , 51 * 5**i, 5**i)
    last_sample_number += 50 * 5**i
  samples[1 + 50 * (magnitude + 1) : n_samples] = last_sample_number + np.arange(5**(magnitude + 1), (samples_beyond_magnitude + 1) * 5**(magnitude + 1), 5**(magnitude + 1))

elif progtype == progtypes.geometric:
  # Largest power of geom_base that will be able to be sampled
  end_power = math.floor(math.log(max_width, geom_base))

  # Create array of sample numbers following geometric progression,
  # with flooring to have samples adhere to integer boundaries,
  # removing duplicate numbers, and prepending 0
  samples = np.insert(np.unique(np.floor(np.logspace(0, end_power, num=end_power + 1, base=geom_base)).astype(np.int64)), 0, 0)

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

# Coordinates wrapped to box size
x0m = np.empty(particles, dtype=np.single)
y0m = np.empty(particles, dtype=np.single)
z0m = np.empty(particles, dtype=np.single)

# Used when interval start times are different
if ta - tc != 0:
  x2 = np.empty(particles, dtype=np.single)
  y2 = np.empty(particles, dtype=np.single)
  z2 = np.empty(particles, dtype=np.single)

  # Coordinates wrapped to box size
  x2m = np.empty(particles, dtype=np.single)
  y2m = np.empty(particles, dtype=np.single)
  z2m = np.empty(particles, dtype=np.single)

# Difference between particle positions at start of intervals, used
# when interval start times are different
if ta - tc != 0:
  xdiff = np.empty(particles, dtype=np.single)
  ydiff = np.empty(particles, dtype=np.single)
  zdiff = np.empty(particles, dtype=np.single)

# Center of mass of each frame
cm = [np.empty((n_frames, 3), dtype=np.float64)] * n_runs

# Bins for holding particle positions

# Bins for first interval
a_bins = np.empty((size_fft, size_fft, size_fft), dtype=np.float64)

# Only one set of bins is needed if intervals are equivalent
if ta != 0 or tc != 0:
  b_bins = np.empty((size_fft, size_fft, size_fft), dtype=np.float64)

# If interval start times are the same, self bins are not needed, as
# the exponential part is always 1
if ta - tc != 0:
  self_bins = np.empty((size_fft, size_fft, size_fft), dtype=np.float64)

# Temporary value for each run to allow for calculation of each run's
# self component of s4
if tc - ta == 0:
  # If start frames of intervals are the same, then the self s4 values
  # do not vary with q
  run_self_s4 = np.empty((1, 1, 1), dtype=np.float64)
else:
  run_self_s4 = np.empty((size_fft, size_fft, (size_fft // 2) + 1), dtype=np.float64)

# Temporary value for each run to allow for calculation of each run's
# total component of s4
run_total_s4 = np.empty((size_fft, size_fft, (size_fft // 2) + 1), dtype=np.float64)

# Structure factor variance for each interval width. The first and
# second fft dimensions include values for negative vectors. Since all
# inputs are real, this is not required for the third fft dimension.
s4 = np.empty((n_stypes, size_fft, size_fft, (size_fft // 2) + 1), dtype=np.float64)

# W function values for each particle
if ta == 0 and tc == 0:
  # If intervals are the same, only one w needs be found
  if wtype == wtypes.theta:
    w = np.empty((1, particles), dtype=np.int8)
  else:
    w = np.empty((1, particles), dtype=np.float64)
else:
  # If intervals are different, different sets of w values must be
  # computed for first and second intervals
  if wtype == wtypes.theta:
    w = np.empty((2, particles), dtype=np.int8)
  else:
    w = np.empty((2, particles), dtype=np.float64)

# Find center of mass of each frame
print("Finding centers of mass for frames", file=sys.stderr)
for i in range(0, n_runs):
  for j in range(0, n_frames):
    which_file = np.searchsorted(fileframes, start + j, side="right") - 1
    offset = start + j - fileframes[which_file]

    if limit_particles == True:
      dcdfiles[i][which_file].gdcdp(x, y, z, offset)
      cm[i][j][0] = np.mean(x[lower_limit:upper_limit])
      cm[i][j][1] = np.mean(y[lower_limit:upper_limit])
      cm[i][j][2] = np.mean(z[lower_limit:upper_limit])
    else:
      dcdfiles[i][which_file].gdcdp(x0, y0, z0, offset)
      cm[i][j][0] = np.mean(x0)
      cm[i][j][1] = np.mean(y0)
      cm[i][j][2] = np.mean(z0)

print("#dt = %d" %framediff)
print("#n_samples = %d" %n_samples)
print("#t_a = %d" %ta)
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

# List of shell numbers to use for shell intervals
qlist_shells = list(range(0, shells))

# List of q values to use for region of discrete q values
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

# Read values for a given frame and run from DCD file, correcting for
# frame center of mass
def get_frame(t0, xb0, yb0, zb0, run):
  which_file = np.searchsorted(fileframes, start + t0, side="right") - 1
  offset = start + t0 - fileframes[which_file]
  if limit_particles == True:
    dcdfiles[run][which_file].gdcdp(x, y, z, offset)
    xb0[:] = x[lower_limit:upper_limit]
    yb0[:] = y[lower_limit:upper_limit]
    zb0[:] = z[lower_limit:upper_limit]
  else:
    dcdfiles[run][which_file].gdcdp(xb0, yb0, zb0, offset)

  # Correct for center of mass
  xb0 -= cm[run][t0][0]
  yb0 -= cm[run][t0][1]
  zb0 -= cm[run][t0][2]

# Get end frame values and calculate w, do not modify start frame
# values
def calculate_w(wa, xa0, ya0, za0, t1, xa1, ya1, za1, run):
  get_frame(t1, xa1, ya1, za1, run)

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

# S4 calculation

print("Entering S4 calculation", file=sys.stderr)

# Iterate over average interval lengths (t_b)
for index, tb in enumerate(samples):
  # Clear interval accumulator
  s4[:, :, :, :] = 0.0

  # Normalization factor for number of sets contributing to value for
  # t_b
  norm = 0.0

  # Iterate over runs (FFT will be averaged over runs)
  for i in np.arange(0, n_runs):
    # Clear run accumulators
    run_self_s4[:, :, :] = 0.0
    run_total_s4[:, :, :] = 0.0

    # Iterate over starting points for structure factor
    for j in np.arange(0, initend - start, framediff):
      # Use only indices that are within range
      if (ta < (tc - j) or
          ta - n_frames >= (tc - j) or
          ta < (-tb - j) or
          ta - n_frames >= (-tb - j) or
          j < (tc - tb) or
          j - n_frames >= (tc - tb)):
        continue

      # Get starting frame for first interval and store in x0, y0, z0
      get_frame(j, x0, y0, z0, i)

      # Wrap coordinates to box size for binning
      x0m[:] = x0 % box_size
      y0m[:] = y0 % box_size
      z0m[:] = z0 % box_size

      # If needed, get starting frame for second interval and store in
      # x2, y2, z2
      if ta - tc != 0:
        get_frame(j + ta - tc, x2, y2, z2, i)

        # Wrap coordinates to box size for binning
        x2m[:] = x2 % box_size
        y2m[:] = y2 % box_size
        z2m[:] = z2 % box_size

      # If needed, find differences between particle positions in
      # starting frames
      if ta - tc != 0:
        xdiff[:] = (((x0 // cell) - (x2 // cell) + 0.5) * cell) % box_size
        ydiff[:] = (((y0 // cell) - (y2 // cell) + 0.5) * cell) % box_size
        zdiff[:] = (((z0 // cell) - (z2 // cell) + 0.5) * cell) % box_size

      # Calculate w values for first interval
      calculate_w(w[0], x0, y0, z0, j + tb - tc, x1, y1, z1, i)

      # Calculate w values for second interval if needed
      if ta != 0 or tc != 0:
        if ta - tc != 0:
          calculate_w(w[1], x2, y2, z2, j + ta + tb, x1, y1, z1, i)
        else:
          calculate_w(w[1], x0, y0, z0, j + ta + tb, x1, y1, z1, i)

      # Sort first interval w values into bins for FFT
      a_bins, dummy = np.histogramdd((x0m, y0m, z0m), bins=size_fft, range=((0, box_size), ) * 3, weights=w[0])

      # Sort second interval w values into bins for FFT if needed
      if ta != 0 or tc != 0:
        if ta - tc == 0:
          b_bins, dummy = np.histogramdd((x0m, y0m, z0m), bins=size_fft, range=((0, box_size), ) * 3, weights=w[1])
        else:
          b_bins, dummy = np.histogramdd((x2m, y2m, z2m), bins=size_fft, range=((0, box_size), ) * 3, weights=w[1])

      # Calculate total part of S4
      if ta != 0 or tc != 0:
        run_total_s4 += fft.fftshift((fft.rfftn(a_bins) * np.conjugate(fft.rfftn(b_bins))).real, axes=(0, 1)) / particles
      else:
        # Uses np.abs(), which calculates norm of complex numbers
        run_total_s4 += fft.fftshift(np.abs(fft.rfftn(a_bins))**2, axes=(0, 1)) / particles

      # Multiply w values for different intervals together for self
      # bins
      if ta != 0 or tc != 0:
        w[0] *= w[1]
      else:
        # Squaring not required if w is same for each interval and is
        # boolean values, as 1*1 = 1 and 0*0 = 0
        if wtype != wtypes.theta:
          w[0] *= w[0]

      # Calculate self part
      if ta - tc != 0:
        # Bin multiplied w values according to coordinate differences
        self_bins, dummy = np.histogramdd((xdiff, ydiff, zdiff), bins=size_fft, range=((0, box_size), ) * 3, weights=w[0])

        # Perform FFT, thereby calculating self S4 for current index
        run_self_s4 += fft.fftshift(fft.rfftn(self_bins).real, axes=(0, 1)) / particles
      else:
        run_self_s4[0][0][0] += np.sum(w[0]) / particles

      # Accumulate the normalization value for this sample value, which
      # we will use later in computing the mean value for each t_b
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
  s4 /= n_runs

  # Calculate standard deviations from normalized variances over runs
  s4[stypes.totalstd.value] = np.sqrt((s4[stypes.totalstd.value] - s4[stypes.total.value]**2) / (n_runs - 1))
  s4[stypes.selfstd.value] = np.sqrt((s4[stypes.selfstd.value] - s4[stypes.self.value]**2) / (n_runs - 1))
  s4[stypes.distinctstd.value] = np.sqrt((s4[stypes.distinctstd.value] - s4[stypes.distinct.value]**2) / (n_runs - 1))

  # Print results for current t_b

  time_tb = tb * timestep * tbsave

  # File to write data for time to
  if dumpfiles == True:
    file = open("tb_%f" %(tb), "w")
  else:
    file = sys.stdout

  # Clear accumulators
  qaccum_discrete[:, :] = 0.0
  qaccum_shells[:, :] = 0.0

  for j in range(qb2l, qb2u + 1):
    for k in range(qb2l, qb2u + 1):
      for l in range(0, qb2u + 1):
        # Index of qlist we are to use
        qcurrent = element_qs[j - qb2l][k - qb2l][l]

        # If matrix element corresponds to used q value in either
        # qlist_discrete_sorted or qlist_shells
        if element_qs[j - qb2l][k - qb2l][l][0] == 0:
          # Accumulate values to corresponding q value
          qaccum_discrete[:, element_qs[j - qb2l][k - qb2l][l][1]] += s4[:, (size_fft//2)+j, (size_fft//2)+k, l]
        if element_qs[j - qb2l][k - qb2l][l][0] == 1:
          # Accumulate values to corresponding q value
          qaccum_shells[:, element_qs[j - qb2l][k - qb2l][l][1]] += s4[:, (size_fft//2)+j, (size_fft//2)+k, l]

  # Normalize q values for number of contributing elements
  qaccum_discrete /= qnorm_discrete_sorted
  qaccum_shells /= qnorm_shells

  # For each discrete q value, print t_b, q value, number of FFT matrix
  # elements contributing to q value, total, self, and distinct
  # averages, standard deviations of total, self, and distinct
  # averages, number of frame sets contributing to such averages, and
  # frame difference corresponding to t_b
  for j in range(0, len(qlist_discrete_sorted)):
    file.write("%f %f %d %f %f %f %f %f %f %d %d\n" %(time_tb,
                                                      qlist_discrete_sorted[j]*2*math.pi/box_size,
                                                      qnorm_discrete_sorted[j],
                                                      qaccum_discrete[stypes.total.value][j],
                                                      qaccum_discrete[stypes.self.value][j],
                                                      qaccum_discrete[stypes.distinct.value][j],
                                                      qaccum_discrete[stypes.totalstd.value][j],
                                                      qaccum_discrete[stypes.selfstd.value][j],
                                                      qaccum_discrete[stypes.distinctstd.value][j],
                                                      norm,
                                                      tb))

  # For each shell, print t_b, midpoint of q value range of fft
  # frequency, number of FFT matrix elements contributing to q value,
  # total, self, and distinct averages, standard deviations of total,
  # self, and distinct averages, number of frame sets contributing to
  # such averages, and frame difference corresponding to t_b
  for j in range(0, len(qlist_shells)):
    file.write("%f %f %d %f %f %f %f %f %f %d %d\n" %(time_tb,
                                                      (qb1+(qlist_shells[j]+0.5)*swidth)*2*math.pi/box_size,
                                                      qnorm_shells[j],
                                                      qaccum_shells[stypes.total.value][j],
                                                      qaccum_shells[stypes.self.value][j],
                                                      qaccum_shells[stypes.distinct.value][j],
                                                      qaccum_shells[stypes.totalstd.value][j],
                                                      qaccum_shells[stypes.selfstd.value][j],
                                                      qaccum_shells[stypes.distinctstd.value][j],
                                                      norm,
                                                      tb))

  # Close file if opened
  if dumpfiles == True:
    file.close()
