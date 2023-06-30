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
import lib.qshell

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
# Offset between centers of beginning and end intervals (t_a)
ta = 0
# Half difference between length of initial and end intervals (t_c)
tc = 0
# Whether to write output to files rather than stdout
dumpfiles = False
# Progression specification/generation object for t_b values
prog = lib.progression.prog()
# Run set opening object
runset = lib.opentraj.runset()
# Trajectory set opening object
trajset = lib.opentraj.trajset(runset)
# Frame reading object
frames = lib.frame.frames(trajset)
# w function calculation object
wcalc = lib.wcalc.wcalc(frames)
# q vector shell sorting object
qshell = lib.qshell.qshell()
# Whether q vector shells are to be used
qshell_active = False

def usage():
  print("Arguments:", file=sys.stderr)
  runset.usage()
  trajset.usage()
  frames.usage()
  print("-k Last frame number in range to use for initial times (index starts at 1)",
        "-d Spacing between initial times (dt)",
        "-x Dimensionality of FFT matrix, length in each dimension in addition to 0",
        "-y Box size in each dimension (assumed to be cubic, required)",
        "-a Offset between centers of begginning and end intervals in frames (t_a)",
        "-c Difference between intervals in frames (t_c)",
        "-i Write output to files, one for each t_b value",
        "-h Print usage",
        sep="\n", file=sys.stderr)
  prog.usage()
  wcalc.usage()
  qshell.usage()
  print("If no q vector shell options specified, all q vector values printed", file=sys.stderr)

try:
  opts, args = getopt.gnu_getopt(sys.argv[1:], "k:d:x:y:a:c:ijh" +
                                               runset.shortopts +
                                               trajset.shortopts +
                                               prog.shortopts +
                                               frames.shortopts +
                                               wcalc.shortopts +
                                               qshell.shortopts,
                                               prog.longopts)
except getopt.GetoptError as err:
  print(err, file=sys.stderr)
  usage()
  sys.exit(1)

for o, a in opts:
  if o == "-h":
    usage()
    sys.exit(0)
  elif o == "-k":
    initend = int(a)
  elif o == "-s":
    start = int(a) - 1
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
  elif o == "-i":
    dumpfiles = True
  elif o == "-j":
    print("-j is default, no need to specify", file=sys.stderr)
  elif runset.catch_opt(o, a) == True:
    pass
  elif trajset.catch_opt(o, a) == True:
    pass
  elif prog.catch_opt(o, a) == True:
    pass
  elif frames.catch_opt(o, a) == True:
    pass
  elif wcalc.catch_opt(o, a) == True:
    pass
  elif qshell.catch_opt(o, a) == True:
    qshell_active = True

if box_size == None:
  raise RuntimeError("Must define box size dimensions")

if size_fft == None:
  raise RuntimeError("Must specify size for FFT matrix")

# Open trajectory files
trajset.opentraj_multirun(1, True)

# Prepare frames object for calculation
frames.prepare()

# Verify correctness of parameters for w calculation from arguments
wcalc.prepare()

# Generate qshell elements if onion shells are used, used for sorting
# values into shells
if qshell_active == True:
  qshell.prepare(size_fft, box_size)

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

# Largest and smallest possible average interval widths (t_b),
# adjusting for both space taken up by t_a and t_c and intervals at the
# beginning which may not be accessible
prog.min_val = -framediff * ((frames.n_frames - 1) // framediff) + max(tc, 0)
prog.max_val = frames.n_frames - 1 - (framediff * ((max(tc - ta, 0) + (framediff - 1)) // framediff)) - max(ta + tc, 0)

# Construct progression of interval values using previously-specified
# parameters
tbvals = prog.construct()

# Stores coordinates of all particles in a frame
x0 = np.empty(frames.particles, dtype=np.single)
y0 = np.empty(frames.particles, dtype=np.single)
z0 = np.empty(frames.particles, dtype=np.single)
x1 = np.empty(frames.particles, dtype=np.single)
y1 = np.empty(frames.particles, dtype=np.single)
z1 = np.empty(frames.particles, dtype=np.single)

# Used when interval start times are different
if ta - tc != 0:
  x2 = np.empty(frames.particles, dtype=np.single)
  y2 = np.empty(frames.particles, dtype=np.single)
  z2 = np.empty(frames.particles, dtype=np.single)

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
# self component of S4
if tc - ta == 0:
  # If start frames of intervals are the same, then the self S4 values
  # do not vary with q
  run_self_s4 = np.empty((1, 1, 1), dtype=np.float64)
else:
  run_self_s4 = np.empty((size_fft, size_fft, (size_fft // 2) + 1), dtype=np.float64)

# Temporary value for each run to allow for calculation of each run's
# total component of S4
run_total_s4 = np.empty((size_fft, size_fft, (size_fft // 2) + 1), dtype=np.float64)

# Structure factor variance for each interval width. The first and
# second fft dimensions include values for negative vectors. Since all
# inputs are real, this is not required for the third fft dimension.
s4 = np.empty((n_stypes, size_fft, size_fft, (size_fft // 2) + 1), dtype=np.float64)

# W function values for each particle
if ta == 0 and tc == 0:
  # If intervals are the same, only one w needs be found
  if wcalc.wtype == lib.wcalc.wtypes.theta:
    w = np.empty((1, frames.particles), dtype=np.int8)
  else:
    w = np.empty((1, frames.particles), dtype=np.float64)
else:
  # If intervals are different, different sets of w values must be
  # computed for first and second intervals
  if wcalc.wtype == lib.wcalc.wtypes.theta:
    w = np.empty((2, frames.particles), dtype=np.int8)
  else:
    w = np.empty((2, frames.particles), dtype=np.float64)

print("#dt = %d" %framediff)
print("#n_tbvals = %d" %tbvals.size)
print("#t_a = %d" %ta)
print("#t_c = %d" %tc)

# Print information about w function calculation
wcalc.print_info()

# S4 calcuation

print("Entering S4 calculation", file=sys.stderr)

# If output files not used, write to stdout
if dumpfiles == False:
  outfile = sys.stdout

# Iterate over average interval lengths (t_b)
for index, tb in enumerate(tbvals):
  # Clear interval accumulator
  s4[:, :, :, :] = 0.0

  # Normalization factor for number of frame pairs contributing to
  # current t_b value
  norm = 0

  # Iterate over runs (FFT will be averaged over runs)
  for i in range(0, runset.n_runs):
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

      # Get starting frame for first interval and store in x0, y0, z0
      frames.get_frame(j, x0, y0, z0, i)

      # If needed, get starting frame for second interval and store in
      # x2, y2, z2
      if ta - tc != 0:
        frames.get_frame(j + ta - tc, x2, y2, z2, i)

      # Calculate w values for first interval
      wcalc.calculate_w_half(w[0], x0, y0, z0, j + tb - tc, x1, y1, z1, i)

      # Calculate w values for second interval if needed
      if ta != 0 or tc != 0:
        if ta - tc != 0:
          wcalc.calculate_w_half(w[1], x2, y2, z2, j + ta + tb, x1, y1, z1, i)
        else:
          wcalc.calculate_w_half(w[1], x0, y0, z0, j + ta + tb, x1, y1, z1, i)

      # Convert particle positions into bin numbers and wrap for
      # binning
      x0i = (x0 // cell).astype(np.int64) % size_fft
      y0i = (y0 // cell).astype(np.int64) % size_fft
      z0i = (z0 // cell).astype(np.int64) % size_fft

      # Sort first interval w values into bins for FFT
      a_bins, dummy = np.histogramdd((x0i, y0i, z0i), bins=size_fft, range=((-0.5, size_fft - 0.5), ) * 3, weights=w[0])

      # Sort second interval w values into bins for FFT if needed
      if ta != 0 or tc != 0:
        if ta - tc == 0:
          b_bins, dummy = np.histogramdd((x0i, y0i, z0i), bins=size_fft, range=((-0.5, size_fft - 0.5), ) * 3, weights=w[1])
        else:
          # Convert particle positions into bin numbers and wrap for
          # binning
          x2i = (x2 // cell).astype(np.int64) % size_fft
          y2i = (y2 // cell).astype(np.int64) % size_fft
          z2i = (z2 // cell).astype(np.int64) % size_fft

          b_bins, dummy = np.histogramdd((x2i, y2i, z2i), bins=size_fft, range=((-0.5, size_fft - 0.5), ) * 3, weights=w[1])

      # Calculate total part of S4
      if ta != 0 or tc != 0:
        run_total_s4 += fft.fftshift((fft.rfftn(a_bins) * np.conjugate(fft.rfftn(b_bins))).real, axes=(0, 1)) / frames.particles
      else:
        # Uses np.abs(), which calculates norm of complex numbers
        run_total_s4 += fft.fftshift(np.abs(fft.rfftn(a_bins))**2, axes=(0, 1)) / frames.particles

      # Multiply w values for different intervals together for self
      # bins
      if ta != 0 or tc != 0:
        w[0] *= w[1]
      else:
        # Squaring not required if w is same for each interval and is
        # boolean values, as 1*1 = 1 and 0*0 = 0
        if wcalc.wtype != lib.wcalc.wtypes.theta:
          w[0] *= w[0]

      # Calculate self part
      if ta - tc != 0:
        # Convert particle bin numbers into bin number differences
        # between starting frames of first and second intervals for
        # self part calculation and wrap for binning. Since total
        # exponential is negative, must use reverse difference with
        # positive-exponential FFT.
        x0i = (x0i - x2i) % size_fft
        y0i = (y0i - y2i) % size_fft
        z0i = (z0i - z2i) % size_fft

        # Bin multiplied w values according to coordinate differences
        self_bins, dummy = np.histogramdd((x0i, y0i, z0i), bins=size_fft, range=((-0.5, size_fft - 0.5), ) * 3, weights=w[0])

        # Perform FFT, thereby calculating self S4 for current index
        run_self_s4 += fft.fftshift(fft.rfftn(self_bins).real, axes=(0, 1)) / frames.particles
      else:
        run_self_s4[0][0][0] += np.sum(w[0]) / frames.particles

      # Accumulate the normalization value for this t_b value, which
      # will be used later in computing the mean S4 quantities for each
      # t_b
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
  s4 /= runset.n_runs

  # Calculate standard deviations from normalized variances over runs
  s4[stypes.totalstd.value] = np.sqrt(np.maximum(0.0, s4[stypes.totalstd.value] - s4[stypes.total.value]**2) / (runset.n_runs - 1))
  s4[stypes.selfstd.value] = np.sqrt(np.maximum(0.0, s4[stypes.selfstd.value] - s4[stypes.self.value]**2) / (runset.n_runs - 1))
  s4[stypes.distinctstd.value] = np.sqrt(np.maximum(0.0, s4[stypes.distinctstd.value] - s4[stypes.distinct.value]**2) / (runset.n_runs - 1))

  # Print results for current t_b

  # t_b in real units
  time_tb = tb * trajset.timestep * trajset.tbsave

  # File to write data for time to
  if dumpfiles == True:
    outfile = open("tb_%f" %(tbvals[index]), "w")

  # If q vector shells used, sort by q vector magnitude into onion
  # shells and discrete magnitudes and print the averages of values for
  # each
  if qshell_active == True:
    discrete_s4, shell_s4 = qshell.to_shells(s4)

    # Print output columns for first region disctinct q magnitudes:
    # 1 - t_b
    # 2 - q vector magnitude
    # 3 - number of q vectors with given magnitude
    # 4 - S4 total component run average
    # 5 - S4 self component run average
    # 6 - S4 distinct component run average
    # 7 - S4 total component standard deviation
    # 8 - S4 self component standard deviation
    # 9 - S4 distinct component standard deviation
    # 10 - number of frame sets in each run contributing to
    #      average of quantities
    # 11 - frame difference corresponding to t_b
    for i in range(0, discrete_s4.shape[-1]):
      outfile.write("%f %f %d %f %f %f %f %f %f %d %d\n"
                    %(time_tb,
                      qshell.qlist_discrete[i]*2*math.pi/box_size,
                      qshell.qnorm_discrete[i],
                      discrete_s4[stypes.total.value][i],
                      discrete_s4[stypes.self.value][i],
                      discrete_s4[stypes.distinct.value][i],
                      discrete_s4[stypes.totalstd.value][i],
                      discrete_s4[stypes.selfstd.value][i],
                      discrete_s4[stypes.distinctstd.value][i],
                      norm,
                      tb))

    # Print output columns for second region q magnitude onion shells:
    # 1 - t_b
    # 2 - q magnitude of midpoint of onion shells
    # 3 - number of q vectors within magnitude range of shell
    # 4 - S4 total component run average
    # 5 - S4 self component run average
    # 6 - S4 distinct component run average
    # 7 - S4 total component standard deviation
    # 8 - S4 self component standard deviation
    # 9 - S4 distinct component standard deviation
    # 10 - number of frame sets in each run contributing to
    #      average of quantities
    # 11 - frame difference corresponding to t_b
    for i in range(0, shell_s4.shape[-1]):
      outfile.write("%f %f %d %f %f %f %f %f %f %d %d\n"
                    %(time_tb,
                      (qshell.qb1a+(qshell.qlist_shells[i]+0.5)*qshell.swidth)*2*math.pi/box_size,
                      qshell.qnorm_shells[i],
                      shell_s4[stypes.total.value][i],
                      shell_s4[stypes.self.value][i],
                      shell_s4[stypes.distinct.value][i],
                      shell_s4[stypes.totalstd.value][i],
                      shell_s4[stypes.selfstd.value][i],
                      shell_s4[stypes.distinctstd.value][i],
                      norm,
                      tb))

  # If q vector shells not used, print all elements
  else:
    for i in range(0, size_fft):
      for j in range(0, size_fft):
        for k in range(0, (size_fft // 2) + 1):
          # Print output columns:
          # 1 - t_b
          # 2 - x component of fft frequency
          # 3 - y component of fft frequency
          # 4 - z component of fft frequency
          # 5 - S4 total component run average
          # 6 - S4 self component run average
          # 7 - S4 distinct component run average
          # 8 - S4 total component standard deviation
          # 9 - S4 self component standard deviation
          # 10 - S4 distinct component standard deviation
          # 11 - number of frame sets in each run contributing to
          #      average of quantities
          # 12 - frame difference corresponding to t_b
          outfile.write("%f %f %f %f %f %f %f %f %f %f %d %d\n"
                        %(time_tb,
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
                          tb))

  # If output files for each t_b value used, close file for this t_b
  # value
  if dumpfiles == True:
    outfile.close()
