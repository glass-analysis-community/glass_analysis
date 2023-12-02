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
# Average length of intervals (t_b)
tb = 1
# Half difference between length of initial and end intervals (t_c)
tc = 0
# Whether to write output to files rather than stdout
dumpfiles = False
# Progression specification/generation object for lags
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
        "-x Dimensionality of FFT matrix, length in each dimension",
        "-y Box size in each dimension (assumed to be cubic, required)",
        "-b Average interval in frames (t_b, default=1)",
        "-c Difference between intervals in frames (t_c, default=0)",
        "-i Write output to files, one for each lag time",
        "-h Print usage",
        sep="\n", file=sys.stderr)
  prog.usage()
  wcalc.usage()
  qshell.usage()
  print("If no q vector shell options specified, all q vector values printed", file=sys.stderr)

try:
  opts, args = getopt.gnu_getopt(sys.argv[1:], "k:d:x:y:b:c:ijh" +
                                               runset.shortopts +
                                               trajset.shortopts +
                                               prog.shortopts +
                                               frames.shortopts +
                                               wcalc.shortopts +
                                               qshell.shortopts,
                                               trajset.longopts +
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
    size_fft = int(a)
  elif o == "-y":
    box_size = float(a)
  elif o == "-b":
    tb = int(a)
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
  initend = frames.final
else:
  if initend > frames.final:
    raise RuntimeError("End initial time frame beyond set of analyzed frames")

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

# Bins for particle positions and computing spatial FFT
a_bins = np.zeros((size_fft, size_fft, size_fft), dtype=np.float64)
b_bins = np.zeros((size_fft, size_fft, size_fft), dtype=np.float64)
self_bins = np.empty((size_fft, size_fft, size_fft), dtype=np.float64)

# Temporary value for each run to allow for calculation of each run's
# self component of S4
run_self_s4 = np.empty((size_fft, size_fft, (size_fft // 2) + 1), dtype=np.float64)

# Temporary value for each run to allow for calculation of each run's
# total component of S4
run_total_s4 = np.empty((size_fft, size_fft, (size_fft // 2) + 1), dtype=np.float64)

# Array for S4 values. The first and second fft dimensions include
# values for negative vectors. Since all inputs are real, this is not
# required for the third fft dimension, as the values would be the same
# for a vector in the exact opposite direction (with all vector
# components of opposite sign).
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
print("#size_fft: %f" %size_fft)

# Print information about w function calculation
wcalc.print_info()

# Create legend with description of output columns
legend = "#\n" \
         + "#Output Columns:\n" \
         + "#  1 - t_a\n"
if qshell_active == True:
  legend += "#  2 - q vector magnitude (in first region) or midpoint of q onion shell (in second region)\n" \
            + "#  3 - Number of q vectors with given magnitude or in given shell\n"
  col_offset1 = 3
else:
  legend += "#  2 - x component of q vector\n" \
            + "#  3 - y component of q vector\n" \
            + "#  4 - z component of q vector\n"
  col_offset1 = 4
legend += "#  %d - Run average of total part of S4\n" %(col_offset1 + 1) \
          + "#  %d - Run average of self part of S4\n" %(col_offset1 + 2) \
          + "#  %d - Run average of distinct part of S4\n" %(col_offset1 + 3) \
          + "#  %d - Standard deviation across runs of total part of S4\n" %(col_offset1 + 4) \
          + "#  %d - Standard deviation across runs of self part of S4\n" %(col_offset1 + 5) \
          + "#  %d - Standard deviation across runs of distinct part of S4\n" %(col_offset1 + 6) \
          + "#  %d - Number of frame sets in each run contributing to average of quantities\n" %(col_offset1 + 7) \
          + "#  %d - Frame difference corresponding to t_a\n" %(col_offset1 + 8) \
          + "#\n"

# S4 calcuation

print("Entering S4 calculation", file=sys.stderr)

# If output files not used, write to stdout
if dumpfiles == False:
  outfile = sys.stdout
  outfile.write(legend)

# Iterate over lags (t_a)
for index, ta in enumerate(lags):
  # Clear lag accumulator
  s4[:, :, :, :] = 0.0

  # Normalization factor for number of frame pairs contributing to
  # current lag value
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

      # Get particle coordinates and calculate w values for first and
      # second intervals
      wcalc.calculate_w(w[0], j, x0, y0, z0, j + tb - tc, x1, y1, z1, i)
      wcalc.calculate_w(w[1], j + ta - tc, x2, y2, z2, j + ta + tb, x1, y1, z1, i)

      # Convert particle positions into bin numbers and wrap for
      # binning
      x0i = (x0 // cell) % size_fft
      y0i = (y0 // cell) % size_fft
      z0i = (z0 // cell) % size_fft
      x2i = (x2 // cell) % size_fft
      y2i = (y2 // cell) % size_fft
      z2i = (z2 // cell) % size_fft

      # Sort first interval w values into bins for FFT
      a_bins = np.histogramdd((x0i, y0i, z0i), bins=size_fft, range=((-0.5, size_fft - 0.5), ) * 3, weights=w[0])[0]

      # Sort second interval w values into bins for FFT
      b_bins = np.histogramdd((x2i, y2i, z2i), bins=size_fft, range=((-0.5, size_fft - 0.5), ) * 3, weights=w[1])[0]

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
      self_bins = np.histogramdd((x0i, y0i, z0i), bins=size_fft, range=((-0.5, size_fft - 0.5), ) * 3, weights=w[0])[0]

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
  s4 /= runset.n_runs

  # Calculate standard deviations from normalized variances over runs
  s4[stypes.totalstd.value] = np.sqrt(np.maximum(0.0, s4[stypes.totalstd.value] - s4[stypes.total.value]**2) / (runset.n_runs - 1))
  s4[stypes.selfstd.value] = np.sqrt(np.maximum(0.0, s4[stypes.selfstd.value] - s4[stypes.self.value]**2) / (runset.n_runs - 1))
  s4[stypes.distinctstd.value] = np.sqrt(np.maximum(0.0, s4[stypes.distinctstd.value] - s4[stypes.distinct.value]**2) / (runset.n_runs - 1))

  # Print results for current lag

  # Lag time in real units
  time_ta = ta * trajset.timestep * trajset.tbsave

  # If output files used, open file for current lag
  if dumpfiles == True:
    outfile = open("lag_%f" %(lags[index]), "w")
    outfile.write(legend)

  # If q vector shells used, sort by q vector magnitude into onion
  # shells and discrete magnitudes and print the averages of values for
  # each
  if qshell_active == True:
    discrete_s4, shell_s4 = qshell.to_shells(s4)

    # Print output columns for first region disctinct q magnitudes
    for i in range(0, discrete_s4.shape[-1]):
      outfile.write("%f %f %d %f %f %f %f %f %f %d %d\n"
                    %(time_ta,
                      qshell.qlist_discrete[i]*2*math.pi/box_size,
                      qshell.qnorm_discrete[i],
                      discrete_s4[stypes.total.value][i],
                      discrete_s4[stypes.self.value][i],
                      discrete_s4[stypes.distinct.value][i],
                      discrete_s4[stypes.totalstd.value][i],
                      discrete_s4[stypes.selfstd.value][i],
                      discrete_s4[stypes.distinctstd.value][i],
                      norm,
                      ta))

    # Print output columns for second region q magnitude onion shells
    for i in range(0, shell_s4.shape[-1]):
      outfile.write("%f %f %d %f %f %f %f %f %f %d %d\n"
                    %(time_ta,
                      (qshell.qb1a+(qshell.qlist_shells[i]+0.5)*qshell.swidth)*2*math.pi/box_size,
                      qshell.qnorm_shells[i],
                      shell_s4[stypes.total.value][i],
                      shell_s4[stypes.self.value][i],
                      shell_s4[stypes.distinct.value][i],
                      shell_s4[stypes.totalstd.value][i],
                      shell_s4[stypes.selfstd.value][i],
                      shell_s4[stypes.distinctstd.value][i],
                      norm,
                      ta))

  # If q vector shells not used, print output columns for all q vectors
  else:
    for i in range(0, size_fft):
      for j in range(0, size_fft):
        for k in range(0, (size_fft // 2) + 1):
          outfile.write("%f %f %f %f %f %f %f %f %f %f %d %d\n"
                        %(time_ta,
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
                          ta))

  # If output files for each lag used, close file for this lag
  if dumpfiles == True:
    outfile.close()
