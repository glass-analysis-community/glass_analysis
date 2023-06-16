import numpy as np
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
# Number of Fourier transform vector constants (including q=0)
size_ft = None
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
        "-x Number of Fourier transform vector lengths to be used in each direction in addition to q=0",
        "-y Box size in each dimension (assumed to be cubic, required)",
        "-b Average interval in frames (t_b)",
        "-c Difference between intervals in frames (t_c)",
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
    size_ft = int(a) + 1
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

if size_ft == None:
  raise RuntimeError("Must specify number of values for each q vector component")

# Open trajectory files
trajset.opentraj_multirun(1, True)

# Prepare frames object for calculation
frames.prepare()

# Verify correctness of parameters for w calculation from arguments
wcalc.prepare()

# Generate qshell elements if onion shells are used, used for sorting
# values into shells
if qshell_active == True:
  qshell.prepare(2*size_ft - 1, box_size)

# Print basic properties shared across the files
print("#nset: %d" %trajset.fileframes[-1])
print("#N: %d" %trajset.fparticles)
print("#timestep: %f" %trajset.timestep)
print("#tbsave: %f" %trajset.tbsave)

# End of set of frames to use for initial times
if initend == None:
  initend = trajset.fileframes[-1]
else:
  if initend > trajset.fileframes[-1]:
    raise RuntimeError("End initial time frame beyond set of frames")

# Largest possible positive and negative lags
prog.max_val = frames.n_frames - 1 - max(tb, tb - tc)
prog.min_val = -((framediff * ((frames.n_frames - 1 - (tb - 2 * tc)) // framediff)) - tc)

# Construct progression of interval values using previously-specified
# parameters
lags = prog.construct()

# Array with progression of q values to use, with 0.0 always at index 0.
# All of these create integral number of wave periods inside the box.
# Full array has both positive and negative values.
qs = np.linspace(0.0, (size_ft - 1) * 2 * math.pi / box_size, num=size_ft)
qs_full = np.concatenate((-np.flip(qs[1:]), qs))

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

# Temporary value for each run to allow for calculation of each run's
# self component of S4
run_self_s4 = np.empty((2*size_ft - 1, 2*size_ft - 1, size_ft), dtype=np.float64)

# Temporary value for each run to allow for calculation of each run's
# total component of S4
run_total_s4 = np.empty((2*size_ft - 1, 2*size_ft - 1, size_ft), dtype=np.float64)

# Temporary value for each run to allow for calculation of each frame
# pair's total component of S4. In first dimension, first index is
# real and second is imaginary
lag_total_s4 = np.empty((2, 2*size_ft - 1, 2*size_ft - 1, size_ft), dtype=np.float64)

# Array for S4 values. The first and second q vector dimensions include
# values for negative vectors. Since all inputs are real, this is not
# required for the third fft dimension, as the values would be the same
# for a vector in the exact opposite direction (with all vector
# components of opposite sign).
s4 = np.zeros((n_stypes, 2*size_ft - 1, 2*size_ft - 1, size_ft), dtype=np.float64)

# Arrays for temporarily holding values of cosines and sines of
# particle positions for a given set of q values, used to calculate
# full matrix of S4 self part values for each q vector.
ft_x_edge = np.empty((2, size_ft, frames.particles), dtype=np.float64)
ft_y_edge = np.empty((2, size_ft, frames.particles), dtype=np.float64)
ft_z_edge = np.empty((2, size_ft, frames.particles), dtype=np.float64)

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

# Print information about w function calculation
wcalc.print_info()

# Find center of mass for each frame
print("Finding centers of mass for frames", file=sys.stderr)
frames.generate_cm()

# S4 calcuation

print("Entering S4 calculation", file=sys.stderr)

# If output files not used, write to stdout
if dumpfiles == False:
  outfile = sys.stdout

# Iterate over lags (t_a)
for index, ta in enumerate(lags):
  # Clear lag accumulator
  s4[:, :, :, :] = 0.0

  # Normalization factor for number of frame pairs contributing to
  # current lag value
  norm = 0

  # Iterate over runs
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

      # Complex multiplication is simulated with sin and cos, since
      # using complex numbers with numpy imposes a large performance
      # penalty.

      # Generate edges for 3-dimensional Fourier transform for self
      # part
      for k, q in enumerate(qs):
        ft_x_edge[0][k] = np.cos(q * (x0 - x2))
        ft_x_edge[1][k] = np.sin(q * (x0 - x2))
        ft_y_edge[0][k] = np.cos(q * (y0 - y2))
        ft_y_edge[1][k] = np.sin(q * (y0 - y2))
        ft_z_edge[0][k] = np.cos(q * (z0 - z2))
        ft_z_edge[1][k] = np.sin(q * (z0 - z2))

      # Calculate self part of S4 using edge arrays
      for k in range(0, size_ft):
        for l in range(0, size_ft):
          for m in range(0, size_ft):
            # Calculate real components of S4 that are even and odd
            # with regard to signs of different components of q
            even_s4r = np.mean(w[0] * w[1] * ft_x_edge[0][k] * ft_y_edge[0][l] * ft_z_edge[0][m])
            even_s4r -= np.mean(w[0] * w[1] * ft_x_edge[1][k] * ft_y_edge[1][l] * ft_z_edge[0][m])
            oddx_s4r = -np.mean(w[0] * w[1] * ft_x_edge[1][k] * ft_y_edge[0][l] * ft_z_edge[1][m])
            oddy_s4r = -np.mean(w[0] * w[1] * ft_x_edge[0][k] * ft_y_edge[1][l] * ft_z_edge[1][m])

            # Complete 4 corresponding values of S4 by combining
            # computed values
            run_self_s4[size_ft-1+k,size_ft-1+l,m] += even_s4r + oddx_s4r + oddy_s4r
            if l != 0:
              run_self_s4[size_ft-1+k,size_ft-1-l,m] += even_s4r + oddx_s4r - oddy_s4r
            if k != 0:
              run_self_s4[size_ft-1-k,size_ft-1+l,m] += even_s4r - oddx_s4r + oddy_s4r
            if l != 0 and k != 0:
              run_self_s4[size_ft-1-k,size_ft-1-l,m] += even_s4r - oddx_s4r - oddy_s4r

      # Generate edges for 3-dimensional Fourier transform for end term
      # of total part of S4
      for k, q in enumerate(qs):
        ft_x_edge[0][k] = np.cos(q * -x2)
        ft_x_edge[1][k] = np.sin(q * -x2)
        ft_y_edge[0][k] = np.cos(q * -y2)
        ft_y_edge[1][k] = np.sin(q * -y2)
        ft_z_edge[0][k] = np.cos(q * -z2)
        ft_z_edge[1][k] = np.sin(q * -z2)

      # Calculate end term of total part of S4 using edge arrays
      for k in range(0, size_ft):
        for l in range(0, size_ft):
          for m in range(0, size_ft):
            # Calculate real and imaginary components of S4 that are
            # even and odd with regard to signs of different components
            # of q
            even_s4r = np.mean(w[0] * ft_x_edge[0][k] * ft_y_edge[0][l] * ft_z_edge[0][m])
            even_s4r -= np.mean(w[0] * ft_x_edge[1][k] * ft_y_edge[1][l] * ft_z_edge[0][m])
            oddx_s4r = -np.mean(w[0] * ft_x_edge[1][k] * ft_y_edge[0][l] * ft_z_edge[1][m])
            oddy_s4r = -np.mean(w[0] * ft_x_edge[0][k] * ft_y_edge[1][l] * ft_z_edge[1][m])
            even_s4i = -np.mean(w[0] * ft_x_edge[1][k] * ft_y_edge[1][l] * ft_z_edge[1][m])
            even_s4i += np.mean(w[0] * ft_x_edge[0][k] * ft_y_edge[0][l] * ft_z_edge[1][m])
            oddx_s4i = np.mean(w[0] * ft_x_edge[1][k] * ft_y_edge[0][l] * ft_z_edge[0][m])
            oddy_s4i = np.mean(w[0] * ft_x_edge[0][k] * ft_y_edge[1][l] * ft_z_edge[0][m])

            # Complete 4 corresponding values of S4 by combining
            # computed values
            lag_total_s4[0][size_ft-1+k,size_ft-1+l,m] = even_s4r + oddx_s4r + oddy_s4r
            lag_total_s4[1][size_ft-1+k,size_ft-1+l,m] = even_s4i + oddx_s4i + oddy_s4i
            if l != 0:
              lag_total_s4[0][size_ft-1+k,size_ft-1-l,m] = even_s4r + oddx_s4r - oddy_s4r
              lag_total_s4[1][size_ft-1+k,size_ft-1-l,m] = even_s4i + oddx_s4i - oddy_s4i
            if k != 0:
              lag_total_s4[0][size_ft-1-k,size_ft-1+l,m] = even_s4r - oddx_s4r + oddy_s4r
              lag_total_s4[1][size_ft-1-k,size_ft-1+l,m] = even_s4i - oddx_s4i + oddy_s4i
            if l != 0 and k != 0:
              lag_total_s4[0][size_ft-1-k,size_ft-1-l,m] = even_s4r - oddx_s4r - oddy_s4r
              lag_total_s4[1][size_ft-1-k,size_ft-1-l,m] = even_s4i - oddx_s4i - oddy_s4i

      # Generate edges for 3-dimensional Fourier transform for start
      # term of total part of S4
      for k, q in enumerate(qs):
        ft_x_edge[0][k] = np.cos(q * x0)
        ft_x_edge[1][k] = np.sin(q * x0)
        ft_y_edge[0][k] = np.cos(q * y0)
        ft_y_edge[1][k] = np.sin(q * y0)
        ft_z_edge[0][k] = np.cos(q * z0)
        ft_z_edge[1][k] = np.sin(q * z0)

      # Mutliply total part of S4 by start term using edge arrays.
      # Calculate only real part, as only the real part of the result
      # is used
      for k in range(0, size_ft):
        for l in range(0, size_ft):
          for m in range(0, size_ft):
            # Calculate real components of S4 that are even and odd
            # with regard to signs of different components of q
            even_s4r = np.sum(w[0] * ft_x_edge[0][k] * ft_y_edge[0][l] * ft_z_edge[0][m])
            even_s4r -= np.sum(w[0] * ft_x_edge[1][k] * ft_y_edge[1][l] * ft_z_edge[0][m])
            oddx_s4r = -np.sum(w[0] * ft_x_edge[1][k] * ft_y_edge[0][l] * ft_z_edge[1][m])
            oddy_s4r = -np.sum(w[0] * ft_x_edge[0][k] * ft_y_edge[1][l] * ft_z_edge[1][m])

            # Complete and multiply 4 corresponding values of S4 by
            # combining computed values
            lag_total_s4[0][size_ft-1+k,size_ft-1+l,m] *= even_s4r + oddx_s4r + oddy_s4r
            if l != 0:
              lag_total_s4[0][size_ft-1+k,size_ft-1-l,m] *= even_s4r + oddx_s4r - oddy_s4r
            if k != 0:
              lag_total_s4[0][size_ft-1-k,size_ft-1+l,m] *= even_s4r - oddx_s4r + oddy_s4r
            if l != 0 and k != 0:
              lag_total_s4[0][size_ft-1-k,size_ft-1-l,m] *= even_s4r - oddx_s4r - oddy_s4r

      # Accumulate computed total S4 for lag value to value for run
      run_total_s4 += lag_total_s4[0]

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

  # If q vector shells used, sort by q vector magnitude into onion
  # shells and discrete magnitudes and print the averages of values for
  # each
  if qshell_active == True:
    discrete_s4, shell_s4 = qshell.to_shells(s4)

    # Print output columns for first region disctinct q magnitudes:
    # 1 - t_a
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
    # 11 - frame difference corresponding to t_a
    for i in range(0, discrete_s4.shape[-1]):
      outfile.write("%f %f %d %f %f %f %f %f %f %d %d\n"
                    %(time_ta,
                      qshell.qlist_discrete_sorted[i]*2*math.pi/box_size,
                      qshell.qnorm_discrete_sorted[i],
                      discrete_s4[stypes.total.value][i],
                      discrete_s4[stypes.self.value][i],
                      discrete_s4[stypes.distinct.value][i],
                      discrete_s4[stypes.totalstd.value][i],
                      discrete_s4[stypes.selfstd.value][i],
                      discrete_s4[stypes.distinctstd.value][i],
                      norm,
                      ta))

    # Print output columns for second region q magnitude onion shells:
    # 1 - t_a
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
    # 11 - frame difference corresponding to t_a
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

  # If q vector shells not used, print all elements
  else:
    for i in range(0, 2*size_ft - 1):
      for j in range(0, 2*size_ft - 1):
        for k in range(0, size_ft):
          # Print output columns:
          # 1 - t_a
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
          # 12 - frame difference corresponding to t_a
          outfile.write("%f %f %f %f %f %f %f %f %f %f %d %d\n"
                        %(time_ta,
                          qs_full[i],
                          qs_full[j],
                          qs[k],
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
