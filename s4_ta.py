import numpy as np
import pydcd
import sys
import math

# Import functionality from local library directory
import lib.script
import lib.opentraj
import lib.progression
import lib.frame
import lib.wcalc
import lib.qshell

# Script-specific variables altered by arguments
class svars:
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

# Script-specific short options
shortopts = "k:d:x:y:b:c:ij"

# Description of script-specific arguments
argtext = "-k Last frame number in range to use for initial times (index starts at 1)\n" \
          + "-d Spacing between initial times (dt)\n" \
          + "-x Number of Fourier transform vector lengths to be used in each direction\n" \
          + "-y Box size in each dimension (assumed to be cubic, required)\n" \
          + "-b Average interval in frames (t_b, default=0)\n" \
          + "-c Difference between intervals in frames (t_c, default=0)\n" \
          + "-i Write output to files, one for each lag time\n" \
          + "If no q vector shell options specified, all q vector values printed"

def catch_opt(o, a, svars):
  if o == "-k":
    svars.initend = int(a)
  elif o == "-d":
    svars.framediff = int(a)
  elif o == "-x":
    svars.size_ft = int(a)
  elif o == "-y":
    svars.box_size = float(a)
  elif o == "-b":
    svars.tb = int(a)
  elif o == "-c":
    svars.tc = int(a)
  elif o == "-i":
    svars.dumpfiles = True
  elif o == "-j":
    print("-j is default, no need to specify", file=sys.stderr)
  else:
    return False

  return True

def main_func(svars, prog, runset, trajset, frames, wcalc, qshell):
  if svars.box_size == None:
    raise RuntimeError("Must define box size dimensions")

  if svars.size_ft == None:
    raise RuntimeError("Must specify number of values for each q vector component")

  if runset.n_runs <= 1:
    raise RuntimeError("Must have at least 2 runs")

  # Open trajectory files
  trajset.opentraj()

  # Prepare frames object for calculation
  frames.prepare()

  # Verify correctness of parameters for w calculation from arguments
  wcalc.prepare()

  # Generate qshell elements if onion shells are used, used for sorting
  # values into shells
  if qshell.active == True:
    qshell.prepare(2*svars.size_ft - 1, svars.box_size)

  # Print basic properties shared across the files
  print("#nset: %d" %trajset.fileframes[-1])
  print("#N: %d" %trajset.fparticles)
  print("#timestep: %f" %trajset.timestep)
  print("#tbsave: %f" %trajset.tbsave)

  # End of set of frames to use for initial times
  if svars.initend == None:
    svars.initend = frames.final
  else:
    if svars.initend > frames.final:
      raise RuntimeError("End initial time frame beyond set of analyzed frames")

  # Largest possible positive and negative lags
  prog.max_val = frames.n_frames - 1 - max(svars.tb, svars.tb - svars.tc)
  prog.min_val = -((svars.framediff * ((frames.n_frames - 1 - (svars.tb - 2 * svars.tc)) // svars.framediff)) - svars.tc)

  # Construct progression of interval values using previously-specified
  # parameters
  lags = prog.construct()

  # Indices for types/components of S4 function
  totalcomp = 0
  selfcomp = 1
  distinctcomp = 2
  totalcompstd = 3
  selfcompstd = 4
  distinctcompstd = 5
  n_stypes = 6

  # Array with progression of q values to use, with 0.0 always at index
  # 0. All of these create integral number of wave periods inside the
  # box. Full array has both positive and negative values.
  qs = np.linspace(0.0, (svars.size_ft - 1) * 2 * math.pi / svars.box_size, num=svars.size_ft)
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
  run_self_s4 = np.empty((2*svars.size_ft - 1, 2*svars.size_ft - 1, svars.size_ft), dtype=np.float64)

  # Temporary value for each run to allow for calculation of each run's
  # total component of S4
  run_total_s4 = np.empty((2*svars.size_ft - 1, 2*svars.size_ft - 1, svars.size_ft), dtype=np.float64)

  # Temporary value for each run to allow for calculation of each frame
  # pair's total component of S4. In first dimension, first index is
  # real and second is imaginary
  lag_total_s4 = np.empty((2, 2*svars.size_ft - 1, 2*svars.size_ft - 1, svars.size_ft), dtype=np.float64)

  # Array for S4 values. The first and second q vector dimensions
  # include values for negative vectors. Since all inputs are real,
  # this is not required for the third fft dimension, as the values
  # would be the same for a vector in the exact opposite direction
  # (with all vector components of opposite sign).
  s4 = np.zeros((n_stypes, 2*svars.size_ft - 1, 2*svars.size_ft - 1, svars.size_ft), dtype=np.float64)

  # Arrays for temporarily holding values of cosines and sines of
  # particle positions for a given set of q values, used to calculate
  # full matrix of S4 self part values for each q vector.
  ft_x_edge = np.empty((2, svars.size_ft, frames.particles), dtype=np.float64)
  ft_y_edge = np.empty((2, svars.size_ft, frames.particles), dtype=np.float64)
  ft_z_edge = np.empty((2, svars.size_ft, frames.particles), dtype=np.float64)

  # W function values for each particle and for both initial and end
  # values
  if wcalc.wtype == lib.wcalc.wtypes.theta:
    w = np.empty((2, frames.particles), dtype=np.int8)
  else:
    w = np.empty((2, frames.particles), dtype=np.float64)

  print("#dt = %d" %svars.framediff)
  print("#n_lags = %d" %lags.size)
  print("#t_b = %d" %svars.tb)
  print("#t_c = %d" %svars.tc)

  # Print information about w function calculation
  wcalc.print_info()

  # Create legend with description of output columns
  legend = "#\n" \
           + "#Output Columns:\n" \
           + "#  1 - t_a\n"
  if qshell.active == True:
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
  if svars.dumpfiles == False:
    outfile = sys.stdout
    outfile.write(legend)

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
      for j in np.arange(0, svars.initend - frames.start, svars.framediff):
        # Use only indices that are within range
        if (ta < (svars.tc - j) or
            ta - frames.n_frames >= (svars.tc - j) or
            ta < (-svars.tb - j) or
            ta - frames.n_frames >= (-svars.tb - j) or
            j < (svars.tc - svars.tb) or
            j - frames.n_frames >= (svars.tc - svars.tb)):
          continue

        # Get particle coordinates and calculate w values for first and
        # second intervals
        wcalc.calculate_w(w[0], j, x0, y0, z0, j + svars.tb - svars.tc, x1, y1, z1, i)
        wcalc.calculate_w(w[1], j + ta - svars.tc, x2, y2, z2, j + ta + svars.tb, x1, y1, z1, i)

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
        for k in range(0, svars.size_ft):
          for l in range(0, svars.size_ft):
            for m in range(0, svars.size_ft):
              # Calculate real components of S4 that are even and odd
              # with regard to signs of different components of q
              even_s4r = np.mean(w[0] * w[1] * ft_x_edge[0][k] * ft_y_edge[0][l] * ft_z_edge[0][m])
              even_s4r -= np.mean(w[0] * w[1] * ft_x_edge[1][k] * ft_y_edge[1][l] * ft_z_edge[0][m])
              oddx_s4r = -np.mean(w[0] * w[1] * ft_x_edge[1][k] * ft_y_edge[0][l] * ft_z_edge[1][m])
              oddy_s4r = -np.mean(w[0] * w[1] * ft_x_edge[0][k] * ft_y_edge[1][l] * ft_z_edge[1][m])

              # Complete 4 corresponding values of S4 by combining
              # computed values
              run_self_s4[svars.size_ft-1+k,svars.size_ft-1+l,m] += even_s4r + oddx_s4r + oddy_s4r
              if l != 0:
                run_self_s4[svars.size_ft-1+k,svars.size_ft-1-l,m] += even_s4r + oddx_s4r - oddy_s4r
              if k != 0:
                run_self_s4[svars.size_ft-1-k,svars.size_ft-1+l,m] += even_s4r - oddx_s4r + oddy_s4r
              if l != 0 and k != 0:
                run_self_s4[svars.size_ft-1-k,svars.size_ft-1-l,m] += even_s4r - oddx_s4r - oddy_s4r

        # Generate edges for 3-dimensional Fourier transform for end
        # term of total part of S4
        for k, q in enumerate(qs):
          ft_x_edge[0][k] = np.cos(q * -x2)
          ft_x_edge[1][k] = np.sin(q * -x2)
          ft_y_edge[0][k] = np.cos(q * -y2)
          ft_y_edge[1][k] = np.sin(q * -y2)
          ft_z_edge[0][k] = np.cos(q * -z2)
          ft_z_edge[1][k] = np.sin(q * -z2)

        # Calculate end term of total part of S4 using edge arrays
        for k in range(0, svars.size_ft):
          for l in range(0, svars.size_ft):
            for m in range(0, svars.size_ft):
              # Calculate real and imaginary components of S4 that are
              # even and odd with regard to signs of different
              # components of q
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
              lag_total_s4[0][svars.size_ft-1+k,svars.size_ft-1+l,m] = even_s4r + oddx_s4r + oddy_s4r
              lag_total_s4[1][svars.size_ft-1+k,svars.size_ft-1+l,m] = even_s4i + oddx_s4i + oddy_s4i
              if l != 0:
                lag_total_s4[0][svars.size_ft-1+k,svars.size_ft-1-l,m] = even_s4r + oddx_s4r - oddy_s4r
                lag_total_s4[1][svars.size_ft-1+k,svars.size_ft-1-l,m] = even_s4i + oddx_s4i - oddy_s4i
              if k != 0:
                lag_total_s4[0][svars.size_ft-1-k,svars.size_ft-1+l,m] = even_s4r - oddx_s4r + oddy_s4r
                lag_total_s4[1][svars.size_ft-1-k,svars.size_ft-1+l,m] = even_s4i - oddx_s4i + oddy_s4i
              if l != 0 and k != 0:
                lag_total_s4[0][svars.size_ft-1-k,svars.size_ft-1-l,m] = even_s4r - oddx_s4r - oddy_s4r
                lag_total_s4[1][svars.size_ft-1-k,svars.size_ft-1-l,m] = even_s4i - oddx_s4i - oddy_s4i

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
        for k in range(0, svars.size_ft):
          for l in range(0, svars.size_ft):
            for m in range(0, svars.size_ft):
              # Calculate real components of S4 that are even and odd
              # with regard to signs of different components of q
              even_s4r = np.sum(w[0] * ft_x_edge[0][k] * ft_y_edge[0][l] * ft_z_edge[0][m])
              even_s4r -= np.sum(w[0] * ft_x_edge[1][k] * ft_y_edge[1][l] * ft_z_edge[0][m])
              oddx_s4r = -np.sum(w[0] * ft_x_edge[1][k] * ft_y_edge[0][l] * ft_z_edge[1][m])
              oddy_s4r = -np.sum(w[0] * ft_x_edge[0][k] * ft_y_edge[1][l] * ft_z_edge[1][m])

              # Complete and multiply 4 corresponding values of S4 by
              # combining computed values
              lag_total_s4[0][svars.size_ft-1+k,svars.size_ft-1+l,m] *= even_s4r + oddx_s4r + oddy_s4r
              if l != 0:
                lag_total_s4[0][svars.size_ft-1+k,svars.size_ft-1-l,m] *= even_s4r + oddx_s4r - oddy_s4r
              if k != 0:
                lag_total_s4[0][svars.size_ft-1-k,svars.size_ft-1+l,m] *= even_s4r - oddx_s4r + oddy_s4r
              if l != 0 and k != 0:
                lag_total_s4[0][svars.size_ft-1-k,svars.size_ft-1-l,m] *= even_s4r - oddx_s4r - oddy_s4r

        # Accumulate computed total S4 for lag value to value for run
        run_total_s4 += lag_total_s4[0]

        # Accumulate the normalization value for this lag, which will
        # be used later in computing the mean S4 quantities for each
        # lag
        if i == 0:
          norm += 1

      # Calculate distinct part of S4 for current run
      run_distinct_s4 = run_total_s4 - run_self_s4

      # Normalize the accumulated values, thereby obtaining averages
      # over each pair of frames
      run_total_s4 /= norm
      run_self_s4 /= norm
      run_distinct_s4 /= norm

      # Accumulate total, self, and distinct averages for run
      s4[totalcomp] += run_total_s4
      s4[selfcomp] += run_self_s4
      s4[distinctcomp] += run_distinct_s4

      # Accumulate squares of total, self, and distinct averages for
      # run, holding variances for eventual calculation of standard
      # deviation
      s4[totalcompstd] += run_total_s4**2
      s4[selfcompstd] += run_self_s4**2
      s4[distinctcompstd] += run_distinct_s4**2

    # Normalize S4 values across runs
    s4 /= runset.n_runs

    # Calculate standard deviations from normalized variances over runs
    s4[totalcompstd] = np.sqrt(np.maximum(0.0, s4[totalcompstd] - s4[totalcomp]**2) / (runset.n_runs - 1))
    s4[selfcompstd] = np.sqrt(np.maximum(0.0, s4[selfcompstd] - s4[selfcomp]**2) / (runset.n_runs - 1))
    s4[distinctcompstd] = np.sqrt(np.maximum(0.0, s4[distinctcompstd] - s4[distinctcomp]**2) / (runset.n_runs - 1))

    # Print results for current lag

    # Lag time in real units
    time_ta = ta * trajset.timestep * trajset.tbsave

    # If output files used, open file for current lag
    if svars.dumpfiles == True:
      outfile = open("lag_%f" %(lags[index]), "w")
      outfile.write(legend)

    # If q vector shells used, sort by q vector magnitude into onion
    # shells and discrete magnitudes and print the averages of values
    # for each
    if qshell.active == True:
      discrete_s4, shell_s4 = qshell.to_shells(s4)

      # Print output columns for first region disctinct q magnitudes
      for i in range(0, discrete_s4.shape[-1]):
        outfile.write("%f %f %d %f %f %f %f %f %f %d %d\n"
                      %(time_ta,
                        qshell.qlist_discrete[i]*2*math.pi/svars.box_size,
                        qshell.qnorm_discrete[i],
                        discrete_s4[totalcomp][i],
                        discrete_s4[selfcomp][i],
                        discrete_s4[distinctcomp][i],
                        discrete_s4[totalcompstd][i],
                        discrete_s4[selfcompstd][i],
                        discrete_s4[distinctcompstd][i],
                        norm,
                        ta))

      # Print output columns for second region q magnitude onion shells
      for i in range(0, shell_s4.shape[-1]):
        outfile.write("%f %f %d %f %f %f %f %f %f %d %d\n"
                      %(time_ta,
                        (qshell.qb1a+(qshell.qlist_shells[i]+0.5)*qshell.swidth)*2*math.pi/svars.box_size,
                        qshell.qnorm_shells[i],
                        shell_s4[totalcomp][i],
                        shell_s4[selfcomp][i],
                        shell_s4[distinctcomp][i],
                        shell_s4[totalcompstd][i],
                        shell_s4[selfcompstd][i],
                        shell_s4[distinctcompstd][i],
                        norm,
                        ta))

    # If q vector shells not used, print output columns for all q
    # vectors
    else:
      for i in range(0, 2*svars.size_ft - 1):
        for j in range(0, 2*svars.size_ft - 1):
          for k in range(0, svars.size_ft):
            outfile.write("%f %f %f %f %f %f %f %f %f %f %d %d\n"
                          %(time_ta,
                            qs_full[i],
                            qs_full[j],
                            qs[k],
                            s4[totalcomp][i][j][k],
                            s4[selfcomp][i][j][k],
                            s4[distinctcomp][i][j][k],
                            s4[totalcompstd][i][j][k],
                            s4[selfcompstd][i][j][k],
                            s4[distinctcompstd][i][j][k],
                            norm,
                            ta))

    # If output files for each lag used, close file for this lag
    if svars.dumpfiles == True:
      outfile.close()

# Run full program
lib.script.run(main_func,
               argtext=argtext,
               svars=svars,
               shortopts=shortopts,
               catch_opt=catch_opt,
               modules=[prog, runset, trajset, frames, wcalc, qshell])
