import numpy as np
from numpy import fft
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
  # Limit of number of Fourier transform vector constants (including
  # q=0)
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

# Script-specific short options
shortopts = "k:d:x:y:a:c:ij"

# Description of script-specific arguments
argtext = "-k Last frame number in range to use for initial times (index starts at 1)\n" \
          + "-d Spacing between initial times (dt)\n" \
          + "-x Dimensionality of FFT matrix, length in each dimension\n" \
          + "-y Box size in each dimension (assumed to be cubic, required)\n" \
          + "-a Offset between centers of begginning and end intervals in frames (t_a, default=0)\n" \
          + "-c Difference between intervals in frames (t_c, default=0)\n" \
          + "-i Write output to files, one for each t_b value\n" \
          + "If no q vector shell options specified, all q vector values printed"

def catch_opt(o, a, svars):
  if o == "-k":
    svars.initend = int(a)
  elif o == "-d":
    svars.framediff = int(a)
  elif o == "-x":
    svars.size_fft = int(a)
  elif o == "-y":
    svars.box_size = float(a)
  elif o == "-a":
    svars.ta = int(a)
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

  if svars.size_fft == None:
    raise RuntimeError("Must specify size for FFT matrix")

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
    qshell.prepare(svars.size_fft, svars.box_size)

  # Print basic properties shared across the files
  print("#nset: %d" %trajset.fileframes[-1])
  print("#N: %d" %trajset.fparticles)
  print("#timestep: %f" %trajset.timestep)
  print("#tbsave: %f" %trajset.tbsave)

  # Spatial size of individual cell for FFT
  cell = svars.box_size / svars.size_fft

  # End of set of frames to use for initial times
  if svars.initend == None:
    svars.initend = frames.final
  else:
    if svars.initend > frames.final:
      raise RuntimeError("End initial time frame beyond set of analyzed frames")

  # Largest and smallest possible average interval widths (t_b),
  # adjusting for both space taken up by t_a and t_c and intervals at
  # the beginning which may not be accessible
  prog.min_val = -svars.framediff * ((frames.n_frames - 1) // svars.framediff) + max(svars.tc, 0)
  prog.max_val = frames.n_frames - 1 - (svars.framediff * ((max(svars.tc - svars.ta, 0) + (svars.framediff - 1)) // svars.framediff)) - max(svars.ta + svars.tc, 0)

  # Construct progression of interval values using previously-specified
  # parameters
  tbvals = prog.construct()

  # Indices for types/components of S4 function
  totalcomp = 0
  selfcomp = 1
  distinctcomp = 2
  totalcompstd = 3
  selfcompstd = 4
  distinctcompstd = 5
  n_stypes = 6

  # Stores coordinates of all particles in a frame
  x0 = np.empty(frames.particles, dtype=np.single)
  y0 = np.empty(frames.particles, dtype=np.single)
  z0 = np.empty(frames.particles, dtype=np.single)
  x1 = np.empty(frames.particles, dtype=np.single)
  y1 = np.empty(frames.particles, dtype=np.single)
  z1 = np.empty(frames.particles, dtype=np.single)

  # Used when interval start times are different
  if svars.ta - svars.tc != 0:
    x2 = np.empty(frames.particles, dtype=np.single)
    y2 = np.empty(frames.particles, dtype=np.single)
    z2 = np.empty(frames.particles, dtype=np.single)

  # Bins for holding particle positions

  # Bins for first interval
  a_bins = np.empty((svars.size_fft, svars.size_fft, svars.size_fft), dtype=np.float64)

  # Only one set of bins is needed if intervals are equivalent
  if svars.ta != 0 or svars.tc != 0:
    b_bins = np.empty((svars.size_fft, svars.size_fft, svars.size_fft), dtype=np.float64)

  # If interval start times are the same, self bins are not needed, as
  # the exponential part is always 1
  if svars.ta - svars.tc != 0:
    self_bins = np.empty((svars.size_fft, svars.size_fft, svars.size_fft), dtype=np.float64)

  # Temporary value for each run to allow for calculation of each run's
  # self component of S4
  if svars.tc - svars.ta == 0:
    # If start frames of intervals are the same, then the self S4
    # values do not vary with q
    run_self_s4 = np.empty((1, 1, 1), dtype=np.float64)
  else:
    run_self_s4 = np.empty((svars.size_fft, svars.size_fft, (svars.size_fft // 2) + 1), dtype=np.float64)

  # Temporary value for each run to allow for calculation of each run's
  # total component of S4
  run_total_s4 = np.empty((svars.size_fft, svars.size_fft, (svars.size_fft // 2) + 1), dtype=np.float64)

  # Array for S4 values. The first and second fft dimensions include
  # values for negative vectors. Since all inputs are real, this is not
  # required for the third fft dimension, as the values would be the
  # same for a vector in the exact opposite direction (with all vector
  # components of opposite sign).
  s4 = np.empty((n_stypes, svars.size_fft, svars.size_fft, (svars.size_fft // 2) + 1), dtype=np.float64)

  # W function values for each particle
  if svars.ta == 0 and svars.tc == 0:
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

  print("#dt = %d" %svars.framediff)
  print("#n_tbvals = %d" %tbvals.size)
  print("#t_a = %d" %svars.ta)
  print("#t_c = %d" %svars.tc)
  print("#size_fft: %f" %svars.size_fft)

  # Print information about w function calculation
  wcalc.print_info()

  # Create legend with description of output columns
  legend = "#\n" \
           + "#Output Columns:\n" \
           + "#  1 - t_b\n"
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
            + "#  %d - Frame difference corresponding to t_b\n" %(col_offset1 + 8) \
            + "#\n"

  # S4 calcuation

  print("Entering S4 calculation", file=sys.stderr)

  # If output files not used, write to stdout
  if svars.dumpfiles == False:
    outfile = sys.stdout
    outfile.write(legend)

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
      for j in np.arange(0, svars.initend - frames.start, svars.framediff):
        # Use only indices that are within range
        if (svars.ta < (svars.tc - j) or
            svars.ta - frames.n_frames >= (svars.tc - j) or
            svars.ta < (-tb - j) or
            svars.ta - frames.n_frames >= (-tb - j) or
            j < (svars.tc - tb) or
            j - frames.n_frames >= (svars.tc - tb)):
          continue

        # Get starting frame for first interval and store in x0, y0, z0
        wcalc.get_w_half(j, x0, y0, z0, i)

        # If needed, get starting frame for second interval and store
        # in x2, y2, z2
        if svars.ta - svars.tc != 0:
          wcalc.get_w_half(j + svars.ta - svars.tc, x2, y2, z2, i)

        # Calculate w values for first interval
        wcalc.calculate_w_half(w[0], x0, y0, z0, j + tb - svars.tc, x1, y1, z1, i)

        # Calculate w values for second interval if needed
        if svars.ta != 0 or svars.tc != 0:
          if svars.ta - svars.tc != 0:
            wcalc.calculate_w_half(w[1], x2, y2, z2, j + svars.ta + tb, x1, y1, z1, i)
          else:
            wcalc.calculate_w_half(w[1], x0, y0, z0, j + svars.ta + tb, x1, y1, z1, i)

        # Convert particle positions into bin numbers and wrap for
        # binning
        x0i = (x0 // cell) % svars.size_fft
        y0i = (y0 // cell) % svars.size_fft
        z0i = (z0 // cell) % svars.size_fft

        # Sort first interval w values into bins for FFT
        a_bins = np.histogramdd((x0i, y0i, z0i), bins=svars.size_fft, range=((-0.5, svars.size_fft - 0.5), ) * 3, weights=w[0])[0]

        # Sort second interval w values into bins for FFT if needed
        if svars.ta != 0 or svars.tc != 0:
          if svars.ta - svars.tc == 0:
            b_bins = np.histogramdd((x0i, y0i, z0i), bins=svars.size_fft, range=((-0.5, svars.size_fft - 0.5), ) * 3, weights=w[1])[0]
          else:
            # Convert particle positions into bin numbers and wrap for
            # binning
            x2i = (x2 // cell) % svars.size_fft
            y2i = (y2 // cell) % svars.size_fft
            z2i = (z2 // cell) % svars.size_fft

            b_bins = np.histogramdd((x2i, y2i, z2i), bins=svars.size_fft, range=((-0.5, svars.size_fft - 0.5), ) * 3, weights=w[1])[0]

        # Calculate total part of S4
        if svars.ta != 0 or svars.tc != 0:
          run_total_s4 += fft.fftshift((fft.rfftn(a_bins) * np.conjugate(fft.rfftn(b_bins))).real, axes=(0, 1)) / frames.particles
        else:
          # Uses np.abs(), which calculates norm of complex numbers
          run_total_s4 += fft.fftshift(np.abs(fft.rfftn(a_bins))**2, axes=(0, 1)) / frames.particles

        # Multiply w values for different intervals together for self
        # bins
        if svars.ta != 0 or svars.tc != 0:
          w[0] *= w[1]
        else:
          # Squaring not required if w is same for each interval and is
          # boolean values, as 1*1 = 1 and 0*0 = 0
          if wcalc.wtype != lib.wcalc.wtypes.theta:
            w[0] *= w[0]

        # Calculate self part
        if svars.ta - svars.tc != 0:
          # Convert particle bin numbers into bin number differences
          # between starting frames of first and second intervals for
          # self part calculation and wrap for binning. Since total
          # exponential is negative, must use reverse difference with
          # positive-exponential FFT.
          x0i = (x0i - x2i) % svars.size_fft
          y0i = (y0i - y2i) % svars.size_fft
          z0i = (z0i - z2i) % svars.size_fft

          # Bin multiplied w values according to coordinate differences
          self_bins = np.histogramdd((x0i, y0i, z0i), bins=svars.size_fft, range=((-0.5, svars.size_fft - 0.5), ) * 3, weights=w[0])[0]

          # Perform FFT, thereby calculating self S4 for current index
          run_self_s4 += fft.fftshift(fft.rfftn(self_bins).real, axes=(0, 1)) / frames.particles
        else:
          run_self_s4[0][0][0] += np.sum(w[0]) / frames.particles

        # Accumulate the normalization value for this t_b value, which
        # will be used later in computing the mean S4 quantities for
        # each t_b
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

    # Print results for current t_b

    # t_b in real units
    time_tb = tb * trajset.timestep * trajset.tbsave

    # File to write data for time to
    if svars.dumpfiles == True:
      outfile = open("tb_%f" %(tbvals[index]), "w")
      outfile.write(legend)

    # If q vector shells used, sort by q vector magnitude into onion
    # shells and discrete magnitudes and print the averages of values
    # for each
    if qshell.active == True:
      discrete_s4, shell_s4 = qshell.to_shells(s4)

      # Print output columns for first region disctinct q magnitudes
      for i in range(0, discrete_s4.shape[-1]):
        outfile.write("%f %f %d %f %f %f %f %f %f %d %d\n"
                      %(time_tb,
                        qshell.qlist_discrete[i]*2*math.pi/svars.box_size,
                        qshell.qnorm_discrete[i],
                        discrete_s4[totalcomp][i],
                        discrete_s4[selfcomp][i],
                        discrete_s4[distinctcomp][i],
                        discrete_s4[totalcompstd][i],
                        discrete_s4[selfcompstd][i],
                        discrete_s4[distinctcompstd][i],
                        norm,
                        tb))

      # Print output columns for second region q magnitude onion shells
      for i in range(0, shell_s4.shape[-1]):
        outfile.write("%f %f %d %f %f %f %f %f %f %d %d\n"
                      %(time_tb,
                        (qshell.qb1a+(qshell.qlist_shells[i]+0.5)*qshell.swidth)*2*math.pi/svars.box_size,
                        qshell.qnorm_shells[i],
                        shell_s4[totalcomp][i],
                        shell_s4[selfcomp][i],
                        shell_s4[distinctcomp][i],
                        shell_s4[totalcompstd][i],
                        shell_s4[selfcompstd][i],
                        shell_s4[distinctcompstd][i],
                        norm,
                        tb))

    # If q vector shells not used, print output columns for all q
    # vectors
    else:
      for i in range(0, svars.size_fft):
        for j in range(0, svars.size_fft):
          for k in range(0, (svars.size_fft // 2) + 1):
            outfile.write("%f %f %f %f %f %f %f %f %f %f %d %d\n"
                          %(time_tb,
                            (i-svars.size_fft//2)*2*math.pi/svars.box_size,
                            (j-svars.size_fft//2)*2*math.pi/svars.box_size,
                            k*2*math.pi/svars.box_size,
                            s4[totalcomp][i][j][k],
                            s4[selfcomp][i][j][k],
                            s4[distinctcomp][i][j][k],
                            s4[totalcompstd][i][j][k],
                            s4[selfcompstd][i][j][k],
                            s4[distinctcompstd][i][j][k],
                            norm,
                            tb))

    # If output files for each t_b value used, close file for this t_b
    # value
    if svars.dumpfiles == True:
      outfile.close()

# Run full program
lib.script.run(main_func,
               argtext=argtext,
               svars=svars,
               shortopts=shortopts,
               catch_opt=catch_opt,
               modules=[prog, runset, trajset, frames, wcalc, qshell])
