import numpy as np
from numpy import fft
from scipy import sparse
from scipy.sparse import linalg
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
shortopts = "k:d:x:y:b:c:i"

# Description of script-specific arguments
argtext = "-k Last frame number in range to use for initial times (index starts at 1)\n" \
          + "-d Spacing between initial times as well as lag values (dt)\n" \
          + "-x Dimensionality of FFT matrix, length in each dimension\n" \
          + "-y Box size in each dimension (assumed to be cubic, required)\n" \
          + "-b Average interval in frames (t_b, default=1)\n" \
          + "-c Difference between intervals in frames (t_c, default=0)\n" \
          + "-i Write output to files, one for each lag time\n" \
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
  elif o == "-b":
    svars.tb = int(a)
  elif o == "-c":
    svars.tc = int(a)
  elif o == "-i":
    svars.dumpfiles = True
  else:
    return False

  return True

def construct_fill_matrices(b, threshold, kerndelta, kernvals):
  # Shape of divisor array
  bshape = b.shape

  if np.any(np.array(bshape) < 5):
    raise RuntimeError("FFT matrix not large enough for filling zero-denominator values")

  # Mask with set of X4 values less than first quantum of X4
  zeromask = b < threshold
  nonzeromask = np.logical_not(zeromask)
  zerovals = np.argwhere(zeromask)

  # Create matrix with indices of X4 unknown values in spatial
  # locations of unknown values
  zeronums = np.full(bshape, -1, dtype=np.int64)
  zeronums[zeromask] = np.arange(0, zerovals.shape[0])

  # Create new coefficient matrix
  mat = sparse.lil_matrix((zerovals.shape[0], zerovals.shape[0]), dtype=np.float64)

  # Self coefficient is always 5/6
  mat.setdiag(5/6)

  # Create new matrix for constructing constant vector. Values in FFT
  # matrix are flattened over last dimension to make vector dot product
  # work correctly
  constmat = sparse.lil_matrix((zerovals.shape[0], np.prod(bshape)), dtype=np.float64)

  if zerovals.shape[0] > 0:
    # Iterate over unknown values
    for i, origin in enumerate(zerovals):
      # Iterate over values in kernel
      for delta, val in zip(kerndelta, kernvals):
        # Coordinate of corresponding value
        coord = tuple((origin + delta) % bshape)

        # Number of unknown value if value at coord is unknown,
        # otherwise has value of -1
        num = zeronums[coord]

        if num == -1:
          # If a known value, add to constants. Calculate position in
          # C-order flattened array to insert
          constmat[i, coord[0] * bshape[1]*bshape[2] + coord[1] * bshape[2] + coord[2]] += val
        else:
          # If an unknown value and creating a new coefficient matrix,
          # add to coefficient
          mat[i, num] -= val

    # Convert coefficient matrix to CSR format for solution routine
    mat = mat.tocsr()

    # Convert constant-generation matrix to CSR format for efficient
    # vector product calculation
    constmat = constmat.tocsr()

  return nonzeromask, zerovals, mat, constmat

def div_fill(a, b, nonzeromask, zerovals, mat, constmat):
  # Computed output values
  f = np.empty(b.shape, dtype=np.float64)

  # Set of values that can be directly computed due to large-enough
  # divisors
  f[nonzeromask] = a[nonzeromask] / b[nonzeromask]

  if zerovals.shape[0] > 0:
    # Multiply by constant-generating matrix and then coefficient
    # matrix inverse to solve linear system. Assign solution to unknown
    # values in computed matrix.
    f[tuple(zerovals.transpose())] = sparse.linalg.spsolve(mat, constmat.dot(np.ravel(f, order="C")))

  return f

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
  print("#size_fft: %f" %svars.size_fft)

  # Spatial size of individual cell for FFT
  cell = svars.box_size / svars.size_fft

  # Average density of particles in box
  density = frames.particles / svars.size_fft**3

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
  lag_bins = np.zeros((2, svars.size_fft, svars.size_fft, svars.size_fft), dtype=np.float64)

  # Holds observables calculated from trajectories for each run that
  # are not associated with a spatial vector. In first dimension:
  # 0 - C(t_1,t_2) - First interval average w function (X1)
  # 1 - C(t_3,t_4) - Second interval average w function (X2)
  obsv = np.empty((2, runset.n_runs), dtype=np.float64)

  # Holds observables calculated from trajectories for each run that
  # are associated with a spatial vector. In first dimension:
  # 0 - G(r,t_1,t_3) - Fourier transformed particle positions between
  #     starts of first and second intervals, total part (X3)
  # 1 - G_d(r,t_1,t_3) - Fourier transformed particle positions between
  #     starts of first and second intervals, distinct part (X4)
  # 2 - G_s(r,t_1,t_3) - Fourier transformed particle positions between
  #     starts of first and second intervals, self part (X5)
  # 3 - G4_d(r,t_1,t_2,t_3,t_4) - G4 distinct part (X6)
  # 4 - G4_s(r,t_1,t_2,t_3,t_4) - G4 self part (X7)
  obsv_s = np.empty((5, runset.n_runs, svars.size_fft, svars.size_fft, svars.size_fft), dtype=np.float64)

  # Cross-run and jackknife means of observables and combined
  # observables not associated with a spatial vector. In first
  # dimension:
  # 0 - C(t_1,t_2) (X1)
  # 1 - C(t_3,t_4) (X2)
  # 2 - C(t_1,t_2) * C(t_3,t_4) (X1 * X2)
  mobsv = np.empty(3, dtype=np.float64)
  jmobsv = np.empty(3, dtype=np.float64)

  # Cross-run and jackknife means of observables and combined
  # observables that are associated with a spatial vector. In first
  # dimension:
  # 0 - G(r,t_1,t_3) (X3)
  # 1 - G_d(r,t_1,t_3) (X4)
  # 2 - G_s(r,t_1,t_3) (X5)
  # 3 - G4_d(r,t_1,t_2,t_3,t_4) (X6)
  # 4 - G4_s(r,t_1,t_2,t_3,t_4) (X7)
  # 5 - C(t_1,t_2) * G(r,t_1,t_3) (X1 * X3)
  # 6 - C(t_3,t_4) * G(r,t_1,t_3) (X2 * X3)
  # 7 - G_d(r,t_1,t_3)^2 (X3^2)
  # 8 - G_d(r,t_1,t_3) * G_s(r,t_1,t_3) (X4 * X5)
  # 9 - G_d(r,t_1,t_3) * G4_d(r,t_1,t_2,t_3,t_4) (X4 * X6)
  # 10 - G_s(r,t_1,t_3) * G4_d(r,t_1,t_2,t_3,t_4) (X5 * X6)
  mobsv_s = np.empty((11, svars.size_fft, svars.size_fft, svars.size_fft), dtype=np.float64)
  jmobsv_s = np.empty((11, svars.size_fft, svars.size_fft, svars.size_fft), dtype=np.float64)

  # Holds 4 components of S4 calcuated with unbiased estimators. In
  # first dimension:
  # 0 - S4^st - S4 initial structure contribution
  # 1 - S4^cr - S4 collective relaxation contribution
  # 2 - S4^mc - S4 mixed collective contribution
  # 3 - S4^sp - S4 single particle oscillation contribution
  # 4 - S4^st - S4 initial structure contribution standard deviation
  # 5 - S4^cr - S4 collective relaxation contribution standard
  #             deviation
  # 6 - S4^mc - S4 mixed collective contribution standard deviation
  # 7 - S4^sp - S4 single particle oscillation contribution standard
  #             deviation
  # 8 - S4^st - S4 initial structure jackknife contribution
  # 9 - S4^cr - S4 collective relaxation jackknife contribution
  # 10 - S4^mc - S4 mixed collective jackknife contribution
  # 11 - S4^sp - S4 single particle oscillation jackknife contribution
  if qshell.active == True:
    s4_discrete = np.empty((12, len(qshell.qlist_discrete)), dtype=np.float64)
    s4_shells = np.empty((12, len(qshell.qlist_shells)), dtype=np.float64)
  else:
    s4_comp = np.empty((12, svars.size_fft, svars.size_fft, (svars.size_fft // 2) + 1), dtype=np.float64)

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

  # S4 components calcuation

  # Create kernel for matrix of coefficients in division filling
  # calculation
  kernel = np.zeros((5, 5, 5), dtype=np.float64)

  # Used for constructing kernel. This structure will be repeated 6
  # times in different orientations in the final kernel
  subkernel = np.array((((-1/18, 1/3),   (0.0, -1/18)),
                        ((0.0,   -1/36), (0.0, 0.0))), dtype=np.float64)

  # Iterate over dimensions to orient subkernel in
  for dim in range(0, 3):
    # Add values for positive orientation of subkernel along axis
    kernel[tuple(np.roll(np.mgrid[3:5,2:4,1:3], dim, axis=0))] = subkernel

    # Add values for negative orientation of subkernel along axis
    kernel[tuple(np.roll(np.mgrid[0:2,1:3,2:4], dim, axis=0))] = np.flip(subkernel)

  # Store kernel coordinate differences from center and compress kernel
  # to eliminate 0 values
  kerndelta = np.mgrid[-2:3,-2:3,-2:3].transpose()[kernel != 0.0, :]
  kernvals = kernel[kernel != 0.0]

  # Create legend with description of output columns
  legend = "#\n" \
           + "#Output columns:\n" \
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
  legend += "#  %d - Run average of S4^st (S4 initial structure contribution)\n" %(col_offset1 + 1) \
            + "#  %d - Run average of S4^cr (S4 collective relaxation contribution)\n" %(col_offset1 + 2) \
            + "#  %d - Run average of S4^mc (S4 mixed collective contribution)\n" %(col_offset1 + 3) \
            + "#  %d - Run average of S4^sp (S4 single particle oscillation contribution)\n" %(col_offset1 + 4) \
            + "#  %d - Standard deviation across runs of S4^st\n" %(col_offset1 + 5) \
            + "#  %d - Standard deviation across runs of S4^cr\n" %(col_offset1 + 6) \
            + "#  %d - Standard deviation across runs of S4^mc\n" %(col_offset1 + 7) \
            + "#  %d - Standard deviation across runs of S4^sp\n" %(col_offset1 + 8) \
            + "#  %d - Number of frame sets in each run contributing to average of quantities\n" %(col_offset1 + 9) \
            + "#  %d - Frame difference corresponding to t_a\n" %(col_offset1 + 10) \
            + "#\n"

  print("Entering S4 components calculation", file=sys.stderr)

  # If output files not used, write to stdout
  if svars.dumpfiles == False:
    outfile = sys.stdout
    outfile.write(legend)

  # Accumulated total number of zero elements of divisor matrices
  zero_acc = 0

  # Iterate over lags (t_a)
  for index, ta in enumerate(lags):
    # Clear lag accumulators
    obsv[:, :] = 0.0
    obsv_s[:, :, :, :, :] = 0.0
    if qshell.active == True:
      s4_discrete[:, :] = 0.0
      s4_shells[:, :] = 0.0
    else:
      s4_comp[:, :, :, :] = 0.0

    # Normalization factor for number of frame pairs contributing to
    # current lag value
    norm = 0

    # Iterate over runs
    for i in range(0, runset.n_runs):
      # Iterate over t_1 values (initial times) for observables
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

        # Convert particle positions into bin numbers and wrap for
        # binning
        x0i = (x0 // cell) % svars.size_fft
        y0i = (y0 // cell) % svars.size_fft
        z0i = (z0 // cell) % svars.size_fft
        x2i = (x2 // cell) % svars.size_fft
        y2i = (y2 // cell) % svars.size_fft
        z2i = (z2 // cell) % svars.size_fft

        # Accumulate mean w function values
        obsv[0][i] += np.mean(w[0])
        obsv[1][i] += np.mean(w[1])

        # Bin particle positions, correlate, and accumulate
        lag_bins[0] = np.histogramdd((x0i, y0i, z0i), bins=svars.size_fft, range=((-0.5, svars.size_fft - 0.5), ) * 3)[0]
        lag_bins[1] = np.histogramdd((x2i, y2i, z2i), bins=svars.size_fft, range=((-0.5, svars.size_fft - 0.5), ) * 3)[0]
        lag_bins[0] = fft.irfftn(fft.rfftn(lag_bins[0]) * np.conjugate(fft.rfftn(lag_bins[1])), s=lag_bins.shape[1:])
        obsv_s[0][i] += lag_bins[0]

        # Bin weighted particle positions for G4, correlate, and
        # accumulate. While the accumulated value is the total part, it
        # will be converted to the distinct part later.
        lag_bins[0] = np.histogramdd((x0i, y0i, z0i), bins=svars.size_fft, range=((-0.5, svars.size_fft - 0.5), ) * 3, weights=w[0])[0]
        lag_bins[1] = np.histogramdd((x2i, y2i, z2i), bins=svars.size_fft, range=((-0.5, svars.size_fft - 0.5), ) * 3, weights=w[1])[0]
        lag_bins[0] = fft.irfftn(fft.rfftn(lag_bins[0]) * np.conjugate(fft.rfftn(lag_bins[1])), s=lag_bins.shape[1:])
        obsv_s[3][i] += lag_bins[0]

        # Convert particle bin numbers into bin number differences
        # between starting frames of first and second intervals for
        # self part calculation and wrap for binning. Since total
        # exponential is negative, must use reverse difference with
        # positive-exponential FFT.
        x0i = (x0i - x2i) % svars.size_fft
        y0i = (y0i - y2i) % svars.size_fft
        z0i = (z0i - z2i) % svars.size_fft

        # Multiply w values for different intervals together for self
        # bins
        w[0] *= w[1]

        # Bin self particle position differences and accumulate
        lag_bins[0] = np.histogramdd((x0i, y0i, z0i), bins=svars.size_fft, range=((-0.5, svars.size_fft - 0.5), ) * 3)[0]
        obsv_s[2][i] += lag_bins[0]

        # Bin weighted self particle position differences for G4 and
        # accumulate
        lag_bins[0] = np.histogramdd((x0i, y0i, z0i), bins=svars.size_fft, range=((-0.5, svars.size_fft - 0.5), ) * 3, weights=w[0])[0]
        obsv_s[4][i] += lag_bins[0]

        # Accumulate the normalization value for this lag, which will
        # be used later in computing the mean S4 quantities for each
        # lag
        if i == 0:
          norm += 1

    # Compute distinct part of G from self and total parts
    obsv_s[1] = obsv_s[0] - obsv_s[2]

    # Compute distinct part of G4 from self and total parts
    obsv_s[3] -= obsv_s[4]

    # Normalize values by number of initial times and particles
    obsv /= norm
    obsv_s /= (norm * frames.particles)

    # Compute means of observables for full means
    mobsv[0] = np.mean(obsv[0])
    mobsv[1] = np.mean(obsv[1])
    mobsv[2] = np.mean(obsv[0] * obsv[1])
    mobsv_s[0] = np.mean(obsv_s[0], axis=0)
    mobsv_s[1] = np.mean(obsv_s[1], axis=0)
    mobsv_s[2] = np.mean(obsv_s[2], axis=0)
    mobsv_s[3] = np.mean(obsv_s[3], axis=0)
    mobsv_s[4] = np.mean(obsv_s[4], axis=0)
    mobsv_s[5] = np.mean(obsv[0,:,None,None,None] * obsv_s[0], axis=0)
    mobsv_s[6] = np.mean(obsv[1,:,None,None,None] * obsv_s[0], axis=0)
    mobsv_s[7] = np.mean(obsv_s[1]**2, axis=0)
    mobsv_s[8] = np.mean(obsv_s[1] * obsv_s[2], axis=0)
    mobsv_s[9] = np.mean(obsv_s[1] * obsv_s[3], axis=0)
    mobsv_s[10] = np.mean(obsv_s[2] * obsv_s[3], axis=0)

    # Run division factor for full means
    fn = 1 / (runset.n_runs - 1)

    # Construct matrices of linear system for filling unknown values
    # for full mean
    nonzeromask, zerovals, mat, constmat = construct_fill_matrices(mobsv_s[1], 0.5 / (norm * frames.particles * runset.n_runs), kerndelta, kernvals)

    # Accumulate to number of zero values
    zero_acc += zerovals.shape[0]

    # Compute parts of estimators for full means
    est_r12 = density * ((1 + fn) * mobsv[0] * mobsv[1] - fn * mobsv[2])
    est_123 = (1 + 3*fn) * mobsv[0] * mobsv[1] * mobsv_s[0] \
              - fn * (mobsv_s[0] * mobsv[2]
                      + mobsv[0] * mobsv_s[6]
                      + mobsv[1] * mobsv_s[5])
    if ta - svars.tc == 0:
      # Move G_s out of zero-filling routine numerator for cases when
      # t1 == t3
      est_r6d4 = div_fill(mobsv_s[3], mobsv_s[1], nonzeromask, zerovals, mat, constmat) \
                          - fn * (div_fill(mobsv_s[7] * mobsv_s[3], mobsv_s[1]**3, nonzeromask, zerovals, mat, constmat)
                                  - div_fill(mobsv_s[9], mobsv_s[1]**2, nonzeromask, zerovals, mat, constmat))
      est_65d4 = mobsv_s[2] * est_r6d4
      est_r6d4 *= density
    else:
      est_r6d4 = density * (div_fill(mobsv_s[3], mobsv_s[1], nonzeromask, zerovals, mat, constmat)
                            - fn * (div_fill(mobsv_s[7] * mobsv_s[3], mobsv_s[1]**3, nonzeromask, zerovals, mat, constmat)
                                    - div_fill(mobsv_s[9], mobsv_s[1]**2, nonzeromask, zerovals, mat, constmat)))
      est_65d4 = div_fill(mobsv_s[2] * mobsv_s[3], mobsv_s[1], nonzeromask, zerovals, mat, constmat) \
                 - fn * (div_fill(mobsv_s[7] * mobsv_s[2] * mobsv_s[3], mobsv_s[1]**3, nonzeromask, zerovals, mat, constmat)
                         + div_fill(mobsv_s[10], mobsv_s[1], nonzeromask, zerovals, mat, constmat)
                         - div_fill(mobsv_s[9] * mobsv_s[2], mobsv_s[1]**2, nonzeromask, zerovals, mat, constmat)
                         - div_fill(mobsv_s[8] * mobsv_s[3], mobsv_s[1]**2, nonzeromask, zerovals, mat, constmat))

    # Calculate S4 contribution full means
    if qshell.active == True:
      s4_discrete[0], s4_shells[0] = qshell.to_shells(fft.fftshift(fft.rfftn(est_123 - est_r12).real, axes=(0, 1)))
      s4_discrete[1], s4_shells[1] = qshell.to_shells(fft.fftshift(fft.rfftn(est_r6d4 - est_r12).real, axes=(0, 1)))
      s4_discrete[2], s4_shells[2] = qshell.to_shells(fft.fftshift(fft.rfftn(mobsv_s[3] + est_65d4 - est_r6d4 - est_123 + est_r12).real, axes=(0, 1)))
      s4_discrete[3], s4_shells[3] = qshell.to_shells(fft.fftshift(fft.rfftn(mobsv_s[4] - est_65d4).real, axes=(0, 1)))
    else:
      s4_comp[0] = fft.fftshift(fft.rfftn(est_123 - est_r12).real, axes=(0, 1))
      s4_comp[1] = fft.fftshift(fft.rfftn(est_r6d4 - est_r12).real, axes=(0, 1))
      s4_comp[2] = fft.fftshift(fft.rfftn(mobsv_s[3] + est_65d4 - est_r6d4 - est_123 + est_r12).real, axes=(0, 1))
      s4_comp[3] = fft.fftshift(fft.rfftn(mobsv_s[4] - est_65d4).real, axes=(0, 1))

    # Run division factor for jackknife estimators
    fn = 1 / (runset.n_runs - 2)

    # Adjust full means to aid calculating jackknife means
    mobsv *= runset.n_runs / (runset.n_runs - 1)
    mobsv_s *= runset.n_runs / (runset.n_runs - 1)

    # Compute jackknife estimators of G4 components
    for i in range(0, runset.n_runs):
      # Compute means of observables for jackknife mean
      jmobsv[0] = mobsv[0] - obsv[0][i] / (runset.n_runs - 1)
      jmobsv[1] = mobsv[1] - obsv[1][i] / (runset.n_runs - 1)
      jmobsv[2] = mobsv[2] - obsv[0][i] * obsv[1][i] / (runset.n_runs - 1)
      jmobsv_s[0] = mobsv_s[0] - obsv_s[0][i] / (runset.n_runs - 1)
      jmobsv_s[1] = mobsv_s[1] - obsv_s[1][i] / (runset.n_runs - 1)
      jmobsv_s[2] = mobsv_s[2] - obsv_s[2][i] / (runset.n_runs - 1)
      jmobsv_s[3] = mobsv_s[3] - obsv_s[3][i] / (runset.n_runs - 1)
      jmobsv_s[4] = mobsv_s[4] - obsv_s[4][i] / (runset.n_runs - 1)
      jmobsv_s[5] = mobsv_s[5] - obsv[0][i] * obsv_s[0][i] / (runset.n_runs - 1)
      jmobsv_s[6] = mobsv_s[6] - obsv[1][i] * obsv_s[0][i] / (runset.n_runs - 1)
      jmobsv_s[7] = mobsv_s[7] - obsv_s[1][i]**2 / (runset.n_runs - 1)
      jmobsv_s[8] = mobsv_s[8] - obsv_s[1][i] * obsv_s[2][i] / (runset.n_runs - 1)
      jmobsv_s[9] = mobsv_s[9] - obsv_s[1][i] * obsv_s[3][i] / (runset.n_runs - 1)
      jmobsv_s[10] = mobsv_s[10] - obsv_s[2][i] * obsv_s[3][i] / (runset.n_runs - 1)

      # Construct matrices of linear system for filling unknown values
      # for full mean
      nonzeromask, zerovals, mat, constmat = construct_fill_matrices(jmobsv_s[1], 0.5 / (norm * frames.particles * (runset.n_runs - 1)), kerndelta, kernvals)

      # Accumulate to number of zero values
      zero_acc += zerovals.shape[0]

      # Compute parts of estimators for jackknife mean
      est_r12 = density * ((1 + fn) * jmobsv[0] * jmobsv[1] - fn * jmobsv[2])
      est_123 = (1 + 3*fn) * jmobsv[0] * jmobsv[1] * jmobsv_s[0] \
                - fn * (jmobsv_s[0] * jmobsv[2]
                        + jmobsv[0] * jmobsv_s[6]
                        + jmobsv[1] * jmobsv_s[5])
      if ta - svars.tc == 0:
        # Move G_s out of zero-filling routine numerator for cases when
        # t1 == t3
        est_r6d4 = div_fill(jmobsv_s[3], jmobsv_s[1], nonzeromask, zerovals, mat, constmat) \
                            - fn * (div_fill(jmobsv_s[7] * jmobsv_s[3], jmobsv_s[1]**3, nonzeromask, zerovals, mat, constmat)
                                    - div_fill(jmobsv_s[9], jmobsv_s[1]**2, nonzeromask, zerovals, mat, constmat))
        est_65d4 = jmobsv_s[2] * est_r6d4
        est_r6d4 *= density
      else:
        est_r6d4 = density * (div_fill(jmobsv_s[3], jmobsv_s[1], nonzeromask, zerovals, mat, constmat)
                              - fn * (div_fill(jmobsv_s[7] * jmobsv_s[3], jmobsv_s[1]**3, nonzeromask, zerovals, mat, constmat)
                                      - div_fill(jmobsv_s[9], jmobsv_s[1]**2, nonzeromask, zerovals, mat, constmat)))
        est_65d4 = div_fill(jmobsv_s[2] * jmobsv_s[3], jmobsv_s[1], nonzeromask, zerovals, mat, constmat) \
                   - fn * (div_fill(jmobsv_s[7] * jmobsv_s[2] * jmobsv_s[3], jmobsv_s[1]**3, nonzeromask, zerovals, mat, constmat)
                           + div_fill(jmobsv_s[10], jmobsv_s[1], nonzeromask, zerovals, mat, constmat)
                           - div_fill(jmobsv_s[9] * jmobsv_s[2], jmobsv_s[1]**2, nonzeromask, zerovals, mat, constmat)
                           - div_fill(jmobsv_s[8] * jmobsv_s[3], jmobsv_s[1]**2, nonzeromask, zerovals, mat, constmat))

      # Calculate jackknife contributions
      if qshell.active == True:
        run_st_discrete, run_st_shells = qshell.to_shells(fft.fftshift(fft.rfftn(est_123 - est_r12).real, axes=(0, 1)))
        run_cr_discrete, run_cr_shells = qshell.to_shells(fft.fftshift(fft.rfftn(est_r6d4 - est_r12).real, axes=(0, 1)))
        run_mc_discrete, run_mc_shells = qshell.to_shells(fft.fftshift(fft.rfftn(jmobsv_s[3] + est_65d4 - est_r6d4 - est_123 + est_r12).real, axes=(0, 1)))
        run_sp_discrete, run_sp_shells = qshell.to_shells(fft.fftshift(fft.rfftn(jmobsv_s[4] - est_65d4).real, axes=(0, 1)))
      else:
        run_st = fft.fftshift(fft.rfftn(est_123 - est_r12).real, axes=(0, 1))
        run_cr = fft.fftshift(fft.rfftn(est_r6d4 - est_r12).real, axes=(0, 1))
        run_mc = fft.fftshift(fft.rfftn(jmobsv_s[3] + est_65d4 - est_r6d4 - est_123 + est_r12).real, axes=(0, 1))
        run_sp = fft.fftshift(fft.rfftn(jmobsv_s[4] - est_65d4).real, axes=(0, 1))

      # Accumulate jackknife contributions for jackknife mean
      if qshell.active == True:
        s4_discrete[8] += run_st_discrete
        s4_shells[8] += run_st_shells
        s4_discrete[9] += run_cr_discrete
        s4_shells[9] += run_cr_shells
        s4_discrete[10] += run_mc_discrete
        s4_shells[10] += run_mc_shells
        s4_discrete[11] += run_sp_discrete
        s4_shells[11] += run_sp_shells
      else:
        s4_comp[8] += run_st
        s4_comp[9] += run_cr
        s4_comp[10] += run_mc
        s4_comp[11] += run_sp

      # Accumulate squared jackknife contributions for standard
      # deviation
      if qshell.active == True:
        s4_discrete[4] += run_st_discrete**2
        s4_shells[4] += run_st_shells**2
        s4_discrete[5] += run_cr_discrete**2
        s4_shells[5] += run_cr_shells**2
        s4_discrete[6] += run_mc_discrete**2
        s4_shells[6] += run_mc_shells**2
        s4_discrete[7] += run_sp_discrete**2
        s4_shells[7] += run_sp_shells**2
      else:
        s4_comp[4] += run_st**2
        s4_comp[5] += run_cr**2
        s4_comp[6] += run_mc**2
        s4_comp[7] += run_sp**2

    # Normalize accumulated jackknife means by number of jackknife
    # means
    if qshell.active == True:
      s4_discrete[4:12] /= runset.n_runs
      s4_shells[4:12] /= runset.n_runs
    else:
      s4_comp[4:12] /= runset.n_runs

    # Compute estimated mean from full and jackknife means
    if qshell.active == True:
      s4_discrete[0:4] = runset.n_runs * s4_discrete[0:4] - (runset.n_runs - 1) * s4_discrete[8:12]
      s4_shells[0:4] = runset.n_runs * s4_shells[0:4] - (runset.n_runs - 1) * s4_shells[8:12]
    else:
      s4_comp[0:4] = runset.n_runs * s4_comp[0:4] - (runset.n_runs - 1) * s4_comp[8:12]

    # Compute standard deviation from jackknife variances and means
    if qshell.active == True:
      s4_discrete[4:8] = np.sqrt(((runset.n_runs - 1)/runset.n_runs) * np.maximum(0.0, s4_discrete[4:8] - s4_discrete[8:12]**2))
      s4_shells[4:8] = np.sqrt(((runset.n_runs - 1)/runset.n_runs) * np.maximum(0.0, s4_shells[4:8] - s4_shells[8:12]**2))
    else:
      s4_comp[4:8] = np.sqrt(((runset.n_runs - 1)/runset.n_runs) * np.maximum(0.0, s4_comp[4:8] - s4_comp[8:12]**2))

    # Print results for current lag

    # Lag time in real units
    time_ta = ta * trajset.timestep * trajset.tbsave

    # If output files used, open file for current lag
    if svars.dumpfiles == True:
      outfile = open("lag_%f" %(lags[index]), "w")
      outfile.write(legend)

    # If q vector shells, used print according to discrete and shell q
    # magnitudes
    if qshell.active == True:
      # Print output columns for first region distinct q magnitudes
      for i in range(0, s4_discrete.shape[-1]):
        outfile.write("%f %f %d %f %f %f %f %f %f %f %f %d %d\n"
                      %(time_ta,
                        qshell.qlist_discrete[i]*2*math.pi/svars.box_size,
                        qshell.qnorm_discrete[i],
                        s4_discrete[0][i],
                        s4_discrete[1][i],
                        s4_discrete[2][i],
                        s4_discrete[3][i],
                        s4_discrete[4][i],
                        s4_discrete[5][i],
                        s4_discrete[6][i],
                        s4_discrete[7][i],
                        norm,
                        ta))

      # Print output columns for second region q magnitude onion shells
      for i in range(0, s4_shells.shape[-1]):
        outfile.write("%f %f %d %f %f %f %f %f %f %f %f %d %d\n"
                      %(time_ta,
                        (qshell.qb1a+(qshell.qlist_shells[i]+0.5)*qshell.swidth)*2*math.pi/svars.box_size,
                        qshell.qnorm_shells[i],
                        s4_shells[0][i],
                        s4_shells[1][i],
                        s4_shells[2][i],
                        s4_shells[3][i],
                        s4_shells[4][i],
                        s4_shells[5][i],
                        s4_shells[6][i],
                        s4_shells[7][i],
                        norm,
                        ta))

    # If q vector shells not used, print output columns for all q
    # vectors
    else:
      for i in range(0, svars.size_fft):
        for j in range(0, svars.size_fft):
          for k in range(0, (svars.size_fft // 2) + 1):
            outfile.write("%f %f %f %f %f %f %f %f %f %f %f %f %d %d\n"
                          %(time_ta,
                            (i-svars.size_fft//2)*2*math.pi/svars.box_size,
                            (j-svars.size_fft//2)*2*math.pi/svars.box_size,
                            k*2*math.pi/svars.box_size,
                            s4_comp[0][i][j][k],
                            s4_comp[1][i][j][k],
                            s4_comp[2][i][j][k],
                            s4_comp[3][i][j][k],
                            s4_comp[4][i][j][k],
                            s4_comp[5][i][j][k],
                            s4_comp[6][i][j][k],
                            s4_comp[7][i][j][k],
                            norm,
                            ta))

    # If output files for each lag used, close file for this lag
    if svars.dumpfiles == True:
      outfile.close()

  # Find average number of zero elements in matrices
  zero_acc /= lags.size * (runset.n_runs + 1)

  print("#Performance statistics:",
        "#  FFT size: %d" %svars.size_fft,
        "#  Average matrix zeros: %f" %zero_acc,
        "#  Particle number: %d" %frames.particles,
        "#  Number of lags: %d" %lags.size,
        sep="\n", file=sys.stderr)

# Run full program
lib.script.run(main_func,
               argtext=argtext,
               svars=svars,
               shortopts=shortopts,
               catch_opt=catch_opt,
               modules=[prog, runset, trajset, frames, wcalc, qshell])
