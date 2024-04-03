import numpy as np
import pydcd
import sys
import math

# Import functionality from local library directory
import lib.script
import lib.opentraj
import lib.progression
import lib.frame

# Script-specific variables altered by arguments
class svars:
  # Last frame number to use for initial times
  initend = None
  # Difference between frame pair starts
  framediff = 10
  # Overlap radius for theta function
  radius = 0.25
  # Scattering vector constant
  q = 7.25

# Progression specification/generation object for lags
prog = lib.progression.prog()
# Run set opening object
runset = lib.opentraj.runset()
# Trajectory set opening object
trajset = lib.opentraj.trajset(runset)
# Frame reading object
frames = lib.frame.frames(trajset)

# Script-specific short options
shortopts = "k:d:a:q:"

# Description of script-specific arguments
argtext = "-k Last frame number in range to use for initial times (index starts at 1)\n" \
          + "-d Number of frames between starts of pairs to average (dt)\n" \
          + "-a Overlap radius for theta function (default: 0.25)\n" \
          + "-q Scattering vector constant (default: 7.25)"

def catch_opt(o, a, svars):
  if o == "-k":
    svars.initend = int(a)
  elif o == "-d":
    svars.framediff = int(a)
  elif o == "-a":
    svars.radius = float(a)
  elif o == "-q":
    svars.q = float(a)
  else:
    return False

  return True

def main_func(svars, prog, runset, trajset, frames):
  if runset.n_runs <= 1:
    raise RuntimeError("Must have at least 2 runs")

  # Open trajectory files
  trajset.opentraj()

  # Prepare frames object for calculation
  frames.prepare()

  # Print basic properties of files and analysis
  print("#nset: %d" %frames.fileframes[-1])
  print("#N: %d" %frames.fparticles)
  print("#timestep: %f" %trajset.timestep)
  print("#tbsave: %f" %trajset.tbsave)
  print("#dt = %f" %svars.framediff)
  print("#q = %f" %svars.q)
  print("#a = %f" %svars.radius)

  # End of set of frames to use for initial times
  if svars.initend == None:
    svars.initend = frames.final
  else:
    if svars.initend > frames.final:
      raise RuntimeError("End initial time frame beyond set of analyzed frames")

  # Largest possible positive and negative lags
  prog.max_val = frames.n_frames - 1
  prog.min_val = -frames.n_frames + 1

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

  # Result of scattering function variance for each difference in times
  var_fc = np.zeros((lags.size, 4), dtype=np.float64)

  # Accumulated overlap variance value for each difference in times
  var_overlap = np.zeros(lags.size, dtype=np.float64)

  # Normalization factor for scattering indices
  norm = np.zeros(lags.size, dtype=np.int64)

  # Accumulates values of scattering function across runs
  fc_accum = np.empty(4, dtype=np.float64)

  # Accumulates squared values of scattering function across runs
  fc2_accum = np.empty(4, dtype=np.float64)

  # Hold scattering functions for single run
  run_fc = np.empty(4, dtype=np.float64)

  # Iterate over starting points for functions
  for i in np.arange(0, frames.n_frames, svars.framediff):
    # Iterate over ending points for functions and add to accumulated
    # values, making sure to only use indices which are within the
    # range of the files.
    for index, j in enumerate(lags):
      if j >= (frames.n_frames - i) or j < -i:
        continue

      # Clear accumulator values
      fc_accum[:] = 0.0
      overlap_accum = 0.0
      fc2_accum[:] = 0.0
      overlap2_accum = 0.0

      for k in range(0, runset.n_runs):
        # Get interval start frame
        frames.get_frame(i, x0, y0, z0, k)

        # Get interval end frame
        frames.get_frame(i + j, x1, y1, z1, k)

        # Get means of scattering functions of all the particles for
        # each coordinate
        run_fc[0] = np.mean(np.cos(svars.q * (x1 - x0)))
        run_fc[1] = np.mean(np.cos(svars.q * (y1 - y0)))
        run_fc[2] = np.mean(np.cos(svars.q * (z1 - z0)))
        run_fc[3] = np.mean(run_fc[0:3])

        fc_accum += run_fc
        fc2_accum += run_fc**2

        # Add overlap value to accumulated value
        run_overlap = np.mean(np.less((x1 - x0)**2 +
                                      (y1 - y0)**2 +
                                      (z1 - z0)**2, svars.radius**2).astype(np.int8, copy=False))

        overlap_accum += run_overlap
        overlap2_accum += run_overlap**2

      fc_accum /= runset.n_runs
      fc2_accum /= runset.n_runs
      overlap_accum /= runset.n_runs
      overlap2_accum /= runset.n_runs

      # Calculate variances for lag index
      var_fc[index] += frames.particles * (fc2_accum - fc_accum**2)
      var_overlap[index] += frames.particles * (overlap2_accum - overlap_accum**2)

      # Accumulate the normalization value for this lag, which we will
      # use later in computing the mean scattering value for each lag
      norm[index] += 1

    print("Processed frame %d" %(i + frames.start + 1), file=sys.stderr)

  # Normalize the accumulated scattering values, thereby obtaining
  # averages over each pair of frames
  var_fc /= norm.reshape((lags.size, 1))

  # Normalize the overlap, thereby obtaining an average over each pair
  # of frames
  var_overlap /= norm

  # Print description of output columns
  print("#",
        "#Output Columns:",
        "#  1 - Time difference constituting interval",
        "#  2 - Variance across runs of average overlap",
        "#  3 - Variance across runs of x scattering function",
        "#  4 - Variance across runs of y scattering function",
        "#  5 - Variance across runs of z scattering function",
        "#  6 - Variance across runs of directional average scattering function",
        "#  7 - Number of frame pairs with interval",
        "#  8 - Frame difference corresponding to interval time",
        "#",
        sep="\n")

  for i in range(0, lags.size):
    time = lags[i] * trajset.timestep * trajset.tbsave
    # Print output columns
    print("%f %f %f %f %f %f %d %d"
          %(time,
            var_overlap[i],
            var_fc[i][0],
            var_fc[i][1],
            var_fc[i][2],
            var_fc[i][3],
            norm[i],
            lags[i]))

# Run full program
lib.script.run(main_func,
               argtext=argtext,
               svars=svars,
               shortopts=shortopts,
               catch_opt=catch_opt,
               modules=[prog, runset, trajset, frames])
