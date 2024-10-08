import numpy as np
import pydcd
import sys

# Import functionality from local library directory
import lib.script
import lib.opentraj
import lib.frame

# Script-specific variables altered by arguments
class svars:
  # Start trajectory file index in filenames for second region
  m_start = 1
  # List of initial frame indices
  initial = None
  # Overlap radius for theta function
  radius = 0.25
  # Scattering vector constant
  q = 7.25
  # Threshold for overlap between orientations
  thetab = None
  # Number of Legendre polynomial degrees to use in orientation
  # calculations
  legendre = None

# Script-specific short options
shortopts = "z:i:a:q:"

# Script-specific long options
longopts = ["polyatomic-thetab=", "polyatomic-legendre="]

# Description of script-specific arguments
argtext = "-z short(m).dcd file index to start on (default: 1)\n" \
          + "-i Initial frame numbers specified as list separated by commas (start at 1, offset by start specified with -s)\n" \
          + "-a Overlap radius for theta function (default: 0.25)\n" \
          + "-q Scattering vector constant (default: 7.25)\n" \
          + "--polyatomic-legendre Largest degree of Legendre polynomial to calculate orientational correlation functions with (default: 1)\n" \
          + "--polyatomic-thetab Threshold for theta function of dot product of molecule orientations (default: 0.9)"

# Run set opening object
runset = lib.opentraj.runset()
# First (aging) region trajectory set opening object
trajset1 = lib.opentraj.trajset(runset, opt="n", name="traj", same_tbsave=False)
# Second (geometric) region trajectory set opening object. Default
# number of short trajectories is 0.
trajset2 = lib.opentraj.trajset(runset, opt="t", name="short", default_n_files=0, same_tbsave=False)
# First (aging) region frame reading object
frames = lib.frame.frames([trajset1, trajset2])

def catch_opt(o, a, svars):
  if o == "-z":
    svars.m_start = int(a)
  elif o == "-i":
    svars.initial = np.array(list(map(int, a.split(",")))) - 1
  elif o == "-a":
    svars.radius = float(a)
  elif o == "-q":
    svars.q = float(a)
  elif o == "--polyatomic-thetab":
    svars.thetab = float(a)
  elif o == "--polyatomic-legendre":
    svars.legendre = int(a)
  else:
    return False

  return True

def main_func(svars, runset, trajset1, trajset2, frames):
  if svars.initial is None:
    raise RuntimeError("Must specify set of initial frame indices")

  # Adjust start of trajectory file numbering to match specified index
  trajset2.n_start = svars.m_start

  # Open trajectory files for both trajectory types
  trajset1.opentraj()
  trajset2.opentraj()

  # Prepare frames object for calculation
  frames.prepare()

  if np.any(svars.initial >= frames.n_frames):
    raise RuntimeError("At least one initial frame number is beyond range of frames")

  if frames.n_atoms == None and (svars.legendre != None or svars.thetab != None):
    raise RuntimeError("Legendre polynomials degree or theta function threshold of orientation correlations specified without being in polyatomic mode")

  if frames.n_atoms != None:
    if svars.legendre == None:
      svars.legendre = 1

    if svars.thetab == None:
      svars.thetab = 0.9

  # Print basic properties of files and analysis
  print("#nset: %d" %frames.fileframes[-1])
  print("#N: %d" %frames.fparticles)
  print("#q = %f" %svars.q)
  print("#a = %f" %svars.radius)

  # Stores coordinates of all particles in initial frames
  x0 = np.empty((runset.n_runs, frames.particles), dtype=np.single)
  y0 = np.empty((runset.n_runs, frames.particles), dtype=np.single)
  z0 = np.empty((runset.n_runs, frames.particles), dtype=np.single)

  # Stores coordinates of all particles in end frames
  x1 = np.empty(frames.particles, dtype=np.single)
  y1 = np.empty(frames.particles, dtype=np.single)
  z1 = np.empty(frames.particles, dtype=np.single)

  if frames.n_atoms != None:
    # Stores unit orientation vectors of all particles in a frame
    xo0 = [np.empty(frames.particles, dtype=np.single)] * runset.n_runs
    yo0 = [np.empty(frames.particles, dtype=np.single)] * runset.n_runs
    zo0 = [np.empty(frames.particles, dtype=np.single)] * runset.n_runs
    xo1 = np.empty(frames.particles, dtype=np.single)
    yo1 = np.empty(frames.particles, dtype=np.single)
    zo1 = np.empty(frames.particles, dtype=np.single)

  # Holds computed scattering functions. First 3 indices are for each
  # direction, last element is average of directions
  fc = np.empty(4, dtype=np.float64)

  if frames.n_atoms != None:
    # Orientational correlation functions for each Legendre polynomial
    # degree. In first dimension, first index is total part, second
    # index is self part, and third index is distinct part.
    ocorr = np.empty((3, svars.legendre), dtype=np.float64)

    # Matrix of Legendre polynomial coefficients
    legmat = np.zeros((svars.legendre, svars.legendre + 1), dtype=np.float64)
    for i in range(0, svars.legendre):
      legmat[i,:i + 2] = np.polynomial.legendre.leg2poly([0] * (i + 1) + [1])

    # Array for computed power terms for total correlation functions
    powtotal = np.empty(svars.legendre + 1, dtype=np.float64)
    powtotal[0] = frames.particles

    # Create array of precomputed factorials
    facts = np.cumprod(np.insert(np.arange(1, svars.legendre + 1), 0, 1))

  # Corresponding quantities for individual runs
  run_fc = np.empty(4, dtype=np.float64)
  if frames.n_atoms != None:
    run_ocorr = np.empty((3, svars.legendre), dtype=np.float64)

  if runset.rundirs == True:
    # Corresponding arrays used for calculating standard deviations
    # across runs
    std_fc = np.empty(4, dtype=np.float64)
    if frames.n_atoms != None:
      std_ocorr = np.empty((3, svars.legendre), dtype=np.float64)

  # Print description of output columns
  print("#",
        "#Output Columns:",
        "#  1 - Initial frame index",
        "#  2 - Initial time",
        "#  3 - Final frame index",
        "#  4 - Final time",
        "#  5 - Time difference for interval",
        sep="\n")
  if runset.rundirs == True:
    print("#  6 - Run average of mean squared displacement",
          "#  7 - Run average of average overlap",
          "#  8 - Run average of x self scattering function",
          "#  9 - Run average of y self scattering function",
          "#  10 - Run average of z self scattering function",
          "#  11 - Run average of directional average self scattering function",
          "#  12 - Standard deviation across runs of mean squared displacement",
          "#  13 - Standard deviation across runs of average overlap",
          "#  14 - Standard deviation across runs of x self scattering function",
          "#  15 - Standard deviation across runs of y self scattering function",
          "#  16 - Standard deviation across runs of z self scattering function",
          "#  17 - Standard deviation across runs of directional average self scattering function",
          sep="\n")
    if frames.n_atoms != None:
      print("#  18 - Run average of orientational correlation theta function average",
            "#  19 - Standard deviation across runs of orientational correlation theta function average",
            sep="\n")
      for i in range(1, svars.legendre + 1):
        print("#  %d - Run average of total part of Legendre degree %d of orientational correlation function" %(6 * i + 14, i),
              "#  %d - Run average of self part of Legendre degree %d of orientational correlation function" %(6 * i + 15, i),
              "#  %d - Run average of distinct part of Legendre degree %d of orientational correlation function" %(6 * i + 16, i),
              "#  %d - Standard deviation across runs of total part of Legendre degree %d of orientational correlation function" %(6 * i + 17, i),
              "#  %d - Standard deviation across runs of self part of Legendre degree %d of orientational correlation function" %(6 * i + 18, i),
              "#  %d - Standard deviation across runs of distinct part of Legendre degree %d of orientational correlation function" %(6 * i + 19, i),
              sep="\n")
  else:
    print("#  6 - Mean squared displacement",
          "#  7 - Average overlap",
          "#  8 - x self scattering function",
          "#  9 - y self scattering function",
          "#  10 - z self scattering function",
          "#  11 - Directional average self scattering function",
          sep="\n")
    if frames.n_atoms != None:
      print("#  12 - Orientational correlation theta function average")
      for i in range(1, svars.legendre + 1):
        print("#  %d - Total part of Legendre degree %d of orientational correlation function" %(3 * i + 10, i),
              "#  %d - Self part of Legendre degree %d of orientational correlation function" %(3 * i + 11, i),
              "#  %d - Distinct part of Legendre degree %d of orientational correlation function" %(3 * i + 12, i),
              sep="\n")
  print("#")

  # Iterate over initial frames, which serve as starting points for
  # functions
  for i in svars.initial:
    # Get particle data for initial frame for all runs
    for j in range(0, runset.n_runs):
      frames.get_frame(i, x0[j], y0[j], z0[j], j)
      if frames.n_atoms != None:
        frames.get_orientations(i, xo0[j], yo0[j], zo0[j], j)

    for j in range(i, frames.n_frames):
      # Clear cross-run accumulators
      fc[:] = 0.0
      msd = 0.0
      overlap = 0.0
      if frames.n_atoms != None:
        otheta = 0.0
        ocorr[:, :] = 0.0
      if runset.rundirs == True:
        std_fc[:] = 0.0
        std_msd = 0.0
        std_overlap = 0.0
        if frames.n_atoms != None:
          std_otheta = 0.0
          std_ocorr[:, :] = 0.0

      # Iterate over ending points for functions and add to
      # accumulated values, making sure to only use indices
      # which are within the range of the files.
      for k in range(0, runset.n_runs):
        frames.get_frame(j, x1, y1, z1, k)
        if frames.n_atoms != None:
          frames.get_orientations(j, xo1, yo1, zo1, k)

        # Compute scattering functions of all the particles for each
        # dimension
        run_fc[0] = np.mean(np.cos(svars.q * (x1 - x0[k])))
        run_fc[1] = np.mean(np.cos(svars.q * (y1 - y0[k])))
        run_fc[2] = np.mean(np.cos(svars.q * (z1 - z0[k])))

        # Compute average of directional scattering functions
        run_fc[3] = np.mean(run_fc[0:3])

        # Compute mean squared displacement
        run_msd = np.mean((x1 - x0[k])**2 + (y1 - y0[k])**2 + (z1 - z0[k])**2)

        # Compute overlap value
        run_overlap = np.mean(np.less((x1 - x0[k])**2 +
                                      (y1 - y0[k])**2 +
                                      (z1 - z0[k])**2, svars.radius**2).astype(np.int8, copy=False))

        if frames.n_atoms != None:
          # Compute orientation theta functions and self correlation

          # Find dot product of orientation vectors and evaluate
          # Legendre polynomials
          legself = np.polynomial.legendre.legval(((xo0[k]*xo1)+(yo0[k]*yo1)+(zo0[k]*zo1)), np.eye(N=svars.legendre + 1, M=svars.legendre, k=-1), tensor=True)

          # Calculate and accumulate theta functions
          run_otheta = np.mean(np.greater_equal(legself[0], svars.thetab).astype(int))

          # Accumulate to self part
          run_ocorr[1] = np.mean(legself, axis=1)

          # Compute orientation total correlation

          # Compute dot product total parts for required powers
          for l in range(1, svars.legendre + 1):
            powtotal[l] = 0.0
            # Compute each part of trinomial expansion of 3-component
            # dot product for given power
            for m in range(0, l + 1):
              for n in range(0, l + 1 - m):
                powtotal[l] += (facts[l]/(facts[m]*facts[n]*facts[l-m-n])) * np.sum(xo0[k]**m * yo0[k]**n * zo0[k]**(l+1-m-n)) * np.sum(xo1**m * yo1**n * zo1**(l-m-n)) / frames.particles

          # Compute Legendre polynomial from powers and accumulate
          run_ocorr[0] = np.sum(legmat * powtotal, axis=1)

          # Compute distinct part of orientation correlation from self
          # and total parts
          run_ocorr[2] = run_ocorr[0] - run_ocorr[1]

        # Accumulate computed values for later averaging
        fc += run_fc
        msd += run_msd
        overlap += run_overlap
        if frames.n_atoms != None:
          otheta += run_otheta
          ocorr += run_ocorr

        if runset.rundirs == True:
          # Accumulate squares, to be later used for standard deviation
          # calculation
          std_fc += run_fc**2
          std_msd += run_msd**2
          std_overlap += run_overlap**2
          if frames.n_atoms != None:
            std_otheta += run_otheta**2
            std_ocorr += run_ocorr**2

      # Find real times for printing
      itime = frames.frame_time(i)
      jtime = frames.frame_time(j)

      if runset.rundirs == True:
        # Normalize calculated values across runs
        fc /= runset.n_runs
        msd /= runset.n_runs
        overlap /= runset.n_runs
        std_fc /= runset.n_runs
        std_msd /= runset.n_runs
        std_overlap /= runset.n_runs
        if frames.n_atoms != None:
          otheta /= runset.n_runs
          ocorr /= runset.n_runs
          std_otheta /= runset.n_runs
          std_ocorr /= runset.n_runs

        # Calculate standard deviation with means and means of squares
        # of values
        std_fc = np.sqrt(np.maximum(0.0, std_fc - fc**2) / (runset.n_runs - 1))
        std_msd = np.sqrt(np.maximum(0.0, std_msd - msd**2) / (runset.n_runs - 1))
        std_overlap = np.sqrt(np.maximum(0.0, std_overlap - overlap**2) / (runset.n_runs - 1))
        if frames.n_atoms != None:
          std_ocorr = np.sqrt(np.maximum(0.0, std_ocorr - ocorr**2) / (runset.n_runs - 1))
          std_otheta = np.sqrt(np.maximum(0.0, std_otheta - otheta**2) / (runset.n_runs - 1))

        # Print output columns
        sys.stdout.write("%d %f %d %f %f %f %f %f %f %f %f %f %f %f %f %f %f"
                         %(i + 1,
                           itime,
                           j + 1,
                           jtime,
                           jtime - itime,
                           msd,
                           overlap,
                           fc[0],
                           fc[1],
                           fc[2],
                           fc[3],
                           std_msd,
                           std_overlap,
                           std_fc[0],
                           std_fc[1],
                           std_fc[2],
                           std_fc[3]))
        if frames.n_atoms != None:
          sys.stdout.write(" %f %f"
                           %(otheta,
                             std_otheta))
          for k in range(0, svars.legendre):
            sys.stdout.write(" %f %f %f %f %f %f"
                             %(ocorr[0][k],
                               ocorr[1][k],
                               ocorr[2][k],
                               std_ocorr[0][k],
                               std_ocorr[1][k],
                               std_ocorr[2][k]))
        sys.stdout.write("\n")

      else:
        # Print output columns
        sys.stdout.write("%d %f %d %f %f %f %f %f %f %f %f"
                         %(i + 1,
                           itime,
                           j + 1,
                           jtime,
                           jtime - itime,
                           msd,
                           overlap,
                           fc[0],
                           fc[1],
                           fc[2],
                           fc[3]))
        if frames.n_atoms != None:
          sys.stdout.write(" %f"
                           %(otheta))
          for k in range(0, svars.legendre):
            sys.stdout.write(" %f %f %f"
                             %(ocorr[0][k],
                               ocorr[1][k],
                               ocorr[2][k]))
        sys.stdout.write("\n")

# Run full program
lib.script.run(main_func,
               argtext=argtext,
               svars=svars,
               shortopts=shortopts,
               longopts=longopts,
               catch_opt=catch_opt,
               modules=[runset, trajset1, trajset2, frames])
