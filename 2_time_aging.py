import numpy as np
import pydcd
import sys
import math
import getopt
import enum

# Import functionality from local library directory
import lib.opentraj
import lib.frame

# Start trajectory file index in filenames for second region
m_start = 1
# What frame number to end on
end = None
# List of initial frame indices
initial = None
# Overlap radius for theta function
radius = 0.25
# Scattering vector constant
q = 7.25
# Run set opening object
runset = lib.opentraj.runset()
# First (aging) region trajectory set opening object
trajset1 = lib.opentraj.trajset(runset, opt="n", name="traj")
# Second (geometric) region trajectory set opening object. Default
# number of short trajectories is 0.
trajset2 = lib.opentraj.trajset(runset, opt="t", name="short", default_n_files=0)
# First (aging) region frame reading object
frames = lib.frame.frames([trajset1, trajset2])

def usage():
  print("Arguments:", file=sys.stderr)
  runset.usage()
  trajset1.usage()
  trajset2.usage()
  frames.usage()
  print("-z short(m).dcd file index to start on (default: 1)",
        "-i Initial frame numbers specified as list separated by commas (start at 1, offset by start specified with -s)",
        "-a Overlap radius for theta function (default: 0.25)",
        "-q Scattering vector constant (default: 7.25)",
        "-h Print usage",
        sep="\n", file=sys.stderr)

try:
  opts, args = getopt.gnu_getopt(sys.argv[1:], "z:i:a:q:h" +
                                               runset.shortopts +
                                               trajset1.shortopts +
                                               trajset2.shortopts +
                                               frames.shortopts)
except getopt.GetoptError as err:
  print(err, file=sys.stderr)
  usage()
  sys.exit(1)

for o, a in opts:
  if o == "-h":
    usage()
    sys.exit(0)
  elif o == "-z":
    m_start = int(a)
  elif o == "-i":
    initial = np.array(list(map(int, a.split(",")))) - 1
  elif o == "-a":
    radius = float(a)
  elif o == "-q":
    q = float(a)
  elif runset.catch_opt(o, a) == True:
    pass
  elif trajset1.catch_opt(o, a) == True:
    pass
  elif trajset2.catch_opt(o, a) == True:
    pass
  elif frames.catch_opt(o, a) == True:
    pass

if initial is None:
  raise RuntimeError("Must specify set of initial frame indices")

# Open trajectory files for both trajectory types
if runset.rundirs == True:
  trajset1.opentraj_multirun(1, False)
  trajset2.opentraj_multirun(m_start, False)
else:
  trajset1.opentraj(1, False)
  trajset2.opentraj(m_start, False)

# Prepare frames object for calculation
frames.prepare()

if np.any(initial >= frames.n_frames):
  raise RuntimeError("At least one initial frame number is beyond range of frames")

# Print basic properties of files and analysis
print("#nset: %d" %frames.fileframes[-1])
print("#N: %d" %frames.fparticles)
print("#q = %f" %q)
print("#a = %f" %radius)

# Stores coordinates of all particles in initial frames
x0 = [np.empty(frames.particles, dtype=np.single)] * runset.n_runs
y0 = [np.empty(frames.particles, dtype=np.single)] * runset.n_runs
z0 = [np.empty(frames.particles, dtype=np.single)] * runset.n_runs

# Stores coordinates of all particles in end frames
x1 = np.empty(frames.particles, dtype=np.single)
y1 = np.empty(frames.particles, dtype=np.single)
z1 = np.empty(frames.particles, dtype=np.single)

# Holds computed scattering functions and their standard deviations,
# first 3 elements are for each direction, last element is average of
# directions
run_fc = np.empty(4, dtype=np.float64)
fc = np.empty(4, dtype=np.float64)
std_fc = np.empty(4, dtype=np.float64)

# Find center of mass of each frame
print("Finding centers of mass for frames", file=sys.stderr)
frames.generate_cm()

# Iterate over initial frames, which serve as starting points for
# functions
for i in initial:
  # Get particle data for initial frame for all runs
  for j in range(0, runset.n_runs):
    frames.get_frame(i, x0[j], y0[j], z0[j], j)

  for j in range(i, frames.n_frames):
    # Clear cross-run accumulators
    fc[:] = 0.0
    msd = 0.0
    overlap = 0.0
    std_fc[:] = 0.0
    std_msd = 0.0
    std_overlap = 0.0

    # Iterate over ending points for functions and add to
    # accumulated values, making sure to only use indices
    # which are within the range of the files.
    for k in range(0, runset.n_runs):
      frames.get_frame(j, x1, y1, z1, k)

      # Compute scattering functions of all the particles for each
      # dimension
      run_fc[0] = np.mean(np.cos(q * (x1 - x0[k])))
      run_fc[1] = np.mean(np.cos(q * (y1 - y0[k])))
      run_fc[2] = np.mean(np.cos(q * (z1 - z0[k])))

      # Compute average of directional scattering functions
      run_fc[3] = np.mean(run_fc[0:3])

      # Compute mean squared displacement
      run_msd = np.mean((x1 - x0[k])**2 + (y1 - y0[k])**2 + (z1 - z0[k])**2)

      # Compute overlap value
      run_overlap = np.mean(np.less((x1 - x0[k])**2 +
                                    (y1 - y0[k])**2 +
                                    (z1 - z0[k])**2, radius**2).astype(np.int8, copy=False))

      # Accumulate computed values for later averaging
      fc += run_fc
      msd += run_msd
      overlap += run_overlap

      if runset.rundirs == True:
        # Accumulate squares, to be later used for standard deviation
        # calculation
        std_fc += run_fc**2
        std_msd += run_msd**2
        std_overlap += run_overlap**2

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

      # Calculate standard deviation with means and means of squares of
      # values
      std_fc = np.sqrt(np.maximum(0.0, std_fc - fc**2) / (runset.n_runs - 1))
      std_msd = np.sqrt(np.maximum(0.0, std_msd - msd**2) / (runset.n_runs - 1))
      std_overlap = np.sqrt(np.maximum(0.0, std_overlap - overlap**2) / (runset.n_runs - 1))

      # Print output columns:
      # 1 - initial frame index
      # 2 - initial time
      # 3 - final frame index
      # 4 - final time
      # 5 - time difference for interval
      # 6 - mean squared displacement
      # 7 - averarge overlap run average
      # 8 - x scattering function run average
      # 9 - y scattering function run average
      # 10 - z scattering function run average
      # 11 - directional average scattering function run average
      # 12 - mean squared displacement standard deviation
      # 13 - average overlap standard deviation
      # 14 - x scattering function standard deviation
      # 15 - y scattering function standard deviation
      # 16 - z scattering function standard deviation
      # 17 - directional average scattering function standard deviation
      print("%d %f %d %f %f %f %f %f %f %f %f %f %f %f %f %f %f" %(i + 1,
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

    else:
      # Print output columns:
      # 1 - initial frame index
      # 2 - initial time
      # 3 - final frame index
      # 4 - final time
      # 5 - time difference for interval
      # 6 - mean squared displacement
      # 7 - averarge overlap
      # 8 - x scattering function
      # 9 - y scattering function
      # 10 - z scattering function
      # 11 - directional average scattering function
      print("%d %f %d %f %f %f %f %f %f %f %f" %(i + 1,
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
