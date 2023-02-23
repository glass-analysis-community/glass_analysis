import numpy as np
import pydcd
import sys
import math
import getopt
import enum

# Import functionality from local library directory
import lib.opentraj

def usage():
  print("Arguments:",
        "-n Number of traj(n).dcd files (analyzed as first in sequence)",
        "-m Number of short(m).dcd files (analyzed as second in sequence)",
        "-z short(m).dcd file index to start on (default: 1)",
        "-s Frame number to start on (index starts at 1)",
        "-e Frame number to end on (index starts at 1)",
        "-i Initial frame numbers (start at 1, added to start specified with -s)",
        "-a Overlap radius for theta function (default: 0.25)",
        "-q Scattering vector constant (default: 7.25)",
        "-o Start index (from 1) of particles to limit analysis to",
        "-p End index (from 1) of particles to limit analysis to",
        "-h Print usage",
        sep="\n", file=sys.stderr)

try:
  opts, args = getopt.gnu_getopt(sys.argv[1:], "n:m:z:i:s:e:i:a:q:o:p:h")
except getopt.GetoptError as err:
  print(err, file=sys.stderr)
  usage()
  sys.exit(1)

# Total number of trajectory files
n_files = 0
# Number of short files
m_files = 0
# Start trajectory file index in filenames for second region
m_start = 1
# What frame number to start on
start = 0
# What frame number to end on
end = None
# If initial frame indices defined
initial_defined = False
# Overlap radius for theta function
radius = 0.25
# Scattering vector constant
q = 7.25
# Whether to limit analysis to subset of particles, and upper and lower
# indices for limit.
limit_particles = False
upper_limit = None
lower_limit = None

for o, a in opts:
  if o == "-h":
    usage()
    sys.exit(0)
  elif o == "-n":
    n_files = int(a)
  elif o == "-m":
    m_files = int(a)
  elif o == "-z":
    m_start = int(a)
  elif o == "-s":
    start = int(a) - 1
  elif o == "-e":
    end = int(a) - 1
  elif o == "-i":
    initial = np.array(list(map(int, a.split(",")))) - 1
    initial_defined = True
  elif o == "-a":
    radius = float(a)
  elif o == "-q":
    q = float(a)
  elif o == "-o":
    limit_particles = True
    lower_limit = int(a) - 1
  elif o == "-p":
    limit_particles = True
    upper_limit = int(a)

if initial_defined == False:
  raise RuntimeError("Must specify set of initial frame indices")

# Open trajectory files
dcdfiles, fileframes, fparticles, timestep, tbsaves = lib.opentraj.opentraj(n_files, "traj", 1, False)
short_dcdfiles, short_fileframes, fparticles, timestep, tbsaves = lib.opentraj.opentraj(m_files, "short", m_start, False)
dcdfiles += short_dcdfiles
fileframes = np.append(fileframes, short_fileframes[1:])

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
print("#q = %f" %q)
print("#a = %f" %radius)

# Number of frames to analyze
if end == None:
  n_frames = fileframes[-1] - start
else:
  n_frames = (end + 1) - start

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

# Center of mass of each frame
cm = np.empty((n_frames, 3), dtype=np.float64)

# Find center of mass of each frame
print("Finding centers of mass for frames", file=sys.stderr)
for i in range(0, n_frames):
  which_file = np.searchsorted(fileframes, start + i, side="right") - 1
  offset = start + i - fileframes[which_file]
  if limit_particles == True:
    dcdfiles[which_file].gdcdp(x, y, z, offset)
    x0[:] = x[lower_limit:upper_limit]
    y0[:] = y[lower_limit:upper_limit]
    z0[:] = z[lower_limit:upper_limit]
  else:
    dcdfiles[which_file].gdcdp(x0, y0, z0, offset)

  cm[i][0] = np.mean(x0)
  cm[i][1] = np.mean(y0)
  cm[i][2] = np.mean(z0)

# Iterate over starting points for functions
for i in initial:
  i_which_file = np.searchsorted(fileframes, start + i, side="right") - 1
  i_offset = start + i - fileframes[i_which_file]
  if limit_particles == True:
    dcdfiles[i_which_file].gdcdp(x, y, z, i_offset)
    x0[:] = x[lower_limit:upper_limit]
    y0[:] = y[lower_limit:upper_limit]
    z0[:] = z[lower_limit:upper_limit]
  else:
    dcdfiles[i_which_file].gdcdp(x0, y0, z0, i_offset)

  # Iterate over ending points for functions and add to
  # accumulated values, making sure to only use indices
  # which are within the range of the files.
  for j in range(i, n_frames):
    j_which_file = np.searchsorted(fileframes, start + j, side="right") - 1
    j_offset = start + j - fileframes[j_which_file]
    if limit_particles == True:
      dcdfiles[j_which_file].gdcdp(x, y, z, j_offset)
      x1[:] = x[lower_limit:upper_limit]
      y1[:] = y[lower_limit:upper_limit]
      z1[:] = z[lower_limit:upper_limit]
    else:
      dcdfiles[which_file].gdcdp(x1, y1, z1, j_offset)

    # Compute scattering functions of all the particles for each
    # dimension
    fcx = np.mean(np.cos(q * ((x1 - cm[j][0]) - (x0 - cm[i][0]))))
    fcy = np.mean(np.cos(q * ((y1 - cm[j][1]) - (y0 - cm[i][1]))))
    fcz = np.mean(np.cos(q * ((z1 - cm[j][2]) - (z0 - cm[i][2]))))

    # Compute mean squared displacement
    msd = np.mean(((x1 - cm[j][0]) - (x0 - cm[i][0]))**2 +
                  ((y1 - cm[j][1]) - (y0 - cm[i][1]))**2 +
                  ((z1 - cm[j][2]) - (z0 - cm[i][2]))**2)

    # Compute overlap value
    overlap = np.mean(np.less(((x1 - cm[j][0]) - (x0 - cm[i][0]))**2 +
                              ((y1 - cm[j][1]) - (y0 - cm[i][1]))**2 +
                              ((z1 - cm[j][2]) - (z0 - cm[i][2]))**2, radius**2).astype(np.int8, copy=False))

    itime = dcdfiles[i_which_file].itstart + i_offset * timestep * dcdfiles[i_which_file].tbsave
    jtime = dcdfiles[j_which_file].itstart + j_offset * timestep * dcdfiles[j_which_file].tbsave
    # Print initial frame index, initial time, final frame index, final
    # time, and time difference for interval, along with msd, averarge
    # overlap, x, y, and z scattering functions, and average scattering
    # function for all directions along such interval
    print("%d %f %d %f %f %f %f %f %f %f %f" %(i, itime, j, jtime, jtime - itime, msd, overlap, fcx, fcy, fcz, (fcx+fcy+fcz)/3))
