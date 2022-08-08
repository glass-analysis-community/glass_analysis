import numpy as np
import itertools
import pydcd
import sys
import math
import getopt
import enum

def usage():
  print("Arguments:",
        "-p Number of particles along one dimension (total particles is cube of this)",
        "-s Standard deviation (sigma) for use in random function",
        "-f Number of frames in generated trajectory",
        "-d Initial particle density, over dcd file length units",
        "-h Print usage",
        "Initial vector types:",
        "-z Use zero vector for constant displacement",
        "-i Use same distribution as single particle vectors for constant displacement",
        "-n Use distribution of single particles divided by sqrt(n_particles) for constant displacement",
        sep="\n", file=sys.stderr)

class vectypes(enum.Enum):
  zero = 1
  dist = 2
  narrowdist = 3

try:
  opts, args = getopt.gnu_getopt(sys.argv[1:], "p:s:f:d:hzin")
except getopt.GetoptError as err:
  print(err, file=sys.stderr)
  usage()
  sys.exit(1)

# Number of particles
n_particles = None
# Initial particle density of system
density = None
# Standard deviation to use for Gaussian offset function
sigma = None
# Number of trajectory frames to generate
n_frames = None
# Type of distribution to use for constant vector generation
vectype = None

for o, a in opts:
  if o == "-h":
    usage()
    sys.exit(0)
  elif o == "-p":
    n_particles = int(a)
  elif o == "-s":
    sigma = float(a)
  elif o == "-f":
    n_frames = int(a)
  elif o == "-d":
    density = float(a)
  elif o == "-z":
    vectype = vectypes.zero
  elif o == "-i":
    vectype = vectypes.dist
  elif o == "-n":
    vectype = vectypes.narrowdist

if n_particles == None:
  raise RuntimeError("Must specify number of particles")
if sigma == None:
  raise RuntimeError("Must specify sigma^2 parameter for Gaussian offset function")
if n_frames == None:
  raise RuntimeError("Must specify number of frames in generated trajectory")
if density == None:
  raise RuntimeError("Must specify initial density of system")
if vectype == None:
  raise RuntimeError("Must specify constant vector distribution type")

# Create dcd file with reasonable parameters
traj_file = pydcd.create("traj1.dcd", n_particles**3, 0, 1, 1.0, 1)

# Calculate half-extent of box for given density and number of
# particles
L = (n_particles - 1) * density**(1/3)

# Initialize positions of particles
linarray = np.linspace(-L, L, num=n_particles)
particles = np.meshgrid(linarray, linarray, linarray)
x = particles[0].astype(np.single)
y = particles[1].astype(np.single)
z = particles[2].astype(np.single)

# Unshuffled differences
dx = np.empty(n_particles**3, dtype=np.single)
dy = np.empty(n_particles**3, dtype=np.single)
dz = np.empty(n_particles**3, dtype=np.single)

# Initialize random number generation
rng = np.random.default_rng()

if vectype == vectypes.zero:
  lmx = 0.0
  lmy = 0.0
  lmz = 0.0
elif vectype == vectypes.dist:
  lmx = rng.normal(scale=sigma)
  lmy = rng.normal(scale=sigma)
  lmz = rng.normal(scale=sigma)
elif vectype == vectypes.narrowdist:
  lmx = rng.normal(scale=sigma / math.sqrt(n_particles**3))
  lmy = rng.normal(scale=sigma / math.sqrt(n_particles**3))
  lmz = rng.normal(scale=sigma / math.sqrt(n_particles**3))

# Generate and write frames
for i in range(0, n_frames):
  if n_particles % 2 == 1:
    dx[:(n_particles**3 // 2) + 1] = rng.normal(scale=sigma, size=((n_particles**3 // 2) + 1))
    dy[:(n_particles**3 // 2) + 1] = rng.normal(scale=sigma, size=((n_particles**3 // 2) + 1))
    dz[:(n_particles**3 // 2) + 1] = rng.normal(scale=sigma, size=((n_particles**3 // 2) + 1))
    dx[(n_particles**3 // 2) + 1:] = -dx[:n_particles**3 // 2]
    dy[(n_particles**3 // 2) + 1:] = -dy[:n_particles**3 // 2]
    dz[(n_particles**3 // 2) + 1:] = -dz[:n_particles**3 // 2]
  else:
    dx[:n_particles**3 // 2] = rng.normal(scale=sigma, size=(n_particles**3 // 2))
    dy[:n_particles**3 // 2] = rng.normal(scale=sigma, size=(n_particles**3 // 2))
    dz[:n_particles**3 // 2] = rng.normal(scale=sigma, size=(n_particles**3 // 2))
    dx[n_particles**3 // 2:] = -dx[:n_particles**3 // 2]
    dy[n_particles**3 // 2:] = -dy[:n_particles**3 // 2]
    dz[n_particles**3 // 2:] = -dz[:n_particles**3 // 2]

  dx += lmx
  dy += lmy
  dz += lmz
  x += rng.permutation(dx).reshape((n_particles, n_particles, n_particles))
  y += rng.permutation(dy).reshape((n_particles, n_particles, n_particles))
  z += rng.permutation(dz).reshape((n_particles, n_particles, n_particles))
  traj_file.adcdp(x, y, z)
