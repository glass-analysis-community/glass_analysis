import numpy as np
import itertools
import pydcd
import sys
import math
import getopt

def usage():
  print("Arguments:",
        "-p Number of particles along one dimension (total particles is cube of this)",
        "-s Sigma for use in random function",
        "-f Number of frames in generated trajectory",
        "-d Initial particle density, over dcd file length units",
        "-h Print usage",
        sep="\n", file=sys.stderr)

try:
  opts, args = getopt.gnu_getopt(sys.argv[1:], "p:s:f:d:h")
except getopt.GetoptError as err:
  print(err, file=sys.stderr)
  usage()
  sys.exit(1)

# Number of particles
n_particles = None
# Initial particle density of system
density = None
# Sigma to use for Gaussian offset function
sigma = None
# Number of trajectory frames to generate
n_frames = None

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

if n_particles == None:
  raise RuntimeError("Must specify number of particles")
if sigma == None:
  raise RuntimeError("Must specify sigma parameter for Gaussian offset function")
if n_frames == None:
  raise RuntimeError("Must specify number of frames in generated trajectory")
if density == None:
  raise RuntimeError("Must specify initial density of system")

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

# Initialize random number generation
rng = np.random.default_rng()

# Generate and write frames
for i in range(0, n_frames):
  x += rng.normal(scale=sigma, size=(n_particles, n_particles, n_particles))
  y += rng.normal(scale=sigma, size=(n_particles, n_particles, n_particles))
  z += rng.normal(scale=sigma, size=(n_particles, n_particles, n_particles))
  traj_file.adcdp(x, y, z)
