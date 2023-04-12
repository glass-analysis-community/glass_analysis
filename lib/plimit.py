import sys

def usage():
  print("-o Start index (from 1) of particles to limit analysis to",
        "-p End index (from 1) of particles to limit analysis to",
        sep="\n", file=sys.stderr)

# List of short options processed by this module, used by gnu_getopt()
shortopts = "o:p:"

# Particle-limiting object
class plimit():
  # Whether to limit analysis to subset of particles, and upper and lower
  # indices for limit.
  limit_particles = False
  upper_limit = None
  lower_limit = None

  def final(self, fparticles):
    if self.limit_particles == False:
      particles = fparticles
    else:
      if self.lower_limit == None:
        self.lower_limit = 0
      if self.upper_limit == None:
        self.upper_limit = fparticles

      if self.lower_limit != 0 or self.upper_limit < fparticles:
        particles = self.upper_limit - self.lower_limit
        limit_particles = True
      else:
        particles = fparticles
        limit_particles = False

    return particles, self.lower_limit, self.upper_limit, limit_particles

  # Determine if option corresponds to this module and process it if so.
  # Returns True if option matched and processed.
  def catch_opt(self, o, a):
    if o == "-o":
      self.lower_limit = int(a) - 1
    elif o == "-p":
      self.upper_limit = int(a)
    else:
      # Option not matched
      return False

    return True
