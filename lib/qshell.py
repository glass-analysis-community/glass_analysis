import numpy as np
import sys
import math

# List of short options processed by this module, used by gnu_getopt()
shortopts = "q:v:l:"

def usage():
  """
  Print help documentation for options processed by the qshell module.
  """
  print("q-vector shell sorting:",
        "-q Upper boundary for first q region with discrete q values",
        "-v Upper boundary for second q region divided into onion shells",
        "-l Number of onion shells to use in second q region",
        sep="\n", file=sys.stderr)

class qshell():
  """
  Class used for sorting of q vectors into shells based on vector
  length.

  Attributes:
    qb1: float - Upper (included) boundary of range of q magnitudes
                 included in first region, which are recorded
                 according to discrete magnitudes. Value is in real
                 units.
    qb2: float - Upper (included) boundary of range of q magnitudes
                 included in second regions, which is partitioned into
                 onion shells of q magnitude ranges and recorded based
                 on which onion shell the magnitude resides in. Value
                 is in real units.
    qb1a: float - qb1 expressed as a multiple of the smallest positive
                  q magnitude that is periodic within box boundaries,
                  corresponding to the q increment represented by each
                  FFT cell.
    qb2a: float - qb2 expressed as a multiple of the smallest positive
                  q magnitude that is periodic within box boundaries,
                  corresponding to the q increment represented by each
                  FFT cell.
    qb2l: int - Lowest index offset from 0 q vector in spatial FFT
                array that falls within second q magnitude region.
    qb2u: int - Highest index offset from 0 q vector in spatial FFT
                array that falls within second q magnitude region.
    shells: int - Number of onion shells to use in second region
    swidth: float - Width of each onion shell in second region
    size_fft: int - Size of FFT array in each direction
    element_qs: np.array(dtype=np.int64) - Array with same dimensions
                                           as FFT array with 2 numbers
                                           corresponding to each value.
                                           First number indicates which
                                           region (or none) q vector is
                                           to be found within. Second
                                           number indicates the index
                                           of the magnitude value
                                           within qlist_discrete_sorted
                                           or qlist_shells to which the
                                           given q vector corresponds
                                           and contributes to.
    qlist_discrete_sorted: list(float) - List of discrete q vector
                                         magnitudes within the first
                                         region.
    qnorm_discrete_sorted: list(int) - Number of q vectors contributing
                                       to each magnitude value in
                                       qlist_discrete_sorted
    qlist_shells: list(float) - Array of shell indexes used. May be
                                less than number of specified shells,
                                as some shells may have no q vectors
                                that fall within their given q
                                magnitude range.
    qnorm_shells: list(int) - Number of q vectors contributing to each
                              magnitude range in qlist_shells
  """

  def prepare(self, size_fft, box_size):
    """
    Prepare qshell object for sorting values into shells later.
    Calculate shell for each spatial element (element_qs), record
    number of elements for each shell (qnorm*), record values of each
    shell or discrete magnitude (qlist*), save size_fft, and calculate
    limits of FFT spatial elements used in shells (qb2l, qb2u).

    Arguments:
      size_fft: int - Number of FFT elements in each dimension
      box_size: float - Real length of simulation box in each dimension
    """
    if self.qb1 == None:
      raise RuntimeError("Must specify upper q boundary for first region if nonzero q values used")

    if self.qb2 == None:
      raise RuntimeError("Must specify upper q boundary for second region if nonzero q values used")

    if self.shells == None:
      raise RuntimeError("Must specify number of onion shells in second region if nonzero q values used")

    # Convert upper boundaries of regions to multipliers of smallest q
    # magnitude
    self.qb1a = self.qb1 * box_size / (2 * math.pi)
    self.qb2a = self.qb2 * box_size / (2 * math.pi)

    # Record size_fft for later use
    self.size_fft = size_fft

    # Upper and lower bounds for dimensions of q that fit within qb2,
    # used for matrix dimensioning
    self.qb2l = max(-int(self.qb2a), -(size_fft // 2))
    self.qb2u = min(int(self.qb2a), (size_fft - 1) // 2)

    # Shell width
    if self.shells != 0:
      self.swidth = (self.qb2a - self.qb1a) / self.shells

    # List of shell numbers to use for shell intervals
    self.qlist_shells = list(range(0, self.shells))

    # List of q values to use for region of discrete q values
    qlist_discrete = list()

    # Norm for number of FFT matrix elements corresponding to each element
    # of qlist, for first and second regions.
    self.qnorm_shells = [0] * self.shells
    qnorm_discrete = list()

    # Array of indices in qlist matrix elements correspond to. The first
    # number in the last dimension is whether the index is not in the q
    # range (-1), is within the first region of discrete q values (0), or
    # is within the second region of shells (1). The second number in the
    # last dimension is the qlist index.
    self.element_qs = np.empty((self.qb2u - self.qb2l + 1, self.qb2u - self.qb2l + 1, self.qb2u + 1, 2), dtype=np.int64)

    # Initialize to default of no corresponding index
    self.element_qs[:, :, :, 0] = -1

    # Find q lengths corresponding to each set of q coordinates
    for i in range(self.qb2l, self.qb2u + 1):
      for j in range(self.qb2l, self.qb2u + 1):
        for k in range(0, self.qb2u + 1):
          hyp = float(np.linalg.norm((i, j, k)))
          if hyp > self.qb1a:
            # Index of onion shell that would include given q
            shell_index = self.shells - int((self.qb2a - hyp) // self.swidth) - 1
            if shell_index < self.shells:
              self.element_qs[i - self.qb2l][j - self.qb2l][k][0] = 1
              self.element_qs[i - self.qb2l][j - self.qb2l][k][1] = shell_index
              self.qnorm_shells[shell_index] += 1
          else:
            if not (hyp in qlist_discrete):
              qlist_discrete.append(hyp)
              qnorm_discrete.append(0)
            self.element_qs[i - self.qb2l][j - self.qb2l][k][0] = 0
            self.element_qs[i - self.qb2l][j - self.qb2l][k][1] = qlist_discrete.index(hyp)
            qnorm_discrete[qlist_discrete.index(hyp)] += 1

    # Sorted copies of discrete qlist and qnorm
    self.qlist_discrete_sorted, self.qnorm_discrete_sorted = zip(*sorted(zip(qlist_discrete, qnorm_discrete)))

    # Delete q elements with 0 norm (possible for shells)
    for i in reversed(range(0, len(self.qlist_shells))):
      if self.qnorm_shells[i] == 0:
        self.qlist_shells.pop(i)
        self.qnorm_shells.pop(i)
        # Shift element_qs values to take into account new ordering of
        # qlistsorted
        for j in range(self.qb2l, self.qb2u + 1):
          for k in range(self.qb2l, self.qb2u + 1):
            for l in range(0, self.qb2u + 1):
              # If within shell region and above deleted shell value
              if self.element_qs[j][k][l][0] == 1 and self.element_qs[j][k][l][1] >= i:
                self.element_qs[j][k][l][1] -= 1

    # Modify element_qs values to point to indices in qlistsorted rather
    # than qlist
    for i in range(self.qb2l, self.qb2u + 1):
      for j in range(self.qb2l, self.qb2u + 1):
        for k in range(0, self.qb2u + 1):
          # Only sort discrete values, shell values already sorted
          if self.element_qs[i][j][k][0] == 0:
            self.element_qs[i][j][k][1] = self.qlist_discrete_sorted.index(qlist_discrete[self.element_qs[i][j][k][1]])

  def to_shells(self, values):
    """
    Sort a set of FFT values organized by spatial dimension into
    averaged values for q vector values, consisting of regions of both
    discrete q vector magnitudes and onion shells of q vector magnitude
    ranges. Returns two arrays, the first for discrete q vector
    magnitudes and the second for shells of q magnitude ranges. Index
    for q magnitudes is last dimension of returned arrays.

    Arguments:
      values: np.array() -  Values to be sorted and averaged into
                            output values based on q magnitude. Last
                            three dimensions are taken to be x, y, and
                            z spatial indices for q vector.
    """
    # Create accumulators for values in shells with same shape as
    # input values array, except for last three dimensions, which are
    # spatial dimensions that are converted to a single dimension for
    # the shell index
    qaccum_discrete = np.zeros_like(values[...,0,0,0], shape=(*(values[...,0,0,0].shape), len(self.qlist_discrete_sorted)))
    qaccum_shells = np.zeros_like(values[...,0,0,0], shape=(*(values[...,0,0,0].shape), len(self.qlist_shells)))

    for i in range(self.qb2l, self.qb2u + 1):
      for j in range(self.qb2l, self.qb2u + 1):
        for k in range(0, self.qb2u + 1):
          # Index of qlist we are to use
          qcurrent = self.element_qs[i - self.qb2l][j - self.qb2l][k]

          # If matrix element corresponds to used q value in either
          # qlist_discrete_sorted or qlist_shells
          if self.element_qs[i - self.qb2l][j - self.qb2l][k][0] == 0:
            # Accumulate values to corresponding discrete q length
            qaccum_discrete[..., self.element_qs[i - self.qb2l][j - self.qb2l][k][1]] += values[..., (self.size_fft//2)+i, (self.size_fft//2)+j, k]
          elif self.element_qs[i - self.qb2l][j - self.qb2l][k][0] == 1:
            # Accumulate values to corresponding q shell index
            qaccum_shells[..., self.element_qs[i - self.qb2l][j - self.qb2l][k][1]] += values[..., (self.size_fft//2)+i, (self.size_fft//2)+j, k]

    # Normalize q values for number of contributing elements
    qaccum_discrete /= self.qnorm_discrete_sorted
    qaccum_shells /= self.qnorm_shells

    return qaccum_discrete, qaccum_shells

  def catch_opt(self, o, a):
    """
    Determine if option corresponds to the qshell module and process
    it if so. Returns True if option matched and processed, False
    otherwise.

    Arguments:
      o: str - Name of option to process, from array produced by
               gnu_getopt().
      a: str - Value of option to process, from array produced by
               gnu_getopt().
    """
    if o == "-q":
      self.qb1 = float(a)
    elif o == "-v":
      self.qb2 = float(a)
    elif o == "-l":
      self.shells = int(a)
    else:
      # Option not matched
      return False

    return True
