import numpy as np
import sys
import math

class qshell():
  """
  Class used for sorting of q vectors into shells based on vector
  length.

  Attributes:
    active: bool - Whether or not q vector shells, and therefore this
      module, is to be used, as determined by whether or not an
      argument for qshell has been specified on the command line
    qb1: float - Upper (included) boundary of range of q magnitudes
      included in first region, which are recorded according to
      discrete magnitudes. Value is in real units.
    qb2: float - Upper (included) boundary of range of q magnitudes
      included in second regions, which is partitioned into onion
      shells of q magnitude ranges and recorded based on which onion
      shell the magnitude resides in. Value is in real units.
    qb1a: float - qb1 expressed as a multiple of the smallest positive
      q magnitude that is periodic within box boundaries, corresponding
      to the q increment represented by each FFT cell.
    qb2a: float - qb2 expressed as a multiple of the smallest positive
      q magnitude that is periodic within box boundaries, corresponding
      to the q increment represented by each FFT cell.
    qb2s: int - Largest element withi qb2 which will fit inside smaller
      direction of values in fft matrix
    qb2g: int - Largest element withi qb2 which will fit inside greater
      direction of values in fft matrix
    shells: int - Number of onion shells to use in second region
    swidth: float - Width of each onion shell in second region
    trim_q2: tuple - Tuple used for trimming input values to be sorted
    element_qs: np.array(dtype=np.int64) - Array with same dimensions
      as FFT array with an index corresponding to how the given q
      vector is to be sorted. Indices less than len(qlist_discrete)
      refer to discrete q vectors in first region and indices greater
      than len(qlist_discrete) refer to q vectors falling into either a
      shell in the second region or outside both regions. The offset
      of the index above the lower limit for each region specifies the
      offset into the corresponding qlist array that the q vector
      corresponds to.
    qlist_discrete: list(float) - List of discrete q vector magnitudes
      within the first region.
    qnorm_discrete: list(int) - Number of q vectors contributing to
      each magnitude value in qlist_discrete
    qlist_shells: list(float) - Array of shell indices used. May be
      less than number of specified shells, as some shells may have no
      q vectors that fall within their q magnitude range.
    qnorm_shells: list(int) - Number of q vectors contributing to each
      magnitude range in qlist_shells
    shortopts: str - List of short options processed by this module,
      used by gnu_getopt()
    argtext: str - Description of arguments processed by this module
  """
  active = False

  qb1 = None
  qb2 = None
  shells = None

  shortopts = "q:v:l:"

  argtext = "q-vector shell sorting:\n" \
            + "  -q Upper boundary for first q region with discrete q values\n" \
            + "  -v Upper boundary for second q region divided into onion shells\n" \
            + "  -l Number of onion shells to use in second q region"

  def prepare(self, size_fft, box_size):
    """
    Prepare qshell object for sorting values into shells later.
    Calculate shell for each spatial element (element_qs), record
    number of elements for each shell (qnorm*), record values of each
    shell or discrete magnitude (qlist*), save size_fft, create initial
    trimming range for sorted values (trim_q2), and calculate limits of
    FFT spatial elements used in shells (qb2g, qb2s).

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

    # Greater and smaller bounds for dimensions of q that fit within
    # second q magnitude region, for different directional behaviours
    self.qb2g = min(int(self.qb2a), size_fft // 2)
    self.qb2s = min(int(self.qb2a), (size_fft - 1) // 2)

    # Tuple used for trimming input values arrays to used values that
    # fall withing second q magnitude region
    self.trim_q2 = (slice(size_fft//2 - self.qb2g, size_fft//2 + self.qb2s + 1),
                    slice(size_fft//2 - self.qb2g, size_fft//2 + self.qb2s + 1),
                    slice(0, self.qb2g + 1))

    # Shell width
    if self.shells != 0:
      self.swidth = (self.qb2a - self.qb1a) / self.shells

    # List of shell numbers to use for shell intervals
    self.qlist_shells = np.arange(0, self.shells)

    # Construct array of squared hypotenuses (q magnitudes)
    hyp = np.zeros((self.qb2g + 1 + self.qb2s, ) * 3, dtype=np.float64)
    dim_contrib = np.arange(-self.qb2g, 1 + self.qb2g)**2
    hyp += dim_contrib[self.qb2g-self.qb2s:]
    hyp += dim_contrib[:self.qb2g+1+self.qb2s,None]
    hyp += dim_contrib[:self.qb2g+1+self.qb2s,None,None]

    # Find list of q magitude values that fall within first discrete
    # q magnitude region, as well as number of elements contributing
    # to each magnitude. Trim to remove values beyond first region.
    # Values in the second region are clipped to maximum value so that
    # when shell indices are added, they point to indices of qnorm
    # corresponding to shells.
    self.qlist_discrete, element_qs_discrete, self.qnorm_discrete = np.unique(hyp, return_inverse=True, return_counts=True)
    q1_high_index = np.searchsorted(self.qlist_discrete, self.qb1a**2, side="right")
    self.qlist_discrete = np.sqrt(self.qlist_discrete[:q1_high_index])
    self.qnorm_discrete = self.qnorm_discrete[:q1_high_index]
    element_qs_discrete = np.minimum(q1_high_index - 1, element_qs_discrete)

    # Construct bin edges for number of elements contributing to each
    # shell in second region of q magnitude shells and find location
    # of each element in shells. If shell number is 0, element is in
    # first region.
    q2_edges = np.linspace(self.qb1a, self.qb2a, num=self.shells + 1)
    element_qs_shells = np.searchsorted(q2_edges**2, hyp, side="left")
    self.qnorm_shells = np.histogram(element_qs_shells, bins=self.shells, range=(0.5, self.shells + 0.5))[0]

    # Remove shells in second region with no contributing elements
    empty_shells = np.argwhere(self.qnorm_shells == 0).flatten()
    if empty_shells.size > 0:
      self.qnorm_shells = np.delete(self.qnorm_shells, empty_shells)
      self.qlist_shells = np.delete(self.qlist_shells, empty_shells)
      element_qs_shells -= np.searchsorted(empty_shells + 1, element_qs_shells, side="right")

    # Create combined qnorm array for both regions
    self.qnorm = np.concatenate((self.qnorm_discrete, self.qnorm_shells))

    # Add element_qs for first and second region together so that they
    # correspond to correct index in qnorm.
    self.element_qs = element_qs_discrete.reshape(hyp.shape) + element_qs_shells

    # Cut out portion of FFT elements array to account for reduced size
    # due to real-valued FFT
    self.element_qs = self.element_qs[...,self.qb2s:]

  def to_shells(self, values):
    """
    Sort a set of FFT values organized by spatial dimension into
    averaged values for q vector values, consisting of regions of both
    discrete q vector magnitudes and onion shells of q vector magnitude
    ranges. Returns two arrays, the first for discrete q vector
    magnitudes and the second for shells of q magnitude ranges. Index
    for q magnitudes is last dimension of returned arrays.

    Arguments:
      values: np.array() - Values to be sorted and averaged into output
        values based on q magnitude. Last three dimensions are taken to
        be x, y, and z spatial indices for q vector. The values may be
        modified.
    """
    # Create accumulators for values with same shape as input values
    # array, except for last three dimensions, which are spatial
    # dimensions that are converted to a single dimension for the shell
    # index
    qout = np.zeros_like(values[...,0,0,0], shape=(*(values[...,0,0,0].shape), len(self.qnorm)))

    # Trim to only include used elements
    mod_values = values[(...,*self.trim_q2)]

    # Account for both directions of q vector for elements that have
    # combined due to real-valued FFT. qnorm accounts for this due to
    # the way the array of hypotenuses was constructed during
    # preparation.
    mod_values[...,1:1+self.qb2s] *= 2

    # Bin values according to q magnitude index in element_qs
    for i in np.ndindex(values[...,0,0,0].shape):
      qout[i] = np.histogram(self.element_qs, bins=len(self.qnorm), range=(-0.5, len(self.qnorm) - 0.5), weights=mod_values[i])[0]

    # Normalize q values for number of contributing elements
    qout /= self.qnorm

    # Return arrays for first and second regions
    return qout[...,:len(self.qlist_discrete)], qout[...,len(self.qlist_discrete):]

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

    self.active = True
    return True
