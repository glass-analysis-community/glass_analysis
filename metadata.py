import pydcd
import sys

# Simple script to print metadata for a list of dcd files

for filename in sys.argv[1:]:
  # The file object can be discarded after converting it to a dcd_file,
  # as the dcd_file duplicates the underlying file descriptor.
  file = open(filename, "r")
  dcd_file = pydcd.dcdfile(file)
  file.close()

  print("%s:" %filename)
  print("    nset: %d" %dcd_file.nset)
  print("    N: %d" %dcd_file.N)
  print("    timestep: %f" %dcd_file.timestep)
  print("    tbsave: %d" %dcd_file.tbsave)
  print("    itstart: %d" %dcd_file.itstart)
  print("    wcell: %d" %dcd_file.wcell)
