#
#  file:  brute_primes.py
#
#  Brute-force prime testing
#
#  RTK, 04-Sep-2022
#  Last update:  12-Sep-2022
#
################################################################

import sys
import time
from math import sqrt

if (len(sys.argv) == 1):
    print()
    print("brute_prime <n>")
    print()
    print("  <n> - the number to test")
    print()
    exit(0)

n = int(sys.argv[1])

#  Sanity checks
if (n == 2):
    print("2 is prime")
    exit(0)
elif (n%2 == 0):
    print("%d is composite" % n)
    exit(0)

s = time.time()
prime = True
limit = int(sqrt(n))
for i in range(3,limit):
    if (n%i == 0):
        prime = False
        break
e = time.time()

if (prime):
    print("%d is prime (%0.8f)" % (n, e-s))
else:
    print("%d is composite (%0.8f)" % (n, e-s))

