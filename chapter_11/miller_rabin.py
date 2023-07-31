#
#  file:  miller_rabin.py
#
#  Miller-Rabin probabilistic primality test
#
#  RTK, 03-Sep-2022
#  Last update:  12-Sep-2022
#
################################################################

import time
import sys
from RE import *

################################################################
#  MillerRabin
#
def MillerRabin(n, rounds=5):
    """Implementation of Miller-Rabin based on Wikipedia pseudocode"""

    #  Sanity checks
    if (n==2):
        return True
    if (n%2 == 0):
        return False

    #  Write n = d*2**s + 1 by incrementing s and dividing d=n by 2 until d is odd
    s = 0
    d = n-1
    while (d%2 == 0):
        s += 1
        d //= 2

    for k in range(rounds):
        a = int(rng.random())  # failure: n=65, a=8|18|47|57|64 (nonwitness numbers for 65)
        x = pow(a,d,n)  #(a**d) % n
        if (x==1) or (x == n-1):
            continue
        b = False
        for j in range(s-1):
            x = pow(x,2,n)  #x**2 % n
            if (x == n-1):
                b = True
                break
        if (b):
            continue
        return False
    return True

#
#  main
#
if (len(sys.argv) == 1):
    print()
    print("miller_rabin <n> <rounds> [<kind> | <kind> <seed>]")
    print()
    print("  <n>      - number to test")
    print("  <rounds> - number of rounds")
    print("  <kind>   - randomness source")
    print("  <seed>   - seed")
    print()
    exit(0)

n = int(sys.argv[1])
k = int(sys.argv[2])

if (len(sys.argv) == 5):
    rng = RE(kind=sys.argv[3], mode="int", low=1, high=n-1, seed=int(sys.argv[4]))
elif (len(sys.argv) == 4):
    rng = RE(kind=sys.argv[3], mode="int", low=1, high=n-1)
else:
    rng = RE(mode="int", low=1, high=n-1)

ans = MillerRabin(n,k)
if (ans):
    print("%d is probably prime" % n)
else:
    print("%d is composite" % n)

