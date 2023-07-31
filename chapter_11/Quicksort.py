#
#  file:  Quicksort.py
#
#  Nonrandom and random implementations of Quicksort
#
#  RTK, 13-Sep-2022
#  Last update: 13-Sep-2022
#
################################################################
import sys
sys.setrecursionlimit(30000)
import numpy as np

################################################################
#  QuicksortRandom
#
def QuicksortRandom(arr):
    """Use random pivot selection"""

    if (len(arr) < 2):
        return arr
    pivot = arr[np.random.randint(0, len(arr))]
    low  = arr[np.where(arr < pivot)]
    same = arr[np.where(arr == pivot)]
    high = arr[np.where(arr > pivot)]
    return np.hstack((QuicksortRandom(low), same, QuicksortRandom(high)))


################################################################
#  Quicksort
#
def Quicksort(arr):
    """Use deterministic pivot selection"""

    if (len(arr) < 2):
        return arr
    pivot = arr[0]
    low  = arr[np.where(arr < pivot)]
    same = arr[np.where(arr == pivot)]
    high = arr[np.where(arr > pivot)]
    return np.hstack((Quicksort(low), same, Quicksort(high)))

