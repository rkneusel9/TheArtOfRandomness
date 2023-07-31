#
#  file:  process_images.py
#
#  RTK, 31-May-2022
#  Last update:  31-May-2022
#
################################################################

import os

algs = ["bare","de","ga","gwo","jaya","pso","ro"]
images = ["apples","barbara","boat","cameraman","fruits","goldhill","lena","peppers","zelda"]

npart = 10
niter = 75
base = "output/"
os.system("mkdir output")

for alg in algs:
    for image in images:
        cmd = "python3 enhance.py images/%s.png %d %d %s pcg64 %s%s_%s" % (image, npart, niter, alg, base, image, alg)
        os.system(cmd)

