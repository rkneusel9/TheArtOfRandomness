import numpy as np
import sys
from PIL import Image

algs = ["bare","de","ga","gwo","jaya","pso","ro"]
src = "output/"

if (len(sys.argv) == 1):
    print()
    print("merge_images <src> <dst>")
    print()
    print("  <src> - source image name (no extension)")
    print("  <dst> - output image (w/extension)")
    print()
    exit(0)

iname = sys.argv[1]
oname = sys.argv[2]

imgs = [np.array(Image.open("%s%s_bare/original.png" % (src,iname)))]
for alg in algs:
    imgs.append(np.array(Image.open("%s%s_%s/enhanced.png" % (src,iname,alg))))

rows, cols = imgs[0].shape
off = 10
image = np.zeros((2*(rows+off),4*(cols+off)), dtype="uint8")
k = 0
for i in range(2):
    for j in range(4):
        image[i*(rows+off):i*(rows+off)+rows,j*(cols+off):j*(cols+off)+cols] = imgs[k]
        k += 1

Image.fromarray(image).save(oname)

