import numpy as np
from PIL import Image

c = []
for alg in ["bare","de","ga","gwo","jaya","pso","ro"]:
    for m in [0,1,2,3,4]:
        fname = "results/map_0%d_towers0_20_300_%s_pcg64/coverage.png" % (m,alg)
        im = np.array(Image.open(fname).convert("L"))
        c.append(im)

off = 10
img = np.zeros((7*(80+off), 5*(80+off)), dtype="uint8")
k = 0
for i in range(7):
    for j in range(5):
        img[(i*(80+off)):(i*(80+off)+80),(j*(80+off)):(j*(80+off)+80)] = c[k]
        k += 1

Image.fromarray(img).save("go_results_plot.png")

