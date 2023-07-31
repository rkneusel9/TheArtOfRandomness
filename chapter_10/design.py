#
#  file:  design.py
#
#  Experimental design simulator
#
#  RTK, 30-Jul-2022
#  Last update:  12-Aug-2022
#
################################################################

import numpy as np
import sys
import os
from RE import *
from scipy.stats import ttest_ind
import matplotlib.pylab as plt

#  Parse command line
if (__name__ == "__main__"): 
    if (len(sys.argv) == 1):
        print()
        print("design <npop> <nsubj> <beta> <nsim> <type> <plot> [<kind> | <kind> <seed>]")
        print()
        print("  <npop>  -  population size (e.g. 1000)")
        print("  <nsubj> -  number of subjects in the experiment (e.g. 40)")
        print("  <beta>  -  supplement effect strength [0..1]")
        print("  <nsim>  -  number of simulations to run (e.g. 100)")
        print("  <typ>   -  selection type: 0=simple, 1=block, 2=stratified")
        print("  <plot>  -  1=show plot, 0=no plot")
        print("  <kind>  -  randomness source")
        print("  <seed>  -  seed value")
        print()
        exit(0)

    npop = int(sys.argv[1])
    nsubj = int(sys.argv[2])
    beta = float(sys.argv[3])
    nsimulations = int(sys.argv[4])
    typ = int(sys.argv[5])
    showPlot = int(sys.argv[6])

#
#  Global RE instance
#
if (len(sys.argv) == 9):
    rng = RE(kind=sys.argv[7], seed=int(sys.argv[8]))
elif (len(sys.argv) == 8):
    rng = RE(kind=sys.argv[7])
else:
    rng = RE()


################################################################
#  Devroye's coin flip binomial distribution -- simulate
#  n coin flips with p probability of "heads"
#
def binomial(n,a,rng):
    """Return a sample from n trials and probability a"""
    k = 0
    p = a if (a <= 0.5) else 1.0-a
    for i in range(n):
        if (rng.random() <= p):
            k += 1
    return k if (a <= 0.5) else n-k


################################################################
#  Cohen's d
#
def Cohen_d(a,b):
    """Cohen's d effect size"""
    s1 = np.std(a, ddof=1)**2
    s2 = np.std(b, ddof=1)**2
    return (a.mean() - b.mean()) / np.sqrt(0.5*(s1+s2))


################################################################
#  Person
#
class Person():
    """A person with randomly selected characteristics"""

    def __init__(self):
        """Constructor"""
        self.age = int(3*rng.random())
        self.income = int(3*rng.random())
        self.smoker = 0
        if (rng.random() < 0.2):
            self.smoker = 1
        self.drink = int(3*rng.random())
        self.adj = 2*(rng.random() - 0.5)

    def Stats(self):
        """Report the person's stats"""
        return self.age, self.income, self.smoker, self.drink, self.adj

    def Health(self):
        """Report the person's health score"""
        return 3*(2-self.age) + 2*self.income - 2*self.smoker - self.drink + self.adj

    def Treat(self, beta=0.03):
        """Apply the intervention"""
        self.adj += 3*binomial(300, beta, rng) / 300  # [0,1]


################################################################
#  Simple
#
def Simple(pop, nsubj):
    """Select subjects using simple randomization"""

    order = np.argsort(rng.random(len(pop)))
    c = []; t = []
    for k in range(nsubj):
        if (rng.random() < 0.5):
            c.append(pop[order[k]])
        else:
            t.append(pop[order[k]])
    return c,t


################################################################
#  Block
#
def Block(pop, nsubj):
    """Select subjects using a balanced block design"""

    ns = 4*(nsubj//4)
    blocks = ["1100","1010","1001","0110","0101","0011"]
    nblocks = ns//4
    seq = ""
    for i in range(nblocks):
        n = int(len(blocks)*rng.random())
        seq += blocks[n]
    order = np.argsort(rng.random(len(pop)))
    c = [];  t = []
    for i in range(ns):
        if (seq[i] == "1"):
            t.append(pop[order[i]])
        else:
            c.append(pop[order[i]])
    return c,t


################################################################
#  Stratified
#
def Stratified(pop, nsubj):
    """Select matched cohorts"""

    def match(n,m,pop,selected):
        if (selected[m]):
            return False
        if (pop[n].age != pop[m].age):
            return False
        if (pop[n].income != pop[m].income):
            return False
        if (pop[n].smoker != pop[m].smoker):
            return False
        if (pop[n].drink != pop[m].drink):
            return False
        return True
    
    selected = np.zeros(len(pop), dtype="uint8")
    c = [];  t = []
    while (len(t) < nsubj//2):
        #  Pick a test subject
        n = int(len(pop)*rng.random())
        while (selected[n] == 1):
            n = int(len(pop)*rng.random())
        selected[n] = 1
        t.append(pop[n])

        #  Now find a matched control
        m = int(len(pop)*rng.random())
        while (not match(n,m, pop, selected)):
            m = int(len(pop)*rng.random())
        selected[m] = 1
        c.append(pop[m])
    return c,t


################################################################
#  Summarize
#
def Summarize(subjects):
    """Summarize the collection of subjects"""

    h = []
    age = income = smoker = drink = 0.0
    for subject in subjects:
        h.append(subject.Health())
        age += subject.age
        income += subject.income
        smoker += subject.smoker
        drink += subject.drink
    age /= len(subjects)
    income /= len(subjects)
    smoker /= len(subjects)
    drink /= len(subjects)
    return np.array(h), age, income, smoker, drink


#  For the desired number of simulations
results = []

for nsim in range(nsimulations):
    #  Create a population
    pop = []
    for i in range(npop):
        pop.append(Person())

    #  Select the experimental subjects
    control, treatment = [Simple, Block, Stratified][typ](pop, nsubj)

    #  Apply the treatment to the treatment group
    for subject in treatment:
        subject.Treat(beta)

    #  Evaluate health
    ch, c_age, c_income, c_smoker, c_drink = Summarize(control)
    th, t_age, t_income, t_smoker, t_drink = Summarize(treatment)

    #  Store results
    results.append({
        "c_age": c_age,
        "c_income": c_income,
        "c_smoker": c_smoker,
        "c_drink": c_drink,
        "t_age": t_age,
        "t_income": t_income,
        "t_smoker": t_smoker,
        "t_drink": t_drink,
        "ttest": ttest_ind(th,ch),
        "d": Cohen_d(th,ch),
    })

#  Keep the lowest and highest 10% of t-test results
p = np.array([r["ttest"][1] for r in results])
idx = np.argsort(p)
n = int(0.1*len(p))
m = len(p) - n
low = []
high = []
for i in range(n):
    low.append(results[idx[i]])
    high.append(results[idx[m:][i]])

#  Mean covariate differences
la = []; li = []; ls = []; ld = []
for t in low:
    la.append(np.abs(t["c_age"]-t["t_age"]))
    li.append(np.abs(t["c_income"]-t["t_income"]))
    ls.append(np.abs(t["c_smoker"]-t["t_smoker"]))
    ld.append(np.abs(t["c_drink"]-t["t_drink"]))
la = np.array(la);  li = np.array(li)
ls = np.array(ls);  ld = np.array(ld)

ha = []; hi = []; hs = []; hd = []
for t in high:
    ha.append(np.abs(t["c_age"]-t["t_age"]))
    hi.append(np.abs(t["c_income"]-t["t_income"]))
    hs.append(np.abs(t["c_smoker"]-t["t_smoker"]))
    hd.append(np.abs(t["c_drink"]-t["t_drink"]))
ha = np.array(ha);  hi = np.array(hi)
hs = np.array(hs);  hd = np.array(hd)

#  t-test comparisons
ta,pa = ttest_ind(ha,la)
ti,pi = ttest_ind(hi,li)
ts,ps = ttest_ind(hs,ls)
td,pd = ttest_ind(hd,ld)

#  p-values
plow = np.array([r["ttest"][1] for r in low])
phigh = np.array([r["ttest"][1] for r in high])

#  Cohen's d
dlow = np.array([r["d"] for r in low])
dhigh = np.array([r["d"] for r in high])

#  results
print()
print("mean p-value (lowest) : %0.5f" % plow.mean())
print("mean p-value (highest): %0.5f" % phigh.mean())
print()
print("mean Cohen's (lowest) : %0.5f" % dlow.mean())
print("mean Cohen's (highest): %0.5f" % dhigh.mean())

if (typ != 2):
    print()
    print("delta age   : (high, low, t, p) = (%0.5f, %0.5f, % 0.5f, %0.5f)" % (ha.mean(), la.mean(), ta,pa))
    print("delta income: (high, low, t, p) = (%0.5f, %0.5f, % 0.5f, %0.5f)" % (hi.mean(), li.mean(), ti,pi))
    print("delta smoker: (high, low, t, p) = (%0.5f, %0.5f, % 0.5f, %0.5f)" % (hs.mean(), ls.mean(), ts,ps))
    print("delta drink : (high, low, t, p) = (%0.5f, %0.5f, % 0.5f, %0.5f)" % (hd.mean(), ld.mean(), td,pd))
print()

#  histogram of p-values
if (showPlot):
    p = np.array([r["ttest"][1] for r in results])
    h,x = np.histogram(p, bins=20)
    h = 100.0*(h / h.sum())
    x = (x[1:] + x[:-1])/2
    plt.bar(x,h, width=0.8*(x[1]-x[0]), color='k')
    plt.plot([0.05,0.05],[0,h.max()], color='k', linestyle='dashed', linewidth=1.5)
    plt.xlabel("p-value")
    plt.ylabel("Percent")
    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    plt.savefig("design_typ_%d_npop_%d_nsubj_%d_nsim_%d.png" % (typ,npop,nsubj,nsimulations))
    plt.savefig("design_typ_%d_npop_%d_nsubj_%d_nsim_%d.eps" % (typ,npop,nsubj,nsimulations))
    plt.show()
    plt.close()

