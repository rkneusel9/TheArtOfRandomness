import numpy as np
import matplotlib.pylab as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

xtrn = np.load("iris_train_data.npy")
ytrn = np.load("iris_train_labels.npy")
xtst = np.load("iris_test_data.npy")
ytst = np.load("iris_test_labels.npy")

np.random.seed(73939133)
dc = DecisionTreeClassifier(max_depth=3)
dc.fit(xtrn,ytrn)
pred = dc.predict(xtst)
cm = np.zeros((3,3), dtype="uint8")
for i in range(len(pred)):
    cm[ytst[i],pred[i]] += 1
acc = np.diag(cm).sum() / cm.sum()
print(cm)
print(acc)

plot_tree(dc, fontsize=7)
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig("decision_tree.png", dpi=300)
plt.savefig("decision_tree.eps", dpi=300)
plt.close()

i0 = np.where(ytrn == 0)[0]
i1 = np.where(ytrn == 1)[0]
i2 = np.where(ytrn == 2)[0]
plt.plot(xtrn[i0,0], xtrn[i0,1], marker='o', color='k', linestyle='none', fillstyle='none')
plt.plot(xtrn[i1,0], xtrn[i1,1], marker='+', color='k', linestyle='none', fillstyle='none')
plt.plot(xtrn[i2,0], xtrn[i2,1], marker='^', color='k', linestyle='none', fillstyle='none')
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig("decision_tree_features.png", dpi=300)
plt.savefig("decision_tree_features.eps", dpi=300)
plt.close()

