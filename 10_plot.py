# %%
from sklearn import svm
from utils.spliting_train_test import get_train_test_set
from pathlib import Path
import pandas as pd
from utils.utils import get_file_name
import matplotlib.pyplot as plt

# %%
X_train, X_test, y_train, y_test = get_train_test_set(
    is_fimo=True, is_cleaned=True)

kernels = ['linear', 'poly', 'rbf', 'sigmoid']
gammas = ['scale', 'auto', 0.1, 1.0, 1.0, 10.0]
Cs = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

# %%

plt.figure(figsize=(10, 20))

results1 = []
for Cs_ in Cs:
    clf = svm.SVC(kernel='linear', C=Cs_)
    clf.fit(X_train, y_train.values.ravel())
    results1.append(clf.score(X_test, y_test.values.ravel()))

plt.subplot(321)
plt.plot(Cs, results1)
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.xscale('log')
plt.title('Kernel: Linear')

results2 = []
for Cs_ in Cs:
    clf = svm.SVC(C=Cs_)
    clf.fit(X_train, y_train.values.ravel())
    results2.append(clf.score(X_test, y_test.values.ravel()))

plt.subplot(322)
plt.plot(Cs, results2)
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title('Kernel: RBF, Gamma: Auto')
plt.xscale('log')

results3 = []
for gamma_ in gammas:
    clf = svm.SVC(gamma=gamma_)
    clf.fit(X_train, y_train.values.ravel())
    results3.append(clf.score(X_test, y_test.values.ravel()))

plt.subplot(323)
plt.plot(gammas, results3)
plt.xlabel('Gamma')
plt.ylabel('Accuracy')
plt.title('Kernel: RBF, C: 1.0')

results4 = []
for gamma_ in gammas:
    clf = svm.SVC(gamma=gamma_, C=10.0)
    clf.fit(X_train, y_train.values.ravel())
    results4.append(clf.score(X_test, y_test.values.ravel()))

plt.subplot(324)
plt.plot(gammas, results4)
plt.xlabel('Gamma')
plt.ylabel('Accuracy')
plt.title('Kernel: RBF, C: 10.0')

results4 = []
for gamma_ in gammas:
    clf = svm.SVC(gamma=gamma_, kernel='poly')
    clf.fit(X_train, y_train.values.ravel())
    results4.append(clf.score(X_test, y_test.values.ravel()))

plt.subplot(325)
plt.plot(gammas, results4)
plt.xlabel('Gamma')
plt.ylabel('Accuracy')
plt.title('Kernel: Poly, C: 1.0')

results4 = []
for gamma_ in gammas:
    clf = svm.SVC(gamma=gamma_, C=10.0, kernel='poly')
    clf.fit(X_train, y_train.values.ravel())
    results4.append(clf.score(X_test, y_test.values.ravel()))

plt.subplot(326)
plt.plot(gammas, results4)
plt.xlabel('Gamma')
plt.ylabel('Accuracy')
plt.title('Kernel: Poly, C: 10.0')


plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)
plt.show()
# %%
