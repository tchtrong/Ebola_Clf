# %%
from sklearn import svm
from utils.spliting_train_test import get_train_test_set
import matplotlib.pyplot as plt

# %%
X_train, X_test, y_train, y_test = get_train_test_set(
    is_fimo=True, is_cleaned=True)

kernels = ['linear', 'poly', 'rbf', 'sigmoid']
gammas = ['scale', 'auto', 0.1, 1.0, 1.0, 10.0]
Cs = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

# %%

plt.figure(figsize=(10, 5))

results = []
for Cs_ in Cs:
    clf = svm.SVC(kernel='linear', C=Cs_)
    clf.fit(X_train, y_train.values.ravel())
    results.append(clf.score(X_test, y_test.values.ravel()))

plt.subplot(231)
plt.plot(Cs, results)
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.xscale('log')
plt.title('Kernel: Linear')

results = []
for Cs_ in Cs:
    clf = svm.SVC(C=Cs_)
    clf.fit(X_train, y_train.values.ravel())
    results.append(clf.score(X_test, y_test.values.ravel()))

plt.subplot(232)
plt.plot(Cs, results)
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title('Kernel: RBF, Gamma: Auto')
plt.xscale('log')

results = []
for gamma_ in gammas:
    clf = svm.SVC(gamma=gamma_)
    clf.fit(X_train, y_train.values.ravel())
    results.append(clf.score(X_test, y_test.values.ravel()))

plt.subplot(233)
plt.plot(gammas, results)
plt.xlabel('Gamma')
plt.ylabel('Accuracy')
plt.title('Kernel: RBF, C: 1.0')

results = []
for gamma_ in gammas:
    clf = svm.SVC(gamma=gamma_, C=100.0)
    clf.fit(X_train, y_train.values.ravel())
    results.append(clf.score(X_test, y_test.values.ravel()))

plt.subplot(234)
plt.plot(gammas, results)
plt.xlabel('Gamma')
plt.ylabel('Accuracy')
plt.title('Kernel: RBF, C: 100.0')

results = []
for gamma_ in gammas:
    clf = svm.SVC(gamma=gamma_, kernel='poly')
    clf.fit(X_train, y_train.values.ravel())
    results.append(clf.score(X_test, y_test.values.ravel()))

plt.subplot(235)
plt.plot(gammas, results)
plt.xlabel('Gamma')
plt.ylabel('Accuracy')
plt.title('Kernel: Poly, C: 1.0')

results = []
for gamma_ in gammas:
    clf = svm.SVC(gamma=gamma_, C=10.0, kernel='poly')
    clf.fit(X_train, y_train.values.ravel())
    results.append(clf.score(X_test, y_test.values.ravel()))

plt.subplot(236)
plt.plot(gammas, results)
plt.xlabel('Gamma')
plt.ylabel('Accuracy')
plt.title('Kernel: Poly, C: 10.0')


plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.5,
                    wspace=0.35)

plt.savefig('SVM.svg')
plt.show()
# %%
