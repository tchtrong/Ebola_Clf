# %%
from sklearn.decomposition import LatentDirichletAllocation as lda
from utils.spliting_train_test import get_train_test_set
from sklearn import svm
from joblib import load
from matplotlib import pyplot as plt

# %%
X_train, X_test, y_train, y_test = get_train_test_set(
    no_X=True, fimo=True)

results_linear = []
results_rbf = []
for i in range(10, 2540, 100):
    lda_model: lda = load('LDA_models/LDA_{}_train_test'.format(i))
    X_LDA_train = lda_model.transform(X_train)
    X_LDA_test = lda_model.transform(X_test)
    clf_linear = svm.SVC(kernel='linear')
    clf_rbf = svm.SVC()
    clf_linear.fit(X_LDA_train, y_train.values.ravel())
    clf_rbf.fit(X_LDA_train, y_train.values.ravel())
    results_linear.append(clf_linear.score(X_LDA_test, y_test.values.ravel()))
    results_rbf.append(clf_rbf.score(X_LDA_test, y_test.values.ravel()))
# %%
plt.figure(figsize=(10, 5))

plt.subplot(121)
plt.plot(list(range(10, 2540, 100)), results_linear)
plt.xlabel('Dimensions')
plt.ylabel('Accuracy')
plt.title('LDA on train + test, SVM with linear kernel')

plt.subplot(122)
plt.plot(list(range(10, 2540, 100)), results_rbf)
plt.xlabel('Dimensions')
plt.ylabel('Accuracy')
plt.title('LDA on train + test, SVM with RBF kernel')

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)
plt.savefig('LDA_train_test.svg')
plt.show()


# %%
