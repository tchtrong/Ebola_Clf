# %%
from sklearn.decomposition import LatentDirichletAllocation as lda
from utils.spliting_train_test import get_train_test_set
from sklearn import svm


# %%
X_train, X_test, y_train, y_test = get_train_test_set(
    is_fimo=True, is_cleaned=True)

# %%
score_max = [0, 0]
for i in range(10, 2540, 10):
    lda_model = lda(n_components=i, random_state=43, n_jobs=4)
    lda_model.fit(X_train)
    X_train_LDA = lda_model.transform(X_train)
    X_test_LDA = lda_model.transform(X_test)
    clf = svm.SVC()
    clf.fit(X_train_LDA, y_train.values.ravel())
    score = clf.score(X_test_LDA, y_test.values.ravel())
    if score > score_max[0]:
        score_max[0] = score
        score_max[1] = i
