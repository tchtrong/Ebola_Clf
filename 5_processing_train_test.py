# %%
from utils.common import DIR
from utils.processing_train_test import processing_train_test

# %%

processing_train_test(no_X=True, fimo=True, dir_type=DIR.SVM_TRAIN_TEST)

# processing_train_test.create_train_test_folders(DIR.SVM_TRAIN_TEST, no_X=True, fimo=True)
# %%
