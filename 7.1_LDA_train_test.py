# %%
from utils.common import DIR
from utils.processing_train_test import processing_train_test

# %%

processing_train_test(no_X=True, fimo=True,
                      dir_type=DIR.LDA_TRAIN_TEST, topic_range=range(100, 3400, 100))
