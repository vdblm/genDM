from genDM.models import test_ratio_model, test_ratio_model_constraint
from genDM.trainer import test_MLE, test_meta_gradient

if __name__ == '__main__':
    test_ratio_model()
    test_ratio_model_constraint()
    test_MLE()
    test_meta_gradient()