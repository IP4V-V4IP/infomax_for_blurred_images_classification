import warnings
import constants as c
from modelvshuman import Evaluate
warnings.simplefilter(action='ignore', category=UserWarning)


def run_evaluation():
    models = c.BEST_MODELS
    datasets = c.DATASETS
    params = {'batch_size': 16, 'print_predictions': True, 'num_workers': 10}
    Evaluate()(models, datasets, **params)


if __name__ == "__main__":
    run_evaluation()
