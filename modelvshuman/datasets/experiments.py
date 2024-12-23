from dataclasses import dataclass, field
from typing import List


@dataclass
class Experiment:
    """
    Experiment parameters
    """
    xlabel: str = 'Condition'
    data_conditions: List = field(default_factory=list)


blur_experiment = Experiment(data_conditions=['0', '1', '3', '5', '7', '10', '15'], xlabel='Filter standard deviation')


@dataclass
class DatasetExperiments:
    name: str
    experiments: [Experiment]


def get_experiments(dataset_names):
    datasets = []
    for name in dataset_names:
        name_for_experiment = name.replace('-', '_')
        if f'{name_for_experiment}_experiment' in globals():
            experiments = eval(f'{name_for_experiment}_experiment')
            experiments.name = name
            datasets.append(DatasetExperiments(name=name, experiments=[experiments]))
        else:
            datasets.append(DatasetExperiments(name=name, experiments=[]))
    return datasets
