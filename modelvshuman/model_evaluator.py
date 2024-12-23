import logging
import os
import torch
from tqdm import tqdm
from .evaluation import evaluate as e
from .models.wrappers.pytorch import device as py_device
from .utils import load_dataset, load_model

logger = logging.getLogger(__name__)
MAX_NUM_MODELS_IN_CACHE = 3


def device():
    return py_device()


class ModelEvaluator:

    def _pytorch_evaluator(self, model_name, model, dataset, *args, **kwargs):
        """
        Evaluate Model on the given dataset and return the accuracy.
        Args:
            model_name:
            model:
            dataset:
            *args:
            **kwargs:
        """

        logging_info = f'Evaluating model {model_name} on dataset {dataset.name}.'
        logger.info(logging_info)
        print(logging_info)
        for metric in dataset.metrics:
            metric.reset()
        with torch.no_grad():
            result_writer = e.ResultPrinter(model_name=model_name, dataset=dataset)

            for images, target, paths in tqdm(dataset.loader):
                images = images.to(device())
                logits = model.forward_batch(images)
                softmax_output = model.softmax(logits)
                if isinstance(target, torch.Tensor):
                    batch_targets = model.to_numpy(target)
                else:
                    batch_targets = target
                predictions = dataset.decision_mapping(softmax_output)
                for metric in dataset.metrics:
                    metric.update(predictions, batch_targets, paths)
                if kwargs['print_predictions']:
                    result_writer.print_batch_to_csv(object_response=predictions, batch_targets=batch_targets,
                                                     paths=paths)

    def _get_datasets(self, dataset_names, *args, **kwargs):
        dataset_list = []
        for dataset in dataset_names:
            dataset = load_dataset(dataset, *args, **kwargs)
            dataset_list.append(dataset)
        return dataset_list

    def _remove_model_from_cache(self, framework, model_name):

        def _format_name(name):
            return name.lower().replace('-', '_')

        try:
            if framework == 'pytorch':
                cachedir = '/root/.cache/torch/checkpoints/'
                downloaded_models = os.listdir(cachedir)
                for dm in downloaded_models:
                    if _format_name(dm).startswith(_format_name(model_name)):
                        os.remove(os.path.join(cachedir, dm))
        except:
            pass

    def __call__(self, models, dataset_names, *args, **kwargs):
        """
        Wrapper call to _evaluate function.

        Args:
            models:
            dataset_names:
            *args:
            **kwargs:

        Returns:

        """
        logging.info('Model evaluation.')
        _datasets = self._get_datasets(dataset_names, *args, **kwargs)
        for model_name in models:
            datasets = _datasets
            model, framework = load_model(model_name, *args)
            evaluator = self._pytorch_evaluator
            logger.info(f'Loaded model: {model_name}')
            for dataset in datasets:
                evaluator(model_name, model, dataset, *args, **kwargs)
                for metric in dataset.metrics:
                    logger.info(str(metric))
                    print(metric)

            if len(models) >= MAX_NUM_MODELS_IN_CACHE:
                self._remove_model_from_cache(framework, model_name)

        logger.info('Finished evaluation.')
