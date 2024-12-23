import os


class Dataset(object):
    """Base Dataset class

    Attributes:
        name (str): name of the dataset
        params (object): Dataclass object contains following attributes path, image_size, metric, decision_mapping and
                        experiments
        loader (pytorch loader): Data loader
        args (dict): Other arguments

    """

    def __init__(self, name, params, loader, *args, **kwargs):

        self.name = name
        self.image_size = params.image_size
        self.decision_mapping = params.decision_mapping
        self.info_mapping = params.info_mapping
        self.experiments = params.experiments
        self.metrics = params.metrics
        self.args = args
        self.kwargs = kwargs

        resize = False if params.image_size == 224 else True
        self.path = params.path
        assert os.path.exists(self.path), f'dataset {self.name} path not found: ' + self.path

        if self.experiments:
            for e in self.experiments:
                e.name = self.name

        self._loader = None  # this will be lazy-loaded the first time self.loader (the dataloader instance) is called
        self._loader_callback = lambda: loader()(self.path, resize=resize, batch_size=self.kwargs['batch_size'],
                                                 num_workers=self.kwargs['num_workers'], info_mapping=self.info_mapping)

    @property
    def loader(self):
        if self._loader is None:
            self._loader = self._loader_callback()
        return self._loader

    @loader.setter
    def loader(self, new_loader):
        self._loader = new_loader
