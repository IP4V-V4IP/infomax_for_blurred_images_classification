from abc import ABC


class ImagePathToInformationMapping(ABC):
    def __init__(self):
        pass

    def __call__(self, full_path):
        pass


class InfoMapping(ImagePathToInformationMapping):
    """
        Directory/filename structure:
        .../{something}_{something}_{something}_{condition}_{category}_{img_name}
    """

    def __call__(self, full_path):
        img_name = full_path.split('/')[-1]
        condition = img_name.split('_')[3]
        category = img_name.split('_')[4]

        return img_name, condition, category
