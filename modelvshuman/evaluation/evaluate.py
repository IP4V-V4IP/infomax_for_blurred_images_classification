"""
Generic evaluation functionality: evaluate on several datasets.
"""

import csv
import os
import constants as c
from os.path import join as pjoin


class ResultPrinter:

    def __init__(self, model_name, dataset, data_parent_dir=c.RESULTS_DIR):

        self.model_name = model_name
        self.dataset = dataset
        self.data_dir = pjoin(data_parent_dir, dataset.name, 'accuracy')
        self.decision_mapping = self.dataset.decision_mapping
        self.info_mapping = self.dataset.info_mapping
        self.create_csv()

    def create_csv(self):
        self.csv_file_path = os.path.join(self.data_dir, self.model_name.replace('_', '-') + '.csv')

        if os.path.exists(self.csv_file_path):
            # print("Warning: the following file will be overwritten: "+self.csv_file_path)
            os.remove(self.csv_file_path)

        directory = os.path.dirname(self.csv_file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # write csv file header row
        with open(self.csv_file_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['subj', 'object_response', 'category', 'condition', 'imagename'])

    def print_batch_to_csv(self, object_response, batch_targets, paths):

        for response, target, path in zip(object_response, batch_targets, paths):

            img_name, condition, category = self.info_mapping(path)

            with open(self.csv_file_path, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([self.model_name, response[0], category, condition, img_name])
