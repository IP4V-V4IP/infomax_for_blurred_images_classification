import glob
import os
import time
import pandas as pd
from pyiqa import create_metric
from tqdm import tqdm
import constants as c


def run_nriqa():

    for dataset in c.DATASETS:

        dataset_path = os.path.join(c.DATA_DIR, dataset, 'dnn')
        image_files = sorted(glob.glob(os.path.join(dataset_path, '*')))
        for metric_name in c.NRIQA_METRICS:
            if metric_name == 'sseq':
                # skip sseq, as it is calculated using a matlab script
                continue
            iqa_model = create_metric(metric_name, metric_mode='NR')
            results = []
            pbar = tqdm(total=len(image_files), unit='image', leave=False)
            for image_path in image_files:

                start_time = time.time()

                if os.path.isfile(image_path):
                    input_paths = [image_path]
                else:
                    input_paths = sorted(glob.glob(os.path.join(dataset_path, '*')))

                avg_score = 0
                condition = -1
                img_name = ''
                test_img_num = len(input_paths)
                for idx, img_path in enumerate(input_paths):
                    img_name = os.path.basename(img_path)
                    condition = int(img_name.split('_')[3])
                    ref_img_path = None

                    score = iqa_model(img_path, ref_img_path).cpu().item()
                    avg_score += score
                    pbar.update(1)
                    pbar.set_description(f'{metric_name:20s} of {img_name:50s}: {score:3.10f}')

                    avg_score /= test_img_num

                el_time = time.time() - start_time
                result = {'dataset': dataset, 'image': img_name[:-4], 'metric': metric_name,
                          'lower_better': iqa_model.lower_better, 'condition': condition, 'time': el_time,
                          'score': avg_score}
                results.append(result)

            df = pd.DataFrame(results)
            df.to_csv(os.path.join(c.RESULTS_DIR, dataset, 'nriqa', f'{metric_name}.csv'), index=False)

            pbar.close()
            test_img_num = len(image_files)
            avg_score = df['score'].mean()
            el_time = df['time'].sum()
            msg = (f'Average {metric_name:20s} score of {dataset:20s} with {test_img_num} images is: {avg_score:3.10f} '
                   f'in {el_time:.2f} seconds')
            print(msg)

    build_nriqa_results()


def build_nriqa_results():
    dfs = []
    for dataset in c.DATASETS:
        for metric in c.NRIQA_METRICS:
            file_name = f'{metric}.csv'
            dfs.append(pd.read_csv(os.path.join(c.RESULTS_DIR, dataset, 'nriqa', file_name)))

    df = pd.concat(dfs)
    df.to_csv(os.path.join(c.RESULTS_DIR, 'nriqa.csv'), index=False)


def run_friqa():

    for dataset in c.DATASETS:

        dataset_path = os.path.join(c.DATA_DIR, dataset, 'dnn')
        image_files = sorted(glob.glob(os.path.join(dataset_path, '*')))
        for metric_name in c.FRIQA_METRICS:
            iqa_model = create_metric(metric_name, metric_mode='FR')
            results = []
            pbar = tqdm(total=len(image_files), unit='image', leave=False)
            faults = 0

            for image_path in image_files:
                start_time = time.time()
                if os.path.isfile(image_path):
                    input_paths = [image_path]
                else:
                    input_paths = sorted(glob.glob(os.path.join(dataset_path, '*')))

                avg_score = 0
                condition = -1
                img_name = ''
                test_img_num = len(input_paths)
                fault = False
                for idx, img_path in enumerate(input_paths):
                    img_name = os.path.basename(img_path)
                    condition = int(img_name.split('_')[3])
                    ref_img_name = image_path.split('/')[-1]
                    ref_name = ref_img_name.split('_')[-2] + '_' + image_path.split('_')[-1]
                    ref_img_path = os.path.join(c.DATA_DIR, 'original', ref_name)
                    score = iqa_model(img_path, ref_img_path).cpu().item()
                    avg_score += score
                    pbar.update(1)
                    pbar.set_description(f'{metric_name:20s} of {img_name:50s}: {score:3.10f}')

                    avg_score /= test_img_num

                el_time = time.time() - start_time
                if fault:
                    faults += 1
                else:
                    result = {'dataset': dataset, 'image': img_name[:-4], 'metric': metric_name,
                              'lower_better': iqa_model.lower_better, 'condition': condition, 'time': el_time,
                              'score': avg_score}
                    results.append(result)

            pbar.close()
            df = pd.DataFrame(results)
            df.to_csv(os.path.join(c.RESULTS_DIR, dataset, 'friqa', f'{metric_name}.csv'), index=False)
            test_img_num = len(image_files)
            avg_score = df['score'].mean()
            el_time = df['time'].sum()
            msg = (f'Average {metric_name:20s} score of {dataset:20s} with {test_img_num} images is: {avg_score:3.10f} '
                   f'in {el_time:.2f} seconds')
            print(msg)

    build_friqa_results()


def build_friqa_results():
    dfs = []
    for dataset in c.DATASETS:
        for metric in c.FRIQA_METRICS:
            file_name = f'{metric}.csv'
            dfs.append(pd.read_csv(os.path.join(c.RESULTS_DIR, dataset, 'friqa', file_name)))

    df = pd.concat(dfs)
    df.to_csv(os.path.join(c.RESULTS_DIR, 'friqa.csv'), index=False)


if __name__ == '__main__':
    # run_nriqa()
    # run_friqa()
    build_nriqa_results()
    build_friqa_results()
