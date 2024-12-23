import os
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import constants as c
from PIL import Image
from scipy.optimize import curve_fit
from matplotlib.patches import Rectangle, FancyArrowPatch, Patch
from matplotlib.lines import Line2D


def build_acc_df(data, datasets):

    def accuracy(data):
        acc_dict = {}
        global_acc = (data['object_response'] == data['category']).sum() / len(data)
        acc_dict['global'] = global_acc
        levels = data['condition'].unique()
        for level in levels:
            df = data[data['condition'] == level]
            level_acc = (df['object_response'] == df['category']).sum() / len(df)
            acc_dict[level] = level_acc

        return acc_dict

    df_rows = []
    for dataset in datasets:
        df = data[data['dataset'] == dataset]
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            acc = accuracy(model_df)
            for condition in acc:
                df_rows.append(
                    {'model': model, 'dataset': dataset, 'condition': condition, 'accuracy': acc[condition]})
    df = pd.DataFrame(df_rows)

    return df


def build_accuracy_dataframe(datasets):
    all_datasets = []
    for dataset in datasets:
        files = os.listdir(os.path.join(c.RESULTS_DIR, dataset, 'accuracy'))
        dataset_files = [pd.read_csv(os.path.join(c.RESULTS_DIR, dataset, 'accuracy', file)) for file in files]
        dataset_df = pd.concat(dataset_files)
        dataset_df['dataset'] = dataset
        all_datasets.append(dataset_df)

    all_datasets = pd.concat(all_datasets)
    all_datasets.rename(columns={'subj': 'model'}, inplace=True)
    all_datasets.rename(columns={'imagename': 'image_id'}, inplace=True)

    return all_datasets


def plot_accuracy_gap(dataset, models):
    df = build_accuracy_dataframe(datasets=c.DATASETS)
    plot_base = dataset != 'blurred'
    subjects_df = df[df['model'].str.startswith('subject')]
    subjects_df = subjects_df[subjects_df['condition'] != 40]
    models_df = df[~df['model'].astype(str).str.startswith('subject')].sort_values(by='model')
    if plot_base:
        base_df = models_df[models_df['dataset'] == 'blurred']
        base_df = base_df[base_df['model'].isin(models)]
        plot_base_df = build_acc_df(base_df, ['blurred'])
        plot_base_df = plot_base_df[plot_base_df['condition'] != 'global']
        plot_base_df = plot_base_df.drop(columns=['model'])
        plot_base_df = plot_base_df.groupby(['condition', 'dataset']).mean().reset_index()

    models_df = models_df[models_df['dataset'] == dataset]
    models_df = models_df[models_df['model'].isin(models)]
    plot_models_df = build_acc_df(models_df, [dataset])
    plot_models_df = plot_models_df[plot_models_df['condition'] != 'global']
    plot_subjects_df = build_acc_df(subjects_df, datasets=['blurred'])
    plot_subjects_df = plot_subjects_df[plot_subjects_df['condition'] != 'global']

    plot_models_df = plot_models_df.drop(columns=['model'])
    plot_subjects_df = plot_subjects_df.drop(columns=['model'])
    plot_subjects_df['dataset'] = 'humans'
    plot_models_df = pd.concat([plot_models_df, plot_subjects_df])
    # group by dataset and condition and calculate the metric's mean
    plot_models_df = plot_models_df.groupby(['condition', 'dataset']).mean().reset_index()
    conditions = sorted(plot_models_df['condition'].unique())
    for condition in conditions:
        plot_models_df.loc[plot_models_df['condition'] == condition, 'condition'] = c.CONDITIONS.index(condition)
        if plot_base:
            plot_base_df.loc[plot_base_df['condition'] == condition, 'condition'] = c.CONDITIONS.index(condition)

    sns.set_theme(style='whitegrid')
    fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
    df1 = plot_models_df[plot_models_df['dataset'] == dataset]
    df2 = plot_models_df[plot_models_df['dataset'] == 'humans']

    if plot_base:
        ax.plot(plot_base_df['condition'], plot_base_df['accuracy'], 'o--', label='blurred', color='gray',
                alpha=0.5)

    if dataset == 'blurred':
        ax.plot(df1['condition'], df1['accuracy'], 'o--', color='gray', alpha=1, label=dataset)
    else:
        ax.plot(df1['condition'], df1['accuracy'], 'o-', color=c.PALETTE['deblurred_erco'], alpha=0.75,
                label=dataset)
    ax.plot(df2['condition'], df2['accuracy'], 'o-', color=c.PALETTE['human'], label='humans')

    ax.fill_between(df1['condition'], df1['accuracy'], df2['accuracy'], hatch='/',
                    where=(df1['accuracy'].values > df2['accuracy'].values),
                    color='#40B0A6', alpha=0.3, interpolate=True)
    ax.fill_between(df1['condition'], df1['accuracy'], df2['accuracy'], hatch='|',
                    where=(df1['accuracy'].values <= df2['accuracy'].values),
                    color='#E1BE6A', alpha=0.3, interpolate=True)

    y_label = 'accuracy'
    title = f'Accuracy gap on {len(models)} pre-trained classification models'
    ax.set_ylim(0.0, 1.05)

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=[c.DATASETS_DICT[l] for l in labels], loc='best', fontsize=14)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel('blur level', fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.tick_params(labelsize=14)

    ax.yaxis.grid(linestyle='-', alpha=0.5, color='grey', linewidth=0.25, zorder=-1)
    ax.xaxis.grid(linestyle='-', alpha=0.5, color='grey', linewidth=0.25, zorder=-1)
    plt.tight_layout()
    plt.show()

    fig.savefig(os.path.join(c.PLOTS_DIR, f'gap_acc_{dataset}_{len(models)}_models.png'), dpi=200, bbox_inches='tight')


def plot_accuracy_boxplot(datasets, models, reference):

    df = build_accuracy_dataframe(datasets=datasets)
    subjects_df = df[df['model'].str.startswith('subject')]
    subjects_df = subjects_df[subjects_df['condition'] != 40]
    models_df = df[~df['model'].astype(str).str.startswith('subject')].sort_values(by='model')
    models_df = models_df[models_df['dataset'].isin(datasets)]
    if models is not None:
        models_df = models_df[models_df['model'].isin(models)]
    plot_models_df = build_acc_df(models_df, datasets)
    models_acc_ref = plot_models_df[plot_models_df['dataset'] == reference]
    plot_subjects_df = build_acc_df(subjects_df, datasets=['blurred'])

    if reference is not None:
        datasets = [item for item in datasets if item != reference]
        for condition in c.CONDITIONS:
            for dataset in datasets:
                for model in models:
                    plot_models_df.loc[(plot_models_df['condition'] == condition) &
                                       (plot_models_df['dataset'] == dataset) &
                                       (plot_models_df['model'] == model), 'accuracy'] = (
                            plot_models_df.loc[(plot_models_df['condition'] == condition) &
                                               (plot_models_df['dataset'] == dataset) &
                                               (plot_models_df['model'] == model), 'accuracy'].values -
                            plot_models_df.loc[(plot_models_df['condition'] == condition) &
                                               (plot_models_df['dataset'] == reference) &
                                               (plot_models_df['model'] == model), 'accuracy'].values)
        plot_models_df = plot_models_df[plot_models_df['dataset'] != reference]
        y_label = 'accuracy gain'
        title = f'Accuracy gain of deblurring methods on {len(models)} pre-trained classification models'
    else:
        y_label = 'accuracy'
        title = f'Accuracy of {len(models)} pre-trained classification models'

    plot_models_df = plot_models_df.drop(columns=['model'])
    plot_subjects_df = plot_subjects_df.drop(columns=['model'])
    plot_subjects_df['dataset'] = 'human'

    sns.set_theme(style='whitegrid')
    fig, ax = plt.subplots(figsize=(12, 5), dpi=200)
    hue_order = datasets

    sns.boxplot(data=plot_models_df, x='condition', y='accuracy', hue='dataset', hue_order=hue_order, ax=ax,
                palette=c.PALETTE,
                order=c.CONDITIONS, dodge=True, linewidth=0.5, fliersize=1.0,
                width=0.75,
                showfliers=False,
                zorder=2
                )
    for condition in c.CONDITIONS:
        subj_acc = plot_subjects_df[plot_subjects_df['condition'] == condition]['accuracy'].values
        base_acc = models_acc_ref[models_acc_ref['condition'] == condition]['accuracy'].values.mean()
        ax.boxplot(subj_acc - base_acc, positions=[c.CONDITIONS.index(condition)],
                   widths=0.75, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor=c.PALETTE['human'], color=c.PALETTE['human'], alpha=0.5, linewidth=0.5,
                                 edgecolor='gray', linestyle='-', hatch='//', fill=True),
                   whiskerprops=dict(color='gray', linewidth=0.5, linestyle='-'),
                   capprops=dict(color='gray', linewidth=0.5, linestyle='-'),
                   medianprops=dict(color='gray', linewidth=0.5, linestyle='-'),
                   zorder=1)

    for i in range(len(c.CONDITIONS)):
        ax.add_patch(Rectangle((i - 0.40, -1), 0.8, 2, alpha=0.1, color='gray', zorder=-1))

    ax.set_ylim(-0.15, 0.38)

    ax.set_xticks(range(len(c.CONDITIONS)))
    ax.set_xticklabels(c.CONDITIONS)

    handles, labels = ax.get_legend_handles_labels()
    labels = [c.DATASETS_DICT[dataset] for dataset in datasets]
    handles = [Patch(facecolor=c.PALETTE['human'], edgecolor='gray', alpha=0.5, linewidth=0.5,
                     linestyle='-', hatch='//', label='human')] + handles
    labels = ['Human'] + labels

    ax.legend(handles=handles, labels=labels, loc='upper left', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('blur level', fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)
    ax.tick_params(labelsize=14)

    ax.hlines(0.0, -0.5, 7.5, color='black', linestyle='--', alpha=0.5, zorder=-1)
    ax.yaxis.grid(linestyle='-', alpha=0.5, color='grey', linewidth=0.25, zorder=-1)
    ax.xaxis.grid(linestyle='-', alpha=0.5, color='grey', linewidth=0.25, zorder=-1)
    plt.tight_layout()
    plt.show()

    # save plot
    fig.savefig(os.path.join(c.PLOTS_DIR, f'acc_boxplot_{len(models)}_models.png'), dpi=200, bbox_inches='tight')


def plot_pocc(datasets=c.DATASETS, models=c.BEST_MODELS):

    def power_law_function(x, a):
        return x ** (1 / a)

    # Load data
    df = build_accuracy_dataframe(datasets=datasets)
    df = df[~df['model'].astype(str).str.startswith('subject')]
    df = df[df['dataset'].isin(datasets)]
    df = df[df['model'].isin(models)]

    df['score'] = (df['object_response'] == df['category']).astype(int)
    df.drop(columns=['object_response', 'category', 'model'], inplace=True)

    # average score over image_id and dataset
    df = df.groupby(['image_id', 'dataset']).mean().reset_index()
    if np.all(df['condition'].apply(lambda x: x == int(x))):
        df['condition'] = df['condition'].astype(int)
    co = 4
    for plot_dataset in [d for d in datasets if d != 'blurred']:
        plot_datasets = ['blurred', plot_dataset]
        df_class = df[df['dataset'].isin(plot_datasets)]
        df_plot = df_class.pivot_table(index=['image_id', 'condition'], columns=['dataset'], values='score').reset_index()
        # set condition as integer and order by condition
        df_plot['condition'] = df_plot['condition'].astype(int)
        df_plot = df_plot.sort_values(by='condition').reset_index(drop=True)

        # power law fit
        x = df_plot[plot_datasets[1]].values
        y = df_plot[plot_datasets[0]].values

        a = curve_fit(power_law_function, x, y)[0][0]
        x = np.linspace(0, 1, 100)
        y = power_law_function(x, 1/a)

        # scatter plot of score one dataset vs another
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
        sns.scatterplot(data=df_plot, x=plot_datasets[0], y=plot_datasets[1], ax=ax,
                        s=40, alpha=0.5, edgecolor=None, linewidth=0.0, zorder=1)
        cols = ['Reds', 'gist_yarg', 'copper_r', 'gist_heat_r', 'rocket_r']
        col = cols[co]
        sns.kdeplot(data=df_plot, x=plot_datasets[0], y=plot_datasets[1], ax=ax, fill=True, alpha=0.5, zorder=0,
                    cmap=col)
        # draw diagonal line
        ax.plot([0, 1], [0, 1], 'k-', alpha=1, zorder=2)

        # plot power law fit
        ax.plot(x, y, 'k--', linewidth=3, zorder=2)

        print(f'POCC of {c.DATASETS_DICT[plot_datasets[0]]} vs {c.DATASETS_DICT[plot_datasets[1]]}. '
                      f'Power law fit: y = x^{a:.3f}')

        ax.set_xticks(np.arange(0, 1.01, 0.1))
        ax.set_yticks(np.arange(0, 1.01, 0.1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)

        ax.yaxis.grid(linestyle='-', alpha=0.25, color='grey', linewidth=0.25, zorder=-1)
        ax.xaxis.grid(linestyle='-', alpha=0.25, color='grey', linewidth=0.25, zorder=-1)
        plt.tight_layout()
        plt.show()

        # save plot
        fig.savefig(os.path.join(c.PLOTS_DIR, f'pocc_{plot_datasets[0]}_vs_{plot_datasets[1]}_{len(models)}_models.png'))


def plot_IQA_AUC(nr=False):

    def auc_iqa(df_ranks, datasets):
        aucs = {key: 0 for key in datasets}
        for dataset in datasets:
            auc = 0
            for i in range(1, len(datasets)):
                auc += df_ranks[dataset].iloc[i - 1]
            aucs[dataset] = auc

        aucs = {key: val / (len(datasets) - 1) for key, val in aucs.items()}
        return aucs

    if nr:
        df_iqa = pd.read_csv(os.path.join(c.RESULTS_DIR, 'nriqa.csv')).drop(columns=['image', 'time'])
        metrics_iqa = c.NRIQA_METRICS
    else:
        df_iqa = pd.read_csv(os.path.join(c.RESULTS_DIR, 'friqa.csv')).drop(columns=['image', 'time'])
        metrics_iqa = c.FRIQA_METRICS
    datasets = [d for d in c.DATASETS if d != 'blurred']
    # dataset,image,metric,lower_better,condition,time,score
    df_iqa = df_iqa[df_iqa['dataset'].isin(datasets)]
    conditions = [cond for cond in c.CONDITIONS if cond != 'global' and cond != 0]

    df_dict = {key: [] for key in datasets}
    for metric in metrics_iqa:
        df_m = df_iqa[df_iqa['metric'] == metric]
        lower_better = df_m['lower_better'].iloc[0]
        for condition in conditions:
            df_mc = df_m[df_m['condition'] == condition]
            df_mc = df_mc.drop(columns=['metric', 'lower_better', 'condition']).groupby(['dataset']).mean().reset_index()
            df_mc = df_mc.sort_values(by='score', ascending=lower_better).reset_index(drop=True)

            for dataset in datasets:
                df_dict[dataset].append(df_mc[df_mc['dataset'] == dataset].index.values[0] + 1)

    df_ranks = {}
    for dataset in datasets:
        df_ranks[dataset] = [df_dict[dataset].count(rank) / len(df_dict[dataset]) for rank in range(1, len(datasets) + 1)]
    df_ranks = pd.DataFrame(df_ranks, index=range(1, len(datasets) + 1))
    df_ranks = df_ranks.cumsum(axis=0)
    aucs = auc_iqa(df_ranks, datasets)
    print(aucs)
    df_ranks = df_ranks.melt(var_name='dataset', value_name='cumulative frequency', ignore_index=False)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4), dpi=200)
    sns.lineplot(data=df_ranks, x=df_ranks.index, y='cumulative frequency', hue='dataset', style='dataset', ax=ax,
                 linewidth=3.0, marker='o', markersize=8, style_order=datasets[::-1], palette=c.PALETTE)

    ax.set_title(f'Ranking of methods on {len(metrics_iqa)} {"NR-IQA" if nr else "FR-IQA"} metrics', fontsize=18)
    ax.set_xlabel('Rank', fontsize=17)
    ax.set_ylabel('Cumulative frequency', fontsize=17)
    ax.tick_params(labelsize=14)
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.set_xticks(range(1, len(datasets) + 1))
    ax.set_xticklabels(range(1, len(datasets) + 1), fontsize=17)

    handles, labels = ax.get_legend_handles_labels()
    labels = [f'{c.DATASETS_DICT[dataset]}' for dataset in datasets]
    ax.legend(handles=handles, labels=labels, loc='best', fontsize=14)
    ax.yaxis.grid(linestyle='-', alpha=0.5, color='grey', linewidth=0.25, zorder=-1)
    ax.xaxis.grid(linestyle='-', alpha=0.5, color='grey', linewidth=0.25, zorder=-1)
    plt.tight_layout()
    plt.show()

    fig.savefig(os.path.join(c.PLOTS_DIR, f'auc_{"NR-IQA" if nr else "FR-IQA"}.png'), dpi=200, bbox_inches='tight')


def plot_gain_beta_per_channel_condition():
    beta_values = scipy.io.loadmat(os.path.join(c.RESULTS_DIR, 'betaMat_v4.mat'))['betaMat']
    nbm = scipy.io.loadmat(os.path.join(c.RESULTS_DIR, 'nbM_v4.mat'))['nbM']
    ref = scipy.io.loadmat(os.path.join(c.RESULTS_DIR, 'betaStats_NB_adhoc_2.mat'))
    beta_values_ref = ref['betaMat']
    nbm_ref = ref['nbM']

    nbm_c = nbm.copy()
    nbm_ref_c = nbm_ref.copy()
    conditions = c.CONDITIONS[:-1]
    nbm[nbm == 0] = 40
    np_stats = np.zeros((len(conditions), beta_values.shape[1], 4))
    for i in range(len(conditions)):
        nbm[nbm_c == (i + 1)] = conditions[i]
        nbm_ref[nbm_ref_c == (i + 1)] = conditions[i]

    for i in range(beta_values.shape[1]):
        for j, condition in enumerate(conditions):
            np_stats[j, i, 0] = np.mean(beta_values[np.squeeze(nbm == condition), i])
            np_stats[j, i, 1] = np.std(beta_values[np.squeeze(nbm == condition), i])
            np_stats[j, i, 2] = np.mean(beta_values_ref[np.squeeze(nbm_ref == condition), i])
            np_stats[j, i, 3] = np.std(beta_values_ref[np.squeeze(nbm_ref == condition), i])

    shifts = [0]
    fig, ax = plt.subplots(1, 1, figsize=(12, 7), dpi=200)
    plt.rcParams.update({'text.usetex': True, 'font.family': 'Helvetica'})
    for i in range(np_stats.shape[0]):
        shifts.append(shifts[i] + 6)
        ax.plot(range(np_stats.shape[1]), np_stats[i, :, 0] + shifts[i], 'o-', alpha=1.0,
                color=c.PALETTE['deblurred_erco'], label=conditions[i])
        ax.plot(range(np_stats.shape[1]), np_stats[i, :, 2] + shifts[i], 'o--', alpha=0.7, color='gray')
        ax.fill_between(range(np_stats.shape[1]), np_stats[i, :, 0] - np_stats[i, :, 1] + shifts[i],
                        np_stats[i, :, 0] + np_stats[i, :, 1] + shifts[i], alpha=0.15,
                        color=c.PALETTE['deblurred_erco'])
        ax.fill_between(range(np_stats.shape[1]), np_stats[i, :, 2] - np_stats[i, :, 3] + shifts[i],
                        np_stats[i, :, 2] + np_stats[i, :, 3] + shifts[i], alpha=0.10, color='gray')

        ax.text(0.5, shifts[i] + 1.5, '$\sigma$=', fontsize=20, color='k')
        ax.text(1.18, shifts[i] + 1.5, f'{conditions[i]}', fontsize=14, color='k')
        ax.text(-0.3, shifts[i] + 4.65, '5', fontsize=10, color='k', alpha=0.8)
        ax.text(-0.3, shifts[i] - 0.3565, '0', fontsize=10, color='k', alpha=0.8)

        ax.hlines(shifts[i], -0.2, 14.2, color='k', linestyle='-', alpha=0.5, linewidth=0.5)
        ax.vlines(0.0, shifts[i] - 0.2, shifts[i] + 5.0, color='k', linestyle='-', alpha=0.5, linewidth=0.5)
        for k in range(5):
            ax.hlines(shifts[i] + 1 + k, -0.1, 0.1, color='k', linestyle='-', alpha=0.5, linewidth=0.5)
        for k in range(15):
            ax.vlines(k, shifts[i] - 0.4, shifts[i] + 0.4, color='k', linestyle='-', alpha=0.5, linewidth=0.5)

    ax.set_xlabel('channel', fontsize=14)
    ax.yaxis.set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(range(np_stats.shape[1]))
    ax.set_xticklabels([f'{i + 1}' for i in range(np_stats.shape[1])])

    handles = [Line2D([0], [0], marker='o', color=c.PALETTE['deblurred_erco'],
                      label=c.DATASETS_DICT['deblurred_erco'],
                      linewidth=1.5, markerfacecolor=c.PALETTE['deblurred_erco'], markersize=6),
               Line2D([0], [0], marker='o', color='gray', label='LS optimal',
                      linewidth=1.5, markerfacecolor='gray', markersize=6)]
    ax.legend(handles=handles, loc='upper right', fontsize=12)
    ax.set_title(r'Obtained gains ($\beta$) per channel and blur level ($\sigma$)', fontsize=16)

    plt.tight_layout()
    plt.show()

    fig.savefig(os.path.join(c.PLOTS_DIR, f'gain_beta_per_channel_and_blur.png'), dpi=200, bbox_inches='tight')


def plot_accuracy_gain(datasets, models=c.ALL_MODELS):

    df = build_accuracy_dataframe(datasets=datasets)
    subjects_df = df[df['model'].str.startswith('subject')]
    subjects_df = subjects_df[subjects_df['condition'] != 40]
    models_df = df[~df['model'].astype(str).str.startswith('subject')].sort_values(by='model')
    models_df = models_df[models_df['model'].isin(models)]
    models_df = models_df[models_df['dataset'].isin(datasets)]
    plot_models_df = build_acc_df(models_df, datasets)
    plot_subjects_df = build_acc_df(subjects_df, datasets)

    plot_models_df = plot_models_df.pivot_table(index=['model', 'condition'], columns='dataset',
                                                values='accuracy').reset_index()

    plot_models_df['line_color'] = ['firebrick' if x[1][datasets[0]] > x[1][datasets[1]] else
                                    'green' for x in plot_models_df.iterrows()]

    fig_size = int(2.5 * len(plot_models_df['model'].unique()) // 5) + 1
    fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size - 2), dpi=200)

    condition = 'global'

    plot_subjects_ij_df = plot_subjects_df[plot_subjects_df['condition'] == condition]
    plot_models_ij_df = plot_models_df[plot_models_df['condition'] == condition]
    my_range = range(1, len(plot_models_ij_df.index) + 1)

    # The horizontal plot is made using the hline function
    # ax.vlines(x=plot_subjects_ij_df['accuracy'], ymin=0.5, ymax=len(plot_models_ij_df), color='grey', alpha=0.25)
    min_h_acc = plot_subjects_ij_df['accuracy'].min()
    max_h_acc = plot_subjects_ij_df['accuracy'].max()
    ax.add_patch(Rectangle((min_h_acc, 0.5), max_h_acc - min_h_acc, len(models), alpha=0.2,
                           color=c.PALETTE['human'], zorder=0))

    # ax.vlines(x=1/16, ymin=0.5, ymax=len(plot_models_ij_df), color='black', linestyle='--', alpha=0.75)
    ax.scatter(plot_models_ij_df[datasets[0]], my_range, color='gray', alpha=1, label=datasets[0], zorder=2)
    ax.scatter(plot_models_ij_df[datasets[1]], my_range, color=c.PALETTE['deblurred_erco'], alpha=1,
               label=datasets[1], zorder=2)

    for i in range(len(plot_models_ij_df.index)):
        x1 = plot_models_ij_df[datasets[0]].iloc[i]
        x2 = plot_models_ij_df[datasets[1]].iloc[i]
        y1 = my_range[i]
        y2 = my_range[i]
        if x2 > x1:
            myArrow = FancyArrowPatch(posA=(x1, y1), posB=(x2, y2), arrowstyle='-|>', linestyle='--', color='gray',
                                      mutation_scale=14, shrinkA=1, shrinkB=2, zorder=1)
        else:
            myArrow = FancyArrowPatch(posA=(x1, y1), posB=(x2, y2), arrowstyle='<|-', linestyle='--', color='gray',
                                      mutation_scale=14, shrinkA=1, shrinkB=2, zorder=1)
        ax.add_artist(myArrow)

    handles, labels = ax.get_legend_handles_labels()
    # add rectangle patch for human accuracy to legend
    handles.append(Patch(facecolor=c.PALETTE['human'], edgecolor=c.PALETTE['human'], alpha=0.2, label='human accuracy'))

    labels = [f'{c.DATASETS_DICT[l]}' for l in labels] + ['Human accuracy']
    ax.legend(handles=handles, labels=labels, fontsize=13, loc='lower right',
              # bbox_to_anchor=(0.52, 0.73)
              )

    ax.spines['top'].set_color('gray')
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_color('gray')
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['left'].set_color('gray')
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_color('gray')
    ax.spines['bottom'].set_linewidth(0.5)
    ax.tick_params(axis='x', colors='gray')
    ax.tick_params(axis='y', colors='gray')
    ax.set_yticks(my_range, [s[:20] for s in plot_models_ij_df['model']])
    ax.set_xticks(np.arange(0, 1.01, 0.1))
    ax.set_xticklabels([f'{i:.1f}' for i in np.arange(0, 1.01, 0.1)], color='k')
    ax.set_yticklabels([s[:20] for s in plot_models_ij_df['model']], color='k')
    ax.set_title(f'Average {c.DATASETS_DICT["deblurred_erco"]} accuracy gain for each ANN model',
                 loc='left', x=-0.30, fontsize=16)
    ax.set_xlabel('accuracy', fontsize=16)
    ax.set_ylabel('model', fontsize=16)

    ax.yaxis.grid(linestyle='-', alpha=0.5, color='grey', linewidth=0.25, zorder=-1)
    ax.xaxis.grid(linestyle='-', alpha=0.5, color='grey', linewidth=0.25, zorder=-1)
    ax.tick_params(labelsize=14)

    # Set the limit
    ax.set_xlim(0.45, 0.85)

    plt.tight_layout()
    plt.show()

    # save plot
    fig.savefig(os.path.join(c.PLOTS_DIR, f'accuracy_gain_{datasets[0]}_vs_{datasets[1]}_{len(models)}_models.png'),
                dpi=200, bbox_inches='tight')


def plot_accuracy_images(models=c.ALL_MODELS, positive=True, force_images=False, random_seed=42):
    np.random.seed(random_seed)
    df = build_accuracy_dataframe(datasets=c.DATASETS)
    df = df[~df['model'].astype(str).str.startswith('subject')]
    df = df[df['dataset'].isin(c.DATASETS)]
    df = df[df['model'].isin(models)]
    df['right_guess'] = (df['object_response'] == df['category']).astype(int)
    dft = df.pivot_table(index=['image_id', 'condition', 'model', 'category'],
                         columns=['dataset'], values='right_guess').reset_index()

    competitors = [d for d in c.DATASETS if d != 'blurred' and d != 'deblurred_erco']
    conditions = c.CONDITIONS[1:-1]
    # for each condition, select the images where ERCO is the only one that gets it right
    df_images = pd.DataFrame(columns=['dataset'] + [f'condition_{cond}' for cond in conditions])
    df_responses = pd.DataFrame(columns=['dataset'] + [f'condition_{cond}' for cond in conditions])
    df_categories = pd.DataFrame(columns=['dataset'] + [f'condition_{cond}' for cond in conditions])
    df_model = pd.DataFrame(columns=['dataset'] + [f'condition_{cond}' for cond in conditions])
    df_images['dataset'] = ['original'] + c.DATASETS
    df_responses['dataset'] = ['original'] + c.DATASETS
    df_categories['dataset'] = ['original'] + c.DATASETS
    df_model['dataset'] = ['original'] + c.DATASETS

    for condition in conditions:
        dfc = dft[dft['condition'] == condition]

        if positive:
            dfc = dfc[(dfc['blurred'] == 0) & (dfc['deblurred_erco'] == 1)]
        else:
            dfc = dfc[(dfc['blurred'] == 1) & (dfc['deblurred_erco'] == 0)]
        dfc['competitors'] = dfc[competitors].sum(axis=1)
        dfc.drop(columns=competitors, inplace=True)

        dfc = dfc[dfc['competitors'] == (0 if positive else len(competitors))]

        if force_images:
            image_name = c.CONDITION_IMAGE_MAPPING[condition]
        else:
            image_name = dfc.sample(1)['image_id'].values[0]

        category = dfc[dfc['image_id'] == image_name]['category'].values[0]
        model = dfc[dfc['image_id'] == image_name]['model'].values[0]
        image_id = image_name.split('_')[-2]+'_'+image_name.split('_')[-1]
        original_path = os.path.join(c.DATA_DIR, 'original', image_id)
        column_img = [original_path]
        column_model = [model] * 7
        column_response = [category]
        column_category = [category] * 7
        for dataset in c.DATASETS:
            column_img.append(os.path.join(c.DATA_DIR, dataset, 'dnn', image_name))
            response = df[(df['image_id'] == image_name) & (df['model'] == model) & (df['condition'] == condition) &
                          (df['dataset'] == dataset)]
            column_response.append(response['object_response'].values[0])

        df_images[f'condition_{condition}'] = column_img
        df_responses[f'condition_{condition}'] = column_response
        df_categories[f'condition_{condition}'] = column_category
        df_model[f'condition_{condition}'] = column_model

    fig, axes = plt.subplots(7, 6, figsize=(11, 14), dpi=200)
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        row = i // 6
        col = i % 6
        image = Image.open(df_images.iloc[row, col + 1])
        image = image.convert('RGB')
        ax.imshow(image)
        if col == 0:
            ax.set_ylabel(c.DATASETS_DICT[df_images.iloc[row, 0]], rotation=90, labelpad=10, fontsize=12)
        if row == 0:
            ax.set_title(f'\n{df_model.iloc[0, col + 1]}', fontsize=10)
            ax.text(112, -55, f'blur level {conditions[col]}', fontsize=14,
                    ha='center', va='center', color='black')
            ax.set_xlabel(df_responses.iloc[row, col + 1], fontsize=16)
        else:
            ax.set_xlabel(df_responses.iloc[row, col + 1], fontsize=16)
        ax.set_xticks([])
        ax.set_yticks([])

        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

    plt.subplots_adjust(bottom=0.0, right=0.99, top=0.99, left=0.0)
    plt.tight_layout()
    plt.show()

    if force_images:
        plot_name = f'{"pos" if positive else "neg"}_selection_{len(models)}_models.png'
    else:
        plot_name = f'{"pos" if positive else "neg"}_{len(models)}_models_{random_seed:02d}.png'
    fig.savefig(os.path.join(c.PLOTS_DIR, plot_name), dpi=200, bbox_inches='tight')


if __name__ == '__main__':
    plot_accuracy_gap(dataset='blurred', models=c.ALL_MODELS)
    plot_accuracy_gap(dataset='blurred', models=c.BEST_MODELS)
    plot_accuracy_gap(dataset='deblurred_erco', models=c.BEST_MODELS)
    plot_accuracy_boxplot(datasets=c.DATASETS, models=c.BEST_MODELS, reference='blurred')
    plot_IQA_AUC(nr=True)
    plot_IQA_AUC(nr=False)
    plot_pocc(datasets=c.DATASETS, models=c.BEST_MODELS)
    plot_gain_beta_per_channel_condition()
    plot_accuracy_gain(datasets=['blurred', 'deblurred_erco'], models=c.BEST_MODELS)
    plot_accuracy_images(models=c.BEST_MODELS, positive=True, force_images=True)
