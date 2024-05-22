import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def compare_predictions(baseline_preds, new_preds):
    if not baseline_preds or not new_preds:
        return 0
    match_count = sum(1 for b, n in zip(baseline_preds, new_preds) if b['prediction'] == n['prediction'])
    total = len(baseline_preds)
    return 100 * match_count / total if total > 0 else 0

def read_metrics(base_path):
    results = []
    results_agreement = []

    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file == "lora_model_eval_prediction.jsonl":
                predictions = [json.loads(line) for line in open(os.path.join(root, file), 'r') if line.strip()]
                path_parts = os.path.relpath(root, base_path).split(os.sep)
                experiment = path_parts[0]
                model_type_raw = path_parts[1]
                # model_type = int(model_type_raw[9 : -8])
                task = path_parts[2]
                rank_raw = path_parts[3]
                rank_raw_list = rank_raw.split("_")
                rank = int(rank_raw_list[1])
                merge_type = rank_raw_list[3]
                results_agreement.append({
                    "experiment": experiment,
                    "model_type": model_type,
                    "task": task,
                    "rank": rank,
                    "merge_type": merge_type,
                    "predictions": predictions,
                })
                # Later you run compare_predictions(lora_predictions, lola_predictions) on the predictions to get the agreement percentage

            if file == "lola_model_metrics.json":
                metrics_file = os.path.join(root, file)
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                
                path_parts = os.path.relpath(root, base_path).split(os.sep)
                experiment = path_parts[0]
                model_type_raw = path_parts[1]

                # print("path_parts =", path_parts)
                # print("model_type_raw =", model_type_raw)

                useful_string = model_type_raw.split("_")[1]
                model_type = int(useful_string[1:-5])

                # model_type = int(model_type_raw[9 : -8])
                task = path_parts[2]
                rank_raw = path_parts[3]
                rank_raw_list = rank_raw.split("_")
                rank = int(rank_raw_list[1])
                merge_type = rank_raw_list[3]

                # if merge_type == "TIES":
                #     model_type = "TIES"

                results.append({
                    "experiment": experiment,
                    "model_type": model_type,
                    "task": task,
                    "rank": rank,
                    "merge_type": merge_type,
                    "exact_match": metrics["exact_match"],
                    "rouge1": metrics["rouge1"],
                    "rougeL": metrics["rougeL"]
                })

            elif file == "base_model_metrics.json":
                metrics_file = os.path.join(root, file)
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                
                path_parts = os.path.relpath(root, base_path).split(os.sep)
                experiment = path_parts[0]
                # model_type_raw = path_parts[1]
                model_type = "base"
                task = path_parts[2]

                rank = 0
                merge_type = "base"

                results.append({
                    "experiment": experiment,
                    "model_type": model_type,
                    "task": task,
                    "rank": rank,
                    "merge_type": merge_type,
                    "exact_match": metrics["exact_match"],
                    "rouge1": metrics["rouge1"],
                    "rougeL": metrics["rougeL"]
                })
            
            elif file == "lora_model_metrics.json":

                # print("os.path.join(root, file) =", os.path.join(root, file))
                if "TIES" not in os.path.join(root, file):

                    metrics_file = os.path.join(root, file)
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                    
                    path_parts = os.path.relpath(root, base_path).split(os.sep)
                    experiment = path_parts[0]

                    model_type = "lora"
                    task = path_parts[2]

                    rank = 0
                    merge_type = "lora"

                    results.append({
                        "experiment": experiment,
                        "model_type": model_type,
                        "task": task,
                        "rank": rank,
                        "merge_type": merge_type,
                        "exact_match": metrics["exact_match"],
                        "rouge1": metrics["rouge1"],
                        "rougeL": metrics["rougeL"]
                    })
                
                else:
                    metrics_file = os.path.join(root, file)
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                    
                    path_parts = os.path.relpath(root, base_path).split(os.sep)
                    experiment = path_parts[0]

                    model_type = "ties"
                    task = path_parts[2]

                    rank = 0
                    merge_type = "ties"

                    results.append({
                        "experiment": experiment,
                        "model_type": model_type,
                        "task": task,
                        "rank": rank,
                        "merge_type": merge_type,
                        "exact_match": metrics["exact_match"],
                        "rouge1": metrics["rouge1"],
                        "rougeL": metrics["rougeL"]
                    })


    return results


def plot_bar_with_error(data, model_type):
    df = pd.DataFrame(data)

    # Filter the data for the selected model type
    model_df = df[df['model_type'] == model_type]

    # Group by the relevant columns and calculate the mean and std for each group
    aggregated_df = model_df.groupby(['task', 'merge_type', 'rank']).agg(
        exact_match_mean=pd.NamedAgg(column='exact_match', aggfunc='mean'),
        exact_match_std=pd.NamedAgg(column='exact_match', aggfunc='std')
    ).reset_index()

    # Set the plot style
    sns.set(style="whitegrid")

    # Initialize the matplotlib figure
    plt.figure(figsize=(30, 10))

    # Draw the bar plot
    sns.barplot(x="task", y="exact_match_mean", hue="merge_type", data=aggregated_df, ci=None)

    # Add error bars
    for task in aggregated_df['task'].unique():
        for merge_type in aggregated_df['merge_type'].unique():
            subset = aggregated_df[(aggregated_df['task'] == task) & (aggregated_df['merge_type'] == merge_type)]
            x = subset.index
            y = subset['exact_match_mean']
            yerr = subset['exact_match_std']
            plt.errorbar(x, y, yerr=yerr, fmt='none', c='black')

    # Set labels and title
    plt.xlabel('Task')
    plt.ylabel('Exact Match Score')
    plt.title(f'Exact Match Scores by Task and Merge Type for Model Type {model_type}')
    plt.xticks(rotation=90)
    plt.legend(title='Merge Type')

    # Show the plot
    plt.tight_layout()
    plt.show()


"""
Automatically generate the Latex code
"""
model_type_list = [10, 50, 100, 500]
merge_type_list = ["diagonal", "full",]
ranks = [16, 32, 64, 128, 256]

def generate_latex_table(raw_df, tasks, tasks_short):

    metric_to_check = "rougeL" # exact_match   rouge1   rougeL
    # metric_to_check = "rouge1"
    # metric_to_check = "exact_match" 

    compress_matrix_string = "gpu_comp_rate"

    latex_code = r'''
    \begin{tabular}{c|c|c|c|c|c|c|c|c|c|c|c|c|c}
    \toprule
    \multirow{2}{*}{Model Type} & \multirow{2}{*}{Method Type} & \multicolumn{10}{c}{Tasks} & \multirow{2}{*}{Average} & \multirow{2}{*}{Compression} \\
    \cmidrule{3-12}
    & & ''' + ' & '.join(tasks_short) + r''' & \\
    \midrule
    '''

    def get_task_metrics(df, metric):
        metrics = []
        for task in tasks:
            task_df = df[df['task'] == task]
            if not task_df.empty:
                mean_val = task_df[metric].mean()
                std_val = task_df[metric].std()
                metrics.append(f'{mean_val:.2f} \\scriptsize{{$\\pm$ {std_val:.2f}}}')
            else:
                metrics.append('-')
        return metrics

    # Base and LoRA models
    model_types = ["base", "lora"]
    for model_type in model_types:
        filtered_df = raw_df[raw_df["model_type"] == model_type]
        task_metrics = get_task_metrics(filtered_df, metric_to_check)
        average_metric = filtered_df[metric_to_check].mean()
        std_metric = filtered_df[metric_to_check].std()

        comp_rate = filtered_df[compress_matrix_string].mean() 
        latex_code += f'& {model_type} & ' + ' & '.join(task_metrics) + \
            f' & {average_metric:.2f} \\scriptsize{{$\\pm$ {std_metric:.2f}}}' + f'& {comp_rate:.2f} \\\\ \n'

    latex_code += r'\midrule\midrule' + '\n'

    # TIES results
    latex_code += r'\multirow{5}{*}{' + 'TIES}' + '\n'
    for model_type in model_type_list:
        
        for rank in [0]:
            method_type = f'{rank} D'

            filtered_df = raw_df[(raw_df["model_type"] == model_type) & (raw_df["merge_type"] == "TIES") & (raw_df["rank"] == rank)]
            task_metrics = get_task_metrics(filtered_df, metric_to_check)
            average_metric = filtered_df[metric_to_check].mean()
            std_metric = filtered_df[metric_to_check].std()

            comp_rate = filtered_df[compress_matrix_string].mean() 

            latex_code += f'& {model_type} & ' + ' & '.join(task_metrics) + \
                  f' & {average_metric:.2f} \\scriptsize{{$\\pm$ {std_metric:.2f}}}' + f' & {comp_rate:.2f} \\\\ \n'
        # latex_code += r'\midrule' + '\n'

            
    latex_code += r'\midrule' + '\n'
    latex_code += r'\midrule' + '\n'

    # SVD results
    svd_ranks = [2, 4, 8, 16]
    latex_code += r'\multirow{4}{*}{SVD}' + '\n'
    for svd_rank in svd_ranks:
        method_type = f'SVD {svd_rank}'
        filtered_df = raw_df[(raw_df["merge_type"] == "SVD") & (raw_df["rank"] == svd_rank)]
        task_metrics = get_task_metrics(filtered_df, metric_to_check)
        average_metric = filtered_df[metric_to_check].mean()
        std_metric = filtered_df[metric_to_check].std()

        comp_rate = filtered_df[compress_matrix_string].mean() 

        latex_code += f'& {method_type} & ' + ' & '.join(task_metrics) + \
                f' & {average_metric:.2f} \\scriptsize{{$\\pm$ {std_metric:.2f}}} ' +  f' & {comp_rate:.2f} \\\\ \n'
    latex_code += r'\midrule\midrule' + '\n'


    # For each model type, for diagonal and full, for rank in [16, 32, 64, 128, 256]
    for model_type in model_type_list:
        latex_code += r'\multirow{5}{*}{' + f'{model_type}' + ' diagonal (D)}' + '\n'
        for rank in ranks:
            method_type = f'{rank} D'

            filtered_df = raw_df[(raw_df["model_type"] == model_type) & (raw_df["merge_type"] == "diagonal") & (raw_df["rank"] == rank)]
            task_metrics = get_task_metrics(filtered_df, metric_to_check)
            average_metric = filtered_df[metric_to_check].mean()
            std_metric = filtered_df[metric_to_check].std()

            comp_rate = filtered_df[compress_matrix_string].mean() 

            latex_code += f'& {method_type} & ' + ' & '.join(task_metrics) + \
                  f' & {average_metric:.2f} \\scriptsize{{$\\pm$ {std_metric:.2f}}}' + f' & {comp_rate:.2f} \\\\ \n'
        latex_code += r'\midrule' + '\n'

        latex_code += r'\multirow{5}{*}{' + f'{model_type}' + ' full (F)}' + '\n'
        for rank in ranks:
            method_type = f'{rank} F'
            filtered_df = raw_df[(raw_df["model_type"] == model_type) & (raw_df["merge_type"] == "full") & (raw_df["rank"] == rank)]
            task_metrics = get_task_metrics(filtered_df, metric_to_check)
            average_metric = filtered_df[metric_to_check].mean()
            std_metric = filtered_df[metric_to_check].std()

            comp_rate = filtered_df[compress_matrix_string].mean() 

            latex_code += f'& {method_type} & ' + ' & '.join(task_metrics) + f' & {average_metric:.2f} \\scriptsize{{$\\pm$ {std_metric:.2f}}}' + f' & {comp_rate:.2f} \\\\ \n'
            
        latex_code += r'\midrule' + '\n'
        latex_code += r'\midrule' + '\n'

    latex_code += r'\end{tabular}'
    return latex_code


def generate_normalized_latex_table(raw_df, tasks, tasks_short):
    metric_to_check = "rougeL"

    latex_code = r'''
    \begin{tabular}{c|c|c|c|c|c|c|c|c|c|c|c|c|c}
    \toprule
    \multirow{2}{*}{Model Type} & \multirow{2}{*}{Method Type} & \multicolumn{10}{c}{Tasks} & \multirow{2}{*}{Average} & \multirow{2}{*}{Compression} \\
    \cmidrule{3-12}
    & & ''' + ' & '.join(tasks_short) + r''' & \\
    \midrule
    '''

    def get_normalized_metrics(df, base_df, lora_df, metric):
        metrics = []
        for task in tasks:
            task_df = df[df['task'] == task]
            base_task_df = base_df[base_df['task'] == task]
            lora_task_df = lora_df[lora_df['task'] == task]
            
            if not task_df.empty and not base_task_df.empty and not lora_task_df.empty:
                compressed_val = task_df[metric].mean()
                base_val = base_task_df[metric].mean()
                lora_val = lora_task_df[metric].mean()
                if (lora_val - base_val) != 0:
                    normalized_val = (compressed_val - base_val) / (lora_val - base_val)
                else:
                    normalized_val = float('nan')
                metrics.append(f'{normalized_val:.2f}')
            else:
                metrics.append('-')
        return metrics

    # Extract base and LoRA DataFrames
    df_base = df[df['model_type'] == 'base']
    df_base = df_base.drop_duplicates(subset=['task', 'merge_type'])

    df_lora = df[df['model_type'] == 'lora']
    df_lora = df_lora.drop_duplicates(subset=['task', 'merge_type'])

    # Base and LoRA models normalized metrics
    for model_type in ["base", "lora"]:
        filtered_df = raw_df[raw_df["model_type"] == model_type]
        if model_type == "base":
            task_metrics = get_normalized_metrics(filtered_df, df_base, df_lora, metric_to_check)
        else:
            task_metrics = get_normalized_metrics(df_lora, df_base, df_lora, metric_to_check)
        average_metric = '-'
        latex_code += f'{model_type} & {model_type} & ' + ' & '.join(task_metrics) + f' & {average_metric} \\\\ \n'

    latex_code += r'\midrule\midrule' + '\n'

    # SVD results
    svd_ranks = [2, 4, 8, 16]
    latex_code += r'\multirow{4}{*}{SVD}' + '\n'
    for svd_rank in svd_ranks:
        method_type = f'SVD {svd_rank}'
        filtered_df = raw_df[(raw_df["merge_type"] == "SVD") & (raw_df["rank"] == svd_rank)]
        task_metrics = get_normalized_metrics(filtered_df, df_base, df_lora, metric_to_check)
        # average_metric = '-'
        # average_metric = filtered_df[metric_to_check].mean()

        # Average every number in task_metrics list, except for the '-'
        task_metrics_ave = [float(x) for x in task_metrics if x != '-']
        if len(task_metrics_ave) == 0:
            average_metric = '-'
        else:
            average_metric = f'{sum(task_metrics_ave) / len(task_metrics_ave):.2f}'

        latex_code += f'& {method_type} & ' + ' & '.join(task_metrics) + f' & {average_metric} \\\\ \n'
    latex_code += r'\midrule\midrule' + '\n'

    # For each model type, for diagonal and full, for rank in [16, 32, 64, 128, 256]
    for model_type in model_type_list:
        latex_code += r'\multirow{5}{*}{' + f'{model_type}' + ' diagonal (D)}' + '\n'
        for rank in ranks:
            method_type = f'{rank} D'
            filtered_df = raw_df[(raw_df["model_type"] == model_type) & (raw_df["merge_type"] == "diagonal") & (raw_df["rank"] == rank)]
            task_metrics = get_normalized_metrics(filtered_df, df_base, df_lora, metric_to_check)
            # average_metric = '-'
            # average_metric = filtered_df[metric_to_check].mean()

            task_metrics_ave = [float(x) for x in task_metrics if x != '-']
            if len(task_metrics_ave) == 0:
                average_metric = '-'
            else:
                average_metric = f'{sum(task_metrics_ave) / len(task_metrics_ave):.2f}'
            comp_rate = filtered_df["total_comp_rate"].mean()
            latex_code += f'& {method_type} & ' + ' & '.join(task_metrics) + f' & {average_metric} ' + f' & {comp_rate}  \\\\ \n'
        latex_code += r'\midrule' + '\n'

        latex_code += r'\multirow{5}{*}{' + f'{model_type}' + ' full (F)}' + '\n'
        for rank in ranks:
            method_type = f'{rank} F'
            filtered_df = raw_df[(raw_df["model_type"] == model_type) & (raw_df["merge_type"] == "full") & (raw_df["rank"] == rank)]
            task_metrics = get_normalized_metrics(filtered_df, df_base, df_lora, metric_to_check)
            # average_metric = '-'
            # average_metric = filtered_df[metric_to_check].mean()

            task_metrics_ave = [float(x) for x in task_metrics if x != '-']
            if len(task_metrics_ave) == 0:
                average_metric = '-'
            else:
                average_metric = f'{sum(task_metrics_ave) / len(task_metrics_ave):.2f}'

            comp_rate = filtered_df["total_comp_rate"].mean()

            latex_code += f'& {method_type} & ' + ' & '.join(task_metrics) + f' & {average_metric}' + f' & {comp_rate} \\\\ \n'
        latex_code += r'\midrule' + '\n'

    latex_code += r'\end{tabular}'
    return latex_code


if __name__ == "__main__":

    base_path = '/Users/jiachengzhu/Desktop/DoResearch/WritingPaper/2024_LoRA_Merging/in-distribution'  # Replace with the actual path to your data
    # base_path = '/Users/jiachengzhu/Desktop/DoResearch/WritingPaper/2024_LoRA_Merging/in-distribution_pr_copy'  # Replace with the actual path to your data
    
    # Read the metrics data as a list of dictionaries
    metrics_data = read_metrics(base_path, )

    df = pd.DataFrame(metrics_data)

    pd.set_option('display.max_rows', None)  # Show all rows
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.width', None)  # No line width limit
    pd.set_option('display.max_colwidth', None)  # No limit for column width

    # print(df)
    print("=" * 10, "\n")

    # exit()

    # Get the unique task names
    task_names = list(df['task'].unique())
    task_names.sort(key=lambda x: int(x.split('_')[0][4:])) 
    print("task_names =", task_names, "\n")

    # Get a new list that are all x.split('_')[0]
    task_names_short = [x.split('_')[0] for x in task_names]

    """
    Generate the "Absolute" Latex Table
    """
    # latex_code = generate_latex_table(df, task_names, task_names_short)
    # print(latex_code)
    # exit()

    """
    Find the relative performance of each model type
    """
    df_base = df[df['model_type'] == 'base']
    df_base = df_base.drop_duplicates(subset=['task', 'merge_type']).set_index('task')
    df_lora = df[df['model_type'] == 'lora']
    df_lora = df_lora.drop_duplicates(subset=['task', 'merge_type']).set_index('task')


    # Join the base and lora data
    df = df.set_index('task')
    df = df.join(df_base[['exact_match', 'rouge1', 'rougeL']], on='task', rsuffix='_base')
    df = df.join(df_lora[['exact_match', 'rouge1', 'rougeL']], on='task', rsuffix='_lora')

    # Compute (xx - base) / (lora - base) for all other model types
    for col in ['exact_match', 'rouge1', 'rougeL']:
        # df[col] = (df[col] - df[f'{col}_base']) / (df[f'{col}_lora'] - df[f'{col}_base'])
        df[col] = (df[col]) / (df[f'{col}_lora'])

    # Reset index and keep only necessary columns
    df = df.reset_index()
    df = df[['experiment', 'model_type', 'task', 'rank', 'merge_type', 'exact_match', 'rouge1', 'rougeL']]

    # Display the result DataFrame
    # print(df)

    """
    Insert compression rate
    Per-model GPU compression rate
    """
    def compute_comp_rate(row):
        if row['merge_type'] == 'diagonal':
            return 1 - (row['rank'] / 131072) # 16*4096*2 = 131072
        elif row['merge_type'] == 'full':
            return 1 - ((row['rank'] ** 2) / 131072)
        elif row['merge_type'] == 'SVD':
            return 1 - (row['rank'] * 4096 * 2 / 131072)
        elif row['merge_type'] == 'TIES':
            return 0
        else:
            return 0  # or some default value if needed
    
    # Total compression rate
    def compute_comp_rate_2(row):
        if row['merge_type'] == 'diagonal':
            return 1 - ((row['rank'] * 4096 * 2 + int(row["model_type"]) * row['rank']) / (int(row["model_type"]) * 131072))
        elif row['merge_type'] == 'full':
            return 1 - ((row['rank'] * 4096 * 2 +  int(row["model_type"]) * (row['rank']**2))/ (int(row["model_type"]) * 131072))
        elif row['merge_type'] == 'SVD':
            return 1 - ((row['rank'] * 4096 * 2) / (131072))
        elif row['merge_type'] == 'TIES':
            return 1 - 1 / int(row["model_type"])
        else:
            return 0
    
    df['total_comp_rate'] = df.apply(compute_comp_rate_2, axis=1)
    df["gpu_comp_rate"] = df.apply(compute_comp_rate, axis=1)

    print(df)

    # exit()

    latex_code = generate_latex_table(df, task_names, task_names_short)
    print(latex_code)



    exit()

    
    df_copy = df.copy()

    df_copy['standardized_model_type'] = df_copy.apply(
        lambda row: 'SVD' if row['merge_type'] == 'SVD' else row['model_type'], axis=1)
    
    df_aggre = df_copy.groupby(['standardized_model_type', 'task', 'rank', 'merge_type']).agg(
        exact_match_mean=pd.NamedAgg(column='exact_match', aggfunc='mean'),
        exact_match_std=pd.NamedAgg(column='exact_match', aggfunc='std'),
        rouge1_mean=pd.NamedAgg(column='rouge1', aggfunc='mean'),
        rouge1_std=pd.NamedAgg(column='rouge1', aggfunc='std'),
        rougeL_mean=pd.NamedAgg(column='rougeL', aggfunc='mean'),
        rougeL_std=pd.NamedAgg(column='rougeL', aggfunc='std'),
        count=pd.NamedAgg(column='exact_match', aggfunc='count')
    ).reset_index()

    # print(df_aggre)

    print("=" * 10)

    # selected_df_missing = df[df["count"] == 2]
    # print(selected_df_missing[["model_type", "task", "rank", "merge_type", "count"]])

    # exit()

    model_type_select = 10

    merge_type_select = "SVD"

    selected_df = df_aggre[["standardized_model_type", "task", "rank", "merge_type", "rougeL_mean", "rougeL_std", "count"]]
    # selected_df = df[df['model_type'] == model_type_select][["standardized_model_type", "task", "rank", "merge_type", "rougeL_mean", "rougeL_std"]]
    # selected_df = df[df['merge_type'] == merge_type_select][["model_type", "task", "rank", "merge_type", "rougeL_mean", "rougeL_std"]]

    print(selected_df[selected_df['standardized_model_type'] == 500])

    exit()

    # Define theme colors for each task
    theme_colors = {
        'task039': (0.2, 0.4, 0.6),  # Blue
        'task1342': (0.6, 0.4, 0.2),  # Orange
        'task1391': (0.4, 0.6, 0.2),  # Green
        'task1598': (0.8, 0.2, 0.4),  # Red
        'task190': (0.4, 0.2, 0.6),  # Purple
        'task280': (0.2, 0.6, 0.4),  # Teal
        'task290': (0.6, 0.2, 0.4),  # Magenta
        'task391': (0.4, 0.4, 0.4),  # Gray
        'task442': (0.8, 0.6, 0.2),  # Yellow
        'task620': (0.2, 0.8, 0.6)   # Cyan
    }

    # plot_bar_with_error(metrics_data, model_type=10)
    # Create the bar plot

    model_type_select = 500
    metric_to_check = "exact_match"
    df = df[df['model_type'] == model_type_select]
    fig, ax = plt.subplots(figsize=(60, 10))
    width = 0.15
    tasks = df['task'].unique()
    x = range(len(tasks))

    for i, task in enumerate(tasks):
        task_df = df[df['task'] == task]
        task_df = task_df.sort_values('merge_type')
        task_id = task.split("_")[0]

        theme_color = theme_colors[task_id]
        
        for j, (_, row) in enumerate(task_df.iterrows()):
            merge_type = row['merge_type']
            mean = row[metric_to_check + '_mean']
            std = row[metric_to_check + '_std']
            rank = row['rank']
            
            if merge_type == 'diagonal':
                color = 'blue'
            else:
                color = 'orange'
            
            # Calculate color gradient based on rank
            color_factor = (rank - task_df['rank'].min()) / (task_df['rank'].max() - task_df['rank'].min()) * 0.75
            color = tuple(c * (1 - color_factor) + color_factor for c in theme_color)
            
            if merge_type == 'diagonal':
                hatch = '+'
            else:
                hatch = 'x'

            if j == 0 or j == (len(task_df) - 1):
                ax.bar(i + i * 0.8 + j * width, mean, width, yerr=std, color=color, alpha=0.7, hatch=hatch, label=f'{task_id} - {merge_type}')
            else:
                ax.bar(i + i * 0.8 + j * width, mean, width, yerr=std, color=color, alpha=0.7,hatch=hatch,)
    
    ax.set_xticks(np.asarray(x) * 1.95)
    ax.set_xticklabels([task.split('_', maxsplit=1)[1] for task in tasks], rotation=15, ha='right')
    ax.set_ylabel('Exact Match Mean')
    ax.set_title('Merging ' + str(model_type_select) + "LoRA's, metric: " + metric_to_check)
    ax.legend(loc='upper left',
              bbox_to_anchor=(1.02, 1)
              )

    # plt.tight_layout()
    plt.show()
    plt.savefig()