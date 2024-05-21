import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

"""
Visualize the performance v.s. compression rate
"""

def read_metrics(base_path):
    results = []

    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file == "lola_model_metrics.json":
                metrics_file = os.path.join(root, file)
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                
                path_parts = os.path.relpath(root, base_path).split(os.sep)
                experiment = path_parts[0]
                model_type_raw = path_parts[1]
                model_type = int(model_type_raw[9 : -8])
                task = path_parts[2]
                rank_raw = path_parts[3]
                rank_raw_list = rank_raw.split("_")
                rank = int(rank_raw_list[1])
                merge_type = rank_raw_list[3]

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
merge_type_list = ["diagonal", "full"]
ranks = [16, 32, 64, 128, 256]

def generate_latex_table(raw_df, tasks, tasks_short):

    metric_to_check = "rougeL"

    latex_code = r'''
    \begin{tabular}{c|c|c|c|c|c|c|c|c|c|c|c|c}
    \toprule
    \multirow{2}{*}{Model Type} & \multirow{2}{*}{Method Type} & \multicolumn{10}{c}{Tasks} & \multirow{2}{*}{Average} \\
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
        latex_code += f'& {model_type} & ' + ' & '.join(task_metrics) + f' & {average_metric:.2f} \\scriptsize{{$\\pm$ {std_metric:.2f}}} \\\\ \n'

    latex_code += r'\midrule\midrule' + '\n'

    # SVD results
    svd_ranks = [2, 4, 8, 16]
    latex_code += r'\multirow{4}{*}{SVD}' + '\n'
    for svd_rank in svd_ranks:
        method_type = f'SVD {svd_rank}'
        filtered_df = raw_df[(raw_df["merge_type"] == "SVD") & (raw_df["rank"] == svd_rank)]
        task_metrics = get_task_metrics(filtered_df, metric_to_check)
        average_metric = filtered_df[metric_to_check].mean()
        std_metric = filtered_df[metric_to_check].std()
        latex_code += f'& {method_type} & ' + ' & '.join(task_metrics) + f' & {average_metric:.2f} \\scriptsize{{$\\pm$ {std_metric:.2f}}} \\\\ \n'
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
            latex_code += f'& {method_type} & ' + ' & '.join(task_metrics) + f' & {average_metric:.2f} \\scriptsize{{$\\pm$ {std_metric:.2f}}} \\\\ \n'
        latex_code += r'\midrule' + '\n'

        latex_code += r'\multirow{5}{*}{' + f'{model_type}' + ' full (F)}' + '\n'
        for rank in ranks:
            method_type = f'{rank} F'
            filtered_df = raw_df[(raw_df["model_type"] == model_type) & (raw_df["merge_type"] == "full") & (raw_df["rank"] == rank)]
            task_metrics = get_task_metrics(filtered_df, metric_to_check)
            average_metric = filtered_df[metric_to_check].mean()
            std_metric = filtered_df[metric_to_check].std()
            latex_code += f'& {method_type} & ' + ' & '.join(task_metrics) + f' & {average_metric:.2f} \\scriptsize{{$\\pm$ {std_metric:.2f}}} \\\\ \n'
        latex_code += r'\midrule' + '\n'
        latex_code += r'\midrule' + '\n'

    latex_code += r'\end{tabular}'
    return latex_code


def generate_normalized_latex_table(raw_df, tasks, tasks_short):
    metric_to_check = "rougeL"

    latex_code = r'''
    \begin{tabular}{c|c|c|c|c|c|c|c|c|c|c|c|c}
    \toprule
    \multirow{2}{*}{Model Type} & \multirow{2}{*}{Method Type} & \multicolumn{10}{c}{Tasks} & \multirow{2}{*}{Average} \\
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

            latex_code += f'& {method_type} & ' + ' & '.join(task_metrics) + f' & {average_metric} \\\\ \n'
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

            latex_code += f'& {method_type} & ' + ' & '.join(task_metrics) + f' & {average_metric} \\\\ \n'
        latex_code += r'\midrule' + '\n'

    latex_code += r'\end{tabular}'
    return latex_code



if __name__ == "__main__":

    base_path = '/Users/jiachengzhu/Desktop/DoResearch/WritingPaper/2024_LoRA_Merging/in-distribution'  # Replace with the actual path to your data
    
    # Read the metrics data as a list of dictionaries
    metrics_data = read_metrics(base_path, )

    df = pd.DataFrame(metrics_data)

    # Save the raw DataFrame to a CSV file
    df.to_csv('metrics_raw.csv', index=False)

    pd.set_option('display.max_rows', None)  # Show all rows
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.width', None)  # No line width limit
    pd.set_option('display.max_colwidth', None)  # No limit for column width

    # print(df)
    # print("=" * 10, "\n")

    # Get the unique task names
    task_names = list(df['task'].unique())
    task_names.sort(key=lambda x: int(x.split('_')[0][4:])) 
    print("task_names =", task_names, "\n")

    # Get a new list that are all x.split('_')[0]
    task_names_short = [x.split('_')[0] for x in task_names]

    """
    Get the unique model types
    """
    # latex_code = generate_latex_table(df, task_names, task_names_short)
    # print(latex_code)


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
        #df[col] = (df[col] - df[f'{col}_base']) / (df[f'{col}_lora'] - df[f'{col}_base'])
        df[col] = (df[col]) / (df[f'{col}_lora'])

    # Reset index and keep only necessary columns
    df = df.reset_index()
    df = df[['experiment', 'model_type', 'task', 'rank', 'merge_type', 'exact_match', 'rouge1', 'rougeL']]

    # Display the result DataFrame
    # print(df)

    # Save this processed DataFrame to a CSV file
    # df.to_csv('metrics_normalized.csv', index=False)

    # exit()

    """
    Plot performance v.s. compression rate
    """
    # Sort the DataFrame by 'rougeL' in descending order
    df_sorted = df.sort_values(by='rougeL', ascending=False)

    # compression_type = "per_lora"
    compression_type = "total_lora"

    # dot_size_selected = "model_type" # or "model_type"
    dot_size_selected = "model_type" # or "model_type"
    mean_or_min = "mean" # or "min"

    # Define a function to compute comp_rate based on merge_type and rank
    def compute_comp_rate(row):
        if row['merge_type'] == 'diagonal':
            return row['rank'] / 131072 # 16*4096*2 = 131072
        elif row['merge_type'] == 'full':
            return (row['rank'] ** 2) / 131072
        elif row['merge_type'] == 'SVD':
            return row['rank'] * 4096 * 2 / 131072
        else:
            return None  # or some default value if needed
    
    def compute_comp_rate_2(row):
        if row['merge_type'] == 'diagonal':
            return (row['rank'] * 4096 * 2 + int(row["model_type"]) * row['rank']) / (int(row["model_type"]) * 131072)
        elif row['merge_type'] == 'full':
            return (row['rank'] * 4096 * 2 +  int(row["model_type"]) * (row['rank']**2))/ (int(row["model_type"]) * 131072)
        elif row['merge_type'] == 'SVD':
            return (row['rank'] * 4096 * 2) / (131072)
        else:
            return None

    # Apply the function to each row in the DataFrame
    if compression_type == "per_lora":
        df_sorted['comp_rate'] = df_sorted.apply(compute_comp_rate, axis=1)
    elif compression_type == "total_lora":
        df_sorted['comp_rate'] = df_sorted.apply(compute_comp_rate_2, axis=1)


    print(df_sorted)

    print("=" * 10, "after aggregation", "=" * 10)
    # df_sorted = df_sorted.groupby(['model_type', 'task', 'rank', 'merge_type']).mean().reset_index()

    df_sorted = df_sorted.groupby(['model_type', 'task', 'merge_type', 'rank']).agg(
        exact_match_mean=pd.NamedAgg(column='exact_match', aggfunc='mean'),
        exact_match_std=pd.NamedAgg(column='exact_match', aggfunc='std'),
        rouge1_mean=pd.NamedAgg(column='rouge1', aggfunc='mean'),
        rougeL_mean=pd.NamedAgg(column='rougeL', aggfunc='mean'),
        comp_rate_mean=pd.NamedAgg(column='comp_rate', aggfunc='mean'),
        ).reset_index()

    df_sorted = df_sorted.sort_values(by='rougeL_mean', ascending=False)
    # print(df_sorted)

    filtered_df = df_sorted[df_sorted['rougeL_mean'] > 0.95]

    # Define the colors and markers based on merge_type and task
    colors = {'SVD': 'black', 'full': 'red', 'diagonal': 'orange'}
    markers = {
        'task039_qasc_find_overlapping_words': 'o',
        'task190_snli_classification': 's',
        'task280_stereoset_classification_stereotype_type': 'D',
        'task290_tellmewhy_question_answerability': 'v',
        'task391_causal_relationship': '^',
        'task442_com_qa_paraphrase_question_generation': '<',
        'task620_ohsumed_medical_subject_headings_answer_generation': '>',
        'task1342_amazon_us_reviews_title': 'p',
        'task1391_winogrande_easy_answer_generation': '*',
        'task1598_nyc_long_text_generation': 'h'
    }


    """
    Plot all the samples
    """
    # plt.figure(figsize=(10, 6))
    # for task in filtered_df['task'].unique():
    #     task_df = filtered_df[filtered_df['task'] == task]
    #     # for merge_type in task_df['merge_type'].unique():
    #     for merge_type in ['diagonal', 'full', 'SVD']:
    #         subset_df = task_df[task_df['merge_type'] == merge_type]
    #         plt.scatter(1 - subset_df['comp_rate_mean'], subset_df['rougeL_mean'], 
    #                     c=colors[merge_type], 
    #                     marker=markers[task], 
    #                     label=f'{task} - {merge_type}')
    # # Adding labels and title
    # plt.xlabel('Comp Rate (1 - active parameter / LoRA)')
    # plt.ylabel('Performance Improvement than LoRA')
    # plt.title('Relationship between Comp Rate and RougeL, RougeL > 0.95')
    # plt.legend(loc='best')
    # plt.show()

    """
    Average across all tasks
    """

    final_aggregated_df = df_sorted.groupby(['model_type', 'merge_type', 'rank']).agg(
        exact_match_mean_avg=pd.NamedAgg(column='exact_match_mean', aggfunc='mean'),
        rouge1_mean_avg=pd.NamedAgg(column='rouge1_mean', aggfunc='mean'),
        rougeL_mean_avg=pd.NamedAgg(column='rougeL_mean', aggfunc='mean'),
        rougeL_min_avg=pd.NamedAgg(column='rougeL_mean', aggfunc='min'),
        comp_rate_mean_avg=pd.NamedAgg(column='comp_rate_mean', aggfunc='mean')
    ).reset_index()

    
    # For merge_type in ['SVD'] average across all ranks and model types, and set model_type to be 10

    final_aggregated_df_svd = final_aggregated_df[final_aggregated_df['merge_type'] == 'SVD']

    print("final_aggregated_df_svd")
    print(final_aggregated_df_svd)

    final_aggregated_df_svd_agg = final_aggregated_df_svd.groupby(['rank', 'merge_type']).mean().reset_index()
    final_aggregated_df_svd_agg['model_type'] = 10
    final_aggregated_df_svd_agg['merge_type'] = 'SVD'

    final_aggregated_df_other = final_aggregated_df[final_aggregated_df['merge_type'] != 'SVD']

    final_aggregated_df = pd.concat([final_aggregated_df_other, final_aggregated_df_svd_agg])


    print(final_aggregated_df)

    # Filter the DataFrame for rougeL_mean_avg > 0.95
    if mean_or_min == "mean":
        filtered_df = final_aggregated_df[final_aggregated_df['rougeL_mean_avg'] > 0.95]
    elif mean_or_min == "min":
        # filtered_df = final_aggregated_df[final_aggregated_df['rougeL_mean_avg'] > 0.95]
        filtered_df = final_aggregated_df[final_aggregated_df['rougeL_min_avg'] > 0.95]

    # Define the colors based on merge_type
    colors = {'SVD': 'black', 'full': 'red', 'diagonal': 'orange'}

    shapes = {'SVD': 'v', 'full': 'o', 'diagonal': 'o'}

    # Create the scatter plot
    plt.figure(figsize=(12, 8))

    

    for merge_type in ['diagonal', 'full', 'SVD']:
        subset_df = filtered_df[filtered_df['merge_type'] == merge_type]
        # print(subset_df)
        subset_df["model_type"] = subset_df["model_type"].astype(int)
        # exit()
        sc = plt.scatter(1 - subset_df['comp_rate_mean_avg'], subset_df[f'rougeL_{mean_or_min}_avg'], 
                    c=colors[merge_type], 
                    # s=subset_df['model_type']*10,
                    s=subset_df[dot_size_selected]*20,
                    marker=shapes[merge_type],
                    label=f'{merge_type}', alpha=0.5)

    # Adding color bar for rank sizes
    # cbar = plt.colorbar(sc)
    # cbar.set_label(dot_size_selected)

    # Adding labels and title
    if compression_type == "per_lora":
        plt.xlabel('GPU Load Saved (%) (1 - active parameter / LoRA)', fontsize=16)
        plt.ylabel(f'{mean_or_min} Performance Improvement over LoRA', fontsize=16)
        plt.title(f'GPU Workload Compress Rate (↑) v.s. {mean_or_min} Improvement over LoRA (↑)', fontsize=16)


    elif compression_type == "total_lora":
        plt.xlabel('Total Parameter Saved (%) (1 - active parameter / LoRA)', fontsize=16)
        plt.ylabel(f'{mean_or_min} Performance Improvement over LoRA', fontsize=16)
        plt.title(f'Total parameter Compress Rate (↑) v.s. {mean_or_min} Improvement over LoRA (↑)', fontsize=16)

        plt.xlim(-0.1, 1.1)

    # Make ticks larger
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    import matplotlib.lines as mlines
    size_coe = 0.5

    if dot_size_selected == "rank":
        legend_elements = [
            mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', markersize=4**size_coe, label='Rank 4'),
            mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', markersize=8**size_coe, label='Rank 8'),
            mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', markersize=16**size_coe, label='Rank 16'),
            mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', markersize=32**size_coe, label='Rank 32'),
            mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', markersize=64**size_coe, label='Rank 64'),
            mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', markersize=128**size_coe, label='Rank 128'),
            mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', markersize=256**size_coe, label='Rank 256'),
        
            mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=10, label='SVD'),
            mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Full'),
            mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Diagonal')
        ]
    elif dot_size_selected == "model_type":

        legend_elements = [
            mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', markersize=10**size_coe, label='LoRA Num: 10'),
            mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', markersize=50**size_coe, label='LoRA Num: 50'),
            mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', markersize=100**size_coe, label='LoRA Num: 100'),
            mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', markersize=500**size_coe, label='LoRA Num: 500'),

            mlines.Line2D([0], [0], marker='v', color='w', markerfacecolor='black', markersize=10, label='SVD'),
            mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Full'),
            mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Diagonal')
        ]

    # Make the legend closer to the plot
    plt.legend(handles=legend_elements, loc='lower left', 
               # bbox_to_anchor=(1.05, 1), 
               fontsize=16)
    # plt.legend(loc='best', bbox_to_anchor=(1.05, 1), fontsize='small')
    # grid
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    exit()

    
