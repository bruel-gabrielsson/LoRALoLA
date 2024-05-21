import os
import shutil

# Define the source and target directories
source_dir = "combined_500rank256_all_runs"
target_dir = "in-distribution"

# Define the list of subdirectories for the source and target
source_subdirs = ["outputs_s500t10in_r1", "outputs_s500t10in_r2", "outputs_s500t10in_r3"]
target_subdirs = ["combined_t10_r1/outputs_s500t10in_r1", 
                  "combined_t10_r2/outputs_s500t10in_r2", 
                  "combined_t10_r3/outputs_s500t10in_r3"]

# Define the folder names to move
folders_to_move  = [
    "task039_qasc_find_overlapping_words",
    "task190_snli_classification",
    "task280_stereoset_classification_stereotype_type",
    "task290_tellmewhy_question_answerability",
    "task391_causal_relationship",
    "task442_com_qa_paraphrase_question_generation",
    "task620_ohsumed_medical_subject_headings_answer_generation",
    "task1342_amazon_us_reviews_title",
    "task1391_winogrande_easy_answer_generation",
    "task1598_nyc_long_text_generation"
]

# Loop through each subdirectory and move the folders
for source_sub, target_sub in zip(source_subdirs, target_subdirs):
    source_path = os.path.join(source_dir, source_sub)
    target_path = os.path.join(target_dir, target_sub)

    # Ensure the target directory exists
    print(f"Moving folders from || {source_path} || to || {target_path} ||")

    # exit()

    # os.makedirs(target_path, exist_ok=True)

    # for task_folder in os.listdir(source_path):
    #     source_path = os.path.join(source_path, task_folder)

    # print("source_path =", source_path)

    # exit()

    for task_folder in folders_to_move:
        print()
        print("task_folder =", task_folder)

        source_path_task = os.path.join(source_path, task_folder)

        print("source_path_task =", source_path_task)

        for model_name in ["rank_256_type_diagonal_trans_normalize",  "rank_256_type_full_trans_normalize"]:
            source_path_model = os.path.join(source_path_task, model_name)

            print("source_path_model =", source_path_model)

            # exit()

            if os.path.exists(source_path_model):
                print("*"*20)
                print("Exist source_path_model =", source_path_model)

                target_path_task = os.path.join(target_path, task_folder, model_name)
                target_path_to_move = os.path.join(target_path, task_folder)

                print("target_path_task =", target_path_task)
                print("os.path.exists(target_path_task) =", os.path.exists(target_path_task))

                if not os.path.exists(target_path_task):
                    os.makedirs(target_path_task, exist_ok=True)
                
                if os.path.exists(target_path_task):
                    # remove the folder
                    shutil.rmtree(target_path_task)

                shutil.move(source_path_model, target_path_to_move,)
                print("Moved folder from ||", source_path_model, "|| to ||", target_path_task, "||")

        # exit()

