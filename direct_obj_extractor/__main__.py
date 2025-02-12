"""
Module Entrypoint

This file is used to allow the project to be invoked via:
    `python -m traditional_feature_extraction`
"""
import configparser
import json
import os
import pandas as pd
import multiprocessing as mp
import time

from gpu_utils import process_on_gpu
from text_utils import clean_determiners

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("../direct-obj-extraction/config.ini")

    # Load Configuration Variables
    dataset_dir_path = config["paths"]["dataset_dir_path"]
    results_dir = config["paths"]["results_dir"]

    dataset_name = config["experiment"]["dataset_name"]

    # Save results
    out_path = results_dir + "/" + dataset_name
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)

    with open(f"{dataset_dir_path}metadata.json", "r", encoding="utf-8") as json_file:
        # The metadata file referenced here is contained in the repo:
        # dataset-download-scripts package hosted in the larger GitHub group.
        metadata_dict = json.load(json_file)

    ds_path = metadata_dict[dataset_name]

    # Begin text feature processing
    df = pd.read_csv(dataset_dir_path + ds_path)

    unique_texts = df[config["data"]["nl_column"]].dropna().unique().tolist()

    num_gpus = len(config["experiment"]["gpu_ids"])
    split_texts = [unique_texts[i::num_gpus] for i in range(num_gpus)]

    with mp.Pool(processes=num_gpus) as pool:
        gpu_results = pool.starmap(process_on_gpu, [(config["experiment"]["gpu_ids"][i], split_texts[i]) for i in range(num_gpus)])

    step_1_results = [item for sublist in gpu_results for item in sublist]
    
    step_1_df = pd.DataFrame(step_1_results, columns=["full_prompt", "llm_response", "direct_objs", "verbs"])
    step_1_df.to_csv(f"{out_path}/step_1_full_results.csv", index=False)

    step_1_df["direct_objects"] = step_1_df["direct_objects"].apply(lambda x: eval(x) if isinstance(x, str) else x)
    step_1_df["verbs"] = step_1_df["verbs"].apply(lambda x: eval(x) if isinstance(x, str) else x)

    exploded_df = step_1_df.explode("direct_objects").explode("verbs")

    step_2_df = exploded_df.groupby("direct_objects")["verbs"].unique().reset_index()
    step_2_df.columns = ["direct_object", "unique_verbs"]

    step_2_df_sorted = step_2_df.assign(cleaned_direct_object=step_2_df["direct_object"].apply(clean_determiners)).sort_values(by="cleaned_direct_object").drop(columns=["cleaned_direct_object"])
    step_2_df_sorted.to_csv(f"{out_path}/step_2_do_verb_results_only.csv", index=False)
