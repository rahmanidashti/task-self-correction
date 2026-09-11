import os
import json

import hydra
from hydra.core.config_store import ConfigStore

from config import MainConfig
from dataset import Dataset
from label import Label
from utils import Util

from tqdm import tqdm
from collections import defaultdict

def get_mc_correct_count(eval_file, ground_truth):
    iteration_counts = {}
    for question_id, question_data in tqdm(eval_file.items()):
        answers = question_data["answers"]
        # Assert to ensure the question matches the ground truth
        assert ground_truth[question_id][0] == question_data["question"], f"Question mismatch for {question_id}"
        for iter_key, answer in answers.items():
            correct = 1 if answer == ground_truth[question_id][1] else 0
            if iter_key not in iteration_counts:
                iteration_counts[iter_key] = 0
            iteration_counts[iter_key] += correct
    return iteration_counts

def get_gen_correct_count(eval_file, eval_metric: str = "true"):
    # Initialize a dictionary to store correct counts and total counts per iteration
    iteration_counts = {}
    # Iterate through questions
    for question_id, question_data in tqdm(eval_file.items()):
        answers = question_data["answers"]
        for iter_key, answer in answers.items():
            correct = 1 if answer[eval_metric] == "yes" else 0
            if iter_key not in iteration_counts:
                iteration_counts[iter_key] = 0
            iteration_counts[iter_key] += correct
    return iteration_counts

def get_mc_cumulative_count(eval_file, ground_truth):
    cumulative_correct = defaultdict(int)
    for question_id, question_data in tqdm(eval_file.items()):
        correctly_predicted = False
        answers = question_data["answers"]
        # Assert to ensure the question matches the ground truth
        assert ground_truth[question_id][0] == question_data["question"], f"Question mismatch for {question_id}"
        for iter_key, answer in answers.items():
            # Check if the answer matches the ground truth
            if answer == ground_truth[question_id][1]:
                correctly_predicted = True
            # Count it as correct for all subsequent iterations once correctly predicted
            cumulative_correct[iter_key] += int(correctly_predicted)
    return cumulative_correct

def get_gen_cumulative_count(eval_file, eval_metric: str = "true"):
    cumulative_correct = defaultdict(int)
    for question_data in tqdm(eval_file.values()):
        correctly_predicted = False
        for iter_key, answer in question_data["answers"].items():
            if answer[eval_metric] == "yes":
                correctly_predicted = True
            # Count the sample as correct if it was correct in any previous or current iteration
            cumulative_correct[iter_key] += int(correctly_predicted)
    # returns a regular dictionary
    return dict(cumulative_correct)

def get_accuracy(correct_count, num_sample):
    accuracy_per_iteration = {
        iter_key: round(counts / num_sample, 4)
        for iter_key, counts in correct_count.items()
    }

    return accuracy_per_iteration

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: MainConfig):
    label = Label()
    dataset = Dataset(dataset=cfg.select.dataset, task=cfg.select.task).load_data()

    util = Util()

    file_name = util.get_file_name(model=cfg.select.model.replace("/", "_"), method=cfg.select.method, task=cfg.select.task,
                                   start_prompt=cfg.select.start_prompt, iterate_prompt=cfg.select.iterate_prompt,
                                   iteration=cfg.param.iteration,sampling=cfg.param.sampling,
                                   temp=cfg.param.temp, top_k=cfg.param.top_k, top_p=cfg.param.top_p)

    with open(f"results/evals/{cfg.select.dataset}/{cfg.select.method}/{cfg.select.task}/{file_name}.json", "r") as file:
        eval_data = json.load(file)
        num_samples = len(eval_data)

    # computing cumulative accuracy
    if cfg.select.task == "multiple-choice":
        labels = label.get_mc1_labels(ds=dataset)
        iteration_counts = get_mc_cumulative_count(eval_file=eval_data, ground_truth=labels)
    else:
        iteration_counts = get_gen_cumulative_count(eval_file=eval_data)

    accuracy_per_iter = get_accuracy(correct_count=iteration_counts, num_sample=num_samples)

    cumulative_dir = f"results/acc/{cfg.select.dataset}/{cfg.select.method}/cumulative/{cfg.select.task}"
    os.makedirs(cumulative_dir, exist_ok=True)
    with open(f"{cumulative_dir}/{file_name}.json", "w") as fout:
        json.dump(accuracy_per_iter, fout, indent=4)

    # computing simple accuracy
    if cfg.select.task == "multiple-choice":
        labels = label.get_mc1_labels(ds=dataset)
        iteration_counts = get_mc_correct_count(eval_file=eval_data, ground_truth=labels)
    else:
        iteration_counts = get_gen_correct_count(eval_file=eval_data)

    accuracy_per_iter = get_accuracy(correct_count=iteration_counts, num_sample=num_samples)

    iteration_dir = f"results/acc/{cfg.select.dataset}/{cfg.select.method}/iteration/{cfg.select.task}"
    os.makedirs(iteration_dir, exist_ok=True)
    with open(f"{iteration_dir}/{file_name}.json", "w") as f:
        json.dump(accuracy_per_iter, f, indent=4)

if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="main_config", node=MainConfig)
    main()
