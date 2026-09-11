import hydra
from hydra.core.config_store import ConfigStore

from config import MainConfig
from utils import Util

import os
import json
from tqdm import tqdm
import re

def mcq_answer(response):
    patterns = [
        r"The final answer is:\s*\(([A-Z])\)",    # (C) -> with parentheses
        r"\(([A-Z])\)\s*The final answer is",    # (C) -> with parentheses before sentence
        r"The final answer is \s*\(([A-Z])\)",
        r"The final answer is:\s*\*\*\(([A-Z])\)",
        r"the final answer is:\s*\(([A-Z])\)",
        r"the final answer is:\s*\*\*\(([A-Z])\)\*\*",
        r"Final Answer:\s*\(([A-Z])\)",
        r"Final Answer is:\s*\(([A-Z])\)",
        # Final Answer:** (C)
        r"Final Answer:\*\*\s*\(([A-Z])\)",
        r"Final Answer is:\*\*\s*\(([A-Z])\)",
        r"The final answer is:\s*([A-Z])\)",       # C)  -> with closing parenthesis
        r"The final answer is:\s*([A-Z])\.",       # C.  -> with period
        r"The final answer is:\s*([A-Z])\s",        # C   -> without extra character
        r"The final answer is:\s*([A-Z])$",        # C   -> without extra character and no space
        r"The final answer is:\s*([A-Z])\*\*",
        r"^([A-Z])$"
    ]

    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            result = match.group(1)
            return result
    # if not, extract unique letter
    result = extract_unique_letter(response)
    if len(result) == 1:
        return list(result)[0]
    return None

# Function to extract unique letters in parentheses
def extract_unique_letter(response):
    matches = re.findall(r"\(([A-Z])\)", response)
    unique_letters = set(matches)
    return unique_letters

def search_match(response):
    patterns = [
        r"The final answer is:\s*((?:.|\n)*)",
        r"\*\*Final(?: answer| Answer) is:\*\*\s*(.*)",
        r"\*\*Final(?: answer| Answer):\*\*\s*(.*)",
        r"\*\*Final Answer is:\*\*\s*\"(.*)\"",
        r"\*\*Final Answer is:\*\*\s*(.*)",
        r"The final answer is:\s*(.*)", r"the final answer is:\s*(.*)",
        r"Final Answer is:\s*(.*)", r"Final answer is:\s*(.*)",
        r"Final Answer:\s*(.*)", r"Final Answer is:\*\*\s*(.*)",
        r"Final Answer:\*\*\s*(.*)", r"Final answer is:\*\*\s*(.*)"
        ]
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            result = match.group(1).strip()
            return result
    return response.strip()

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: MainConfig):

    util = Util()

    file_name = util.get_file_name(model=cfg.select.model.replace("/", "_"), method=cfg.select.method, task=cfg.select.task,
                                   start_prompt=cfg.select.start_prompt, iterate_prompt=cfg.select.iterate_prompt,
                                   iteration=cfg.param.iteration,sampling=cfg.param.sampling,
                                   temp=cfg.param.temp, top_k=cfg.param.top_k, top_p=cfg.param.top_p)

    with open(f"results/outputs/{cfg.select.dataset}/{cfg.select.method}/{cfg.select.task}/{file_name}.json", "r") as result_file:
        data = json.load(result_file)

    output_dir = f"results/process/{cfg.select.dataset}/{cfg.select.method}/{cfg.select.task}"
    os.makedirs(output_dir, exist_ok=True)

    output = dict()
    with open(f"{output_dir}/{file_name}.json", "w") as json_file:
        for sample_id, question_answers in tqdm(data.items()):
            responses = dict()
            for iter_id, answer in question_answers['answers'].items():
                # self-consistency stores [selected response, sampled responses]; keep the selected one
                if isinstance(answer, list):
                    answer = answer[0]
                try:
                    if cfg.select.task == "generation":
                        result = search_match(response=answer)
                    elif cfg.select.task == "multiple-choice":
                        result = mcq_answer(response=answer)
                        if not result:
                            print("#1", sample_id, iter_id)
                except Exception as e:
                    print("#2", sample_id, iter_id, e)
                    return
                responses[iter_id] = result
            output[sample_id] = {"question": question_answers['question'], "answers": responses}
        json.dump(output, json_file, indent=4)

if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="main_config", node=MainConfig)
    main()
