import hydra
from hydra.core.config_store import ConfigStore

import os
import json
from tqdm import tqdm

from evaluation import Evaluator
from config import MainConfig
from utils import Util
from dataset import Dataset
from prompt import Prompt
from models import Models, LiteLLMModel
from generation import Generation

@hydra.main(version_base=None, config_path="conf", config_name="config")
def eval(cfg: MainConfig):

    util = Util()

    file_name = util.get_file_name(model=cfg.select.model.replace("/", "_"), method=cfg.select.method, task=cfg.select.task,
                                   start_prompt=cfg.select.start_prompt, iterate_prompt=cfg.select.iterate_prompt,
                                   iteration=cfg.param.iteration,sampling=cfg.param.sampling,
                                   temp=cfg.param.temp, top_k=cfg.param.top_k, top_p=cfg.param.top_p)

    with open(f"results/process/{cfg.select.dataset}/{cfg.select.method}/{cfg.select.task}/{file_name}.json", "r") as result_file:
        eval_data = json.load(result_file)

    output_dir = f"results/evals/{cfg.select.dataset}/{cfg.select.method}/{cfg.select.task}"
    os.makedirs(output_dir, exist_ok=True)

    if cfg.select.task == "multiple-choice":
        # extracted options are matched against the gold option in result.py, so no judge is needed
        with open(f"{output_dir}/{file_name}.json", "w") as json_file:
            json.dump(eval_data, json_file, indent=4)
    elif cfg.select.dataset == "bbeh-disambiguation-qa":
        output = dict()
        # LLM-as-a-judge (select.eval_model, GPT-4o by default)
        with open(f"{output_dir}/{file_name}.json", "w") as json_file:
            tokenizer, model = Models(model_name=cfg.model[cfg.select.eval_model]).load_model()
            is_api = isinstance(model, LiteLLMModel)
            generation = Generation(tokenizer=tokenizer, model=model)
            dataset = Dataset(dataset=cfg.select.dataset, task=cfg.select.task).load_data()
            prompt = Prompt()
            # the questions and the response are in the same order so we can just iterate both together
            for sample_data, sample_eval in tqdm(zip(dataset, eval_data.items())):
                responses = dict()
                true_answer = sample_data['best_answer']
                sample_id = sample_eval[0]
                question = sample_eval[1]['question']
                # iterate over generated result
                for iter_id, gen_answer in sample_eval[1]['answers'].items():
                    # question, answer, best_answer, prompt
                    eval_prompt = prompt.get_bbeh_eval_prompt(question=question, generated_answer=gen_answer, gold_answer=true_answer)
                    eval_response = generation.generate(is_api=is_api, prompt=eval_prompt, sampling=False, temperature=0, top_k=50, top_p=1)
                    responses[iter_id] = {'info': 'NA', 'true': eval_response}
                output[sample_id] = {"question": question, "answers": responses}
                json_file.seek(0)  # Move cursor to the beginning of the file
                json_file.truncate()  # Remove previous content
                json.dump(output, json_file, indent=4)  # Write new content
                json_file.flush()  # Ensure the data is immediately written to the file
    else:
        # TruthfulQA truth and informativeness judges
        evaluator = Evaluator()
        output = dict()
        with open(f"{output_dir}/{file_name}.json", "w") as json_file:
            for sample_id, question_answers in tqdm(eval_data.items()):
                responses = dict()
                for iter_id, answer in question_answers['answers'].items():
                    info = evaluator.Informativeness(question=question_answers['question'], generated_text=answer)
                    true = evaluator.Truthfulness(question=question_answers['question'], generated_text=answer)
                    responses[iter_id] = {'info': info, 'true': true}
                output[sample_id] = {"question": question_answers['question'], "answers": responses}
                json_file.seek(0)  # Move cursor to the beginning of the file
                json_file.truncate()  # Remove previous content
                json.dump(output, json_file, indent=4)  # Write new content
                json_file.flush()  # Ensure the data is immediately written to the file

if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="main_config", node=MainConfig)
    eval()
