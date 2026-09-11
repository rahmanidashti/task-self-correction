import os

import hydra
from hydra.core.config_store import ConfigStore

from config import MainConfig
from dataset import Dataset
from models import Models, LiteLLMModel
from generation import Generation
from run import Run
from utils import Util

from huggingface_hub import login

METHODS = ("baseline", "CoT", "self-consistency")
TASKS = ("generation", "multiple-choice")

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: MainConfig):
    if cfg.select.method not in METHODS or cfg.select.task not in TASKS:
        raise ValueError(f"Unknown method/task: {cfg.select.method}/{cfg.select.task}")
    if cfg.api.HF:
        login(token=cfg.api.HF)
    util = Util()
    dataset = Dataset(dataset=cfg.select.dataset, task=cfg.select.task).load_data()
    tokenizer, model = Models(model_name=cfg.model[cfg.select.model]).load_model()
    # API models (e.g., Gemini) are called through LiteLLM instead of a local HF model
    is_api = isinstance(model, LiteLLMModel)
    generation = Generation(tokenizer=tokenizer, model=model)
    print("Starting ...")

    file_name = util.get_file_name(model=cfg.select.model.replace("/", "_"), method=cfg.select.method, task=cfg.select.task,
                                   start_prompt=cfg.select.start_prompt, iterate_prompt=cfg.select.iterate_prompt,
                                   iteration=cfg.param.iteration,sampling=cfg.param.sampling,
                                   temp=cfg.param.temp, top_k=cfg.param.top_k, top_p=cfg.param.top_p)

    output_dir = f"results/outputs/{cfg.select.dataset}/{cfg.select.method}/{cfg.select.task}"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/{file_name}.json", "w") as json_file:
        runner = Run(generator=generation, result_file=json_file)
        if cfg.select.method == "self-consistency":
            if cfg.select.task == "generation":
                runner.run_gen_self_consistency(is_api=is_api, dataset=dataset,
                                                sprompt=cfg.select.start_prompt, iprompt=cfg.select.iterate_prompt,
                                                num_iters=cfg.param.iteration, sampling=cfg.param.sampling,
                                                temp=cfg.param.temp, topk=cfg.param.top_k, topp=cfg.param.top_p)
            elif cfg.select.task == "multiple-choice":
                runner.run_mc_self_consistency(is_api=is_api, dataset=dataset,
                                               sprompt=cfg.select.start_prompt, iprompt=cfg.select.iterate_prompt,
                                               num_iters=cfg.param.iteration, sampling=cfg.param.sampling,
                                               temp=cfg.param.temp, topk=cfg.param.top_k, topp=cfg.param.top_p)
        else:
            if cfg.select.task == "generation":
                runner.run_generation(is_api=is_api, dataset=dataset, num_iters=cfg.param.iteration,
                                      sprompt=cfg.select.start_prompt, iprompt=cfg.select.iterate_prompt,
                                      sampling=cfg.param.sampling, temp=cfg.param.temp, topk=cfg.param.top_k, topp=cfg.param.top_p)
            elif cfg.select.task == "multiple-choice":
                runner.run_multiple_choice(is_api=is_api, dataset=dataset, num_iters=cfg.param.iteration,
                                           sprompt=cfg.select.start_prompt, iprompt=cfg.select.iterate_prompt,
                                           sampling=cfg.param.sampling, temp=cfg.param.temp, topp=cfg.param.top_p)

if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="main_config", node=MainConfig)
    main()
