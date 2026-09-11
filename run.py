from tqdm import tqdm

from prompt import Prompt
from generation import Generation
from mc_answer_extract import MCAnswerExtract
import json
import re
from collections import Counter
import random

class Run:

    def __init__(self, generator: Generation, result_file):
        self.prompt = Prompt()
        self.result_file = result_file
        self.start_prompts = self.prompt.load_prompts(file_path="prompts/start_prompts.json")
        self.repeat_prompts = self.prompt.load_prompts(file_path="prompts/iterate_prompts.json")
        self.generator = generator
        self.mc_answer_extract = MCAnswerExtract()

    def run_generation(self, is_api, dataset, sprompt: str, iprompt: str, num_iters: int, sampling: bool, temp: float, topk: float, topp: float):
        output = dict()
        for index, eachline in enumerate(tqdm(dataset, total=len(dataset))):
            responses = dict()
            question = eachline['question']
            prefix_prompt = self.prompt.get_prompt_prefix()
            start_prompt = self.start_prompts[sprompt].copy()
            start_prompt["content"] = self.prompt.get_formatted_prompt(start_prompt=start_prompt["content"], qs=question)
            initial_messages = self.prompt.add_prompt(prefix_prompt=prefix_prompt, prompt_body=start_prompt)
            initial_response = self.generator.generate(is_api=is_api, prompt=initial_messages, sampling=sampling, temperature=temp, top_k=topk, top_p=topp)
            responses["iter_0"] = initial_response
            iteration_response, iteration_messages = initial_response, initial_messages
            repeat_prompt = self.repeat_prompts[iprompt].copy()
            repeat_prompt["content"] = self.prompt.get_formatted_prompt(start_prompt=repeat_prompt["content"], qs=question)
            for i in range(num_iters):
                assistant_conv = {"role": "assistant", "content": iteration_response}
                iteration_messages = self.prompt.add_prompt(prefix_prompt=iteration_messages, prompt_body=assistant_conv)
                iteration_messages = self.prompt.add_prompt(prefix_prompt=iteration_messages, prompt_body=repeat_prompt)
                iteration_response = self.generator.generate(is_api=is_api, prompt=iteration_messages, sampling=sampling, temperature=temp, top_k=topk, top_p=topp)
                responses[f"iter_{i+1}"] = iteration_response
            output[str(index)] = {"question": question, "answers": responses}
            self.result_file.seek(0)
            self.result_file.truncate()
            json.dump(output, self.result_file, indent=4)
            self.result_file.flush()

    def run_multiple_choice(self, is_api, dataset, sprompt: str, iprompt: str, num_iters: int = 5, sampling: bool = True, temp: float = 0.1, topp: float = 0.95):
        output = dict()
        for index, eachline in enumerate(tqdm(dataset, total=len(dataset))):
            responses = dict()
            question = eachline['question']
            choices = eachline['mc1_targets']['choices']
            prefix_prompt = self.prompt.get_prompt_prefix()
            start_prompt = dict(self.start_prompts[sprompt])
            start_prompt["content"] = self.prompt.get_mc_formatted_prompt(prompt=start_prompt["content"],
                                                                          qs=question,
                                                                          choices=choices)
            ## adding start prompt to prefix prompt
            initial_messages = self.prompt.add_prompt(prefix_prompt=prefix_prompt, prompt_body=start_prompt)
            initial_response = self.generator.generate(is_api=is_api, prompt=initial_messages, sampling=sampling, temperature=temp, top_p=topp)
            responses["iter_0"] = initial_response
            iteration_response, iteration_messages = initial_response, initial_messages
            ## getting iterative prompt
            repeat_prompt = dict(self.repeat_prompts[iprompt])
            repeat_prompt["content"] = self.prompt.get_mc_formatted_prompt(prompt=repeat_prompt["content"],
                                                                           qs=question,
                                                                           choices=choices)
            for num_iter in range(num_iters):
                assistant_conv = {"role": "assistant", "content": iteration_response}
                iteration_messages = self.prompt.add_prompt(prefix_prompt=iteration_messages, prompt_body=assistant_conv)
                iteration_messages = self.prompt.add_prompt(prefix_prompt=iteration_messages, prompt_body=repeat_prompt)
                iteration_response = self.generator.generate(is_api=is_api, prompt=iteration_messages, sampling=sampling, temperature=temp, top_p=topp)
                responses[f"iter_{num_iter+1}"] = iteration_response
            output[str(index)] = {"question": question, "answers": responses}
            self.result_file.seek(0)  # Move cursor to the beginning of the file
            self.result_file.truncate()  # Remove previous content
            json.dump(output, self.result_file, indent=4)  # Write new content
            self.result_file.flush()  # Ensure the data is immediately written to the file

    def run_gen_self_consistency(self, is_api, dataset, sprompt: str, iprompt: str, num_iters: int, sampling: bool, temp: float, topk: float, topp: float):
        output = dict()
        for index, eachline in enumerate(tqdm(dataset, total=len(dataset))):
            eval_responses = list()
            responses = dict()
            question = eachline['question']
            prefix_prompt = self.prompt.get_prompt_prefix()
            start_prompt = dict(self.start_prompts[sprompt])
            start_prompt["content"] = self.prompt.get_formatted_prompt(start_prompt=start_prompt["content"], qs=question)
            initial_messages = self.prompt.add_prompt(prefix_prompt=prefix_prompt, prompt_body=start_prompt)
            for num_iter in range(num_iters):
                reasoning_response = self.generator.generate(is_api=is_api, prompt=initial_messages, sampling=sampling, temperature=temp, top_k=topk, top_p=topp)
                eval_responses.append(reasoning_response)
            # do the evaluation with the generated answers to find the most consistent answer
            initial_response = self.universal_eval(is_api, question, eval_responses, sampling, temp, topk, topp)
            # each iteration stores [selected response, sampled responses]
            responses["iter_0"] = [initial_response, eval_responses]
            iteration_response, iteration_messages = initial_response, initial_messages
            repeat_prompt = dict(self.repeat_prompts[iprompt])
            repeat_prompt["content"] = self.prompt.get_formatted_prompt(start_prompt=repeat_prompt["content"], qs=question)
            for i in range(num_iters):
                assistant_conv = {"role": "assistant", "content": iteration_response}
                iteration_messages = self.prompt.add_prompt(prefix_prompt=iteration_messages, prompt_body=assistant_conv)
                iteration_messages = self.prompt.add_prompt(prefix_prompt=iteration_messages, prompt_body=repeat_prompt)
                ## generating itertive response
                eval_responses = []
                for num_iter in range(num_iters):
                    reasoning_response = self.generator.generate(is_api=is_api, prompt=iteration_messages, sampling=sampling, temperature=temp, top_k=topk, top_p=topp)
                    eval_responses.append(reasoning_response)
                iteration_response = self.universal_eval(is_api, question, eval_responses, sampling, temp, topk, topp)
                responses[f"iter_{i+1}"] = [iteration_response, eval_responses]
            output[str(index)] = {"question": question, "answers": responses}
            self.result_file.seek(0)  # Move cursor to the beginning of the file
            self.result_file.truncate()  # Remove previous content
            json.dump(output, self.result_file, indent=4)  # Write new content
            self.result_file.flush()  # Ensure the data is immediately written to the file

    def universal_eval(self, is_api, question: str, responses: list, sampling: bool, temp: float, topk: float, topp: float):
        formatted_responses = "\n".join([f"Response {num_iter}: {response}" for num_iter, response in enumerate(responses)])
        eval_prompt = f"I have generated the following responses to the question: {question}\n\n" \
                    + f"{formatted_responses}\n\n" \
                    + "Evaluate these responses.\n" \
                    + "Select the most consistent response based on majority consensus.\n" \
                    + "Start your answer with 'The most consistent response is Response X:' (without quotes)."
        prompt = [
            {"role": "system", "content": "You are an expert evaluator."},
            {"role": "user", "content": f"{eval_prompt}"}
            ]
        consistent_response = self.generator.generate(is_api=is_api, prompt=prompt, sampling=sampling, temperature=temp, top_k=topk, top_p=topp)
        consistent_response = self.get_consistent_response(consistent_response)
        if consistent_response["number"] is not None:
            if 0 <= int(consistent_response["number"]) <= (len(responses) - 1):
                return responses[int(consistent_response["number"])]
        if consistent_response["text"] is not None:
                return consistent_response["text"]
        else:
            print("Both number and text are none!")
            return f"Random selection response is: {random.choice(responses)}"

    def get_consistent_response(self, response):
        """
        Extracts the response number and text from a sentence.

        Args:
            sentence (str): The input sentence.

        Returns:
            dict: A dictionary with keys 'number' and 'text'.
        """
        patterns = [r"Response\s+(\d+)(:?\s*(.*))?", r".*[^a-zA-Z0-9]?\s*Response\s*(\d+)[^a-zA-Z0-9]*(.*)"]
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                response_number = match.group(1) if match.group(1) else None
                response_text = match.group(3).strip() if match.group(3) else None
                return {"number": response_number, "text": response_text}
        return {"number": None, "text": None}

    def run_mc_self_consistency(self, is_api, dataset, sprompt: str, iprompt: str, num_iters: int, sampling: bool, temp: float, topk: float, topp: float):
        output = dict()
        for index, eachline in enumerate(tqdm(dataset, total=len(dataset))):
            eval_responses = list()
            responses = dict()
            question = eachline['question']
            choices = eachline['mc1_targets']['choices']
            prefix_prompt = self.prompt.get_prompt_prefix()
            start_prompt = dict(self.start_prompts[sprompt])
            start_prompt["content"] = self.prompt.get_mc_formatted_prompt(prompt=start_prompt["content"],
                                                                          qs=question,
                                                                          choices=choices)
            initial_messages = self.prompt.add_prompt(prefix_prompt=prefix_prompt, prompt_body=start_prompt)
            for num_iter in range(num_iters):
                reasoning_response = self.generator.generate(is_api=is_api, prompt=initial_messages, sampling=sampling, temperature=temp, top_k=topk, top_p=topp)
                eval_responses.append(reasoning_response)
            # majority voting to pick the most consistent response
            initial_response = self.majority_voting_eval(eval_responses)
            # each iteration stores [selected option, sampled responses]
            responses["iter_0"] = [initial_response, eval_responses]
            iteration_response, iteration_messages = initial_response, initial_messages
            repeat_prompt = dict(self.repeat_prompts[iprompt])
            repeat_prompt["content"] = self.prompt.get_mc_formatted_prompt(prompt=repeat_prompt["content"],
                                                                           qs=question,
                                                                           choices=choices)
            for i in range(num_iters):
                assistant_conv = {"role": "assistant", "content": iteration_response}
                iteration_messages = self.prompt.add_prompt(prefix_prompt=iteration_messages, prompt_body=assistant_conv)
                ## adding iterative prompt to start prompt
                iteration_messages = self.prompt.add_prompt(prefix_prompt=iteration_messages, prompt_body=repeat_prompt)
                ## generating itertive response
                eval_responses = []
                for num_iter in range(num_iters):
                    reasoning_response = self.generator.generate(is_api=is_api, prompt=iteration_messages, sampling=sampling, temperature=temp, top_k=topk, top_p=topp)
                    eval_responses.append(reasoning_response)
                iteration_response = self.majority_voting_eval(eval_responses)
                responses[f"iter_{i+1}"] = [iteration_response, eval_responses]
            output[str(index)] = {"question": question, "answers": responses}
            self.result_file.seek(0)  # Move cursor to the beginning of the file
            self.result_file.truncate()  # Remove previous content
            json.dump(output, self.result_file, indent=4)  # Write new content
            self.result_file.flush()  # Ensure the data is immediately written to the file

    def majority_voting_eval(self, responses: list):
        options = list()
        for response_id, response in enumerate(responses):
            match = None
            try:
                match, pattern = self.mc_answer_extract.search_match(response=response)
                unique_letters = self.mc_answer_extract.extract_unique_letter(response=match)
                if len(unique_letters) == 1:
                    match = list(unique_letters)[0]
            except Exception as e:
                print("#Error #1: ", e)
                print(f"\nResponse ID: {response_id}, Resposen: {response}, Match: {match}")
            if not match:
                match = "NoMatch"
                print(f"\nResponse ID: {response_id}, Respose: {response}, Match: {match}")
            options.append(match)
        return Counter(options).most_common(1)[0][0]
