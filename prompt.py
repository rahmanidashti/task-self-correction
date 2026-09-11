import json
import string

class Prompt:

    def __init__(self):
        pass

    def load_prompts(self, file_path):
        with open(file_path, "r") as file:
            return json.load(file)
        
    def get_prompt_prefix(self):
        prefix_prompt = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Answer the following questions."},
            {"role": "assistant", "content": "Sure, I'm ready to help you"}
            ]
        return prefix_prompt

    def add_prompt(self, prefix_prompt: list, prompt_body: dict):
        prefix_prompt.append(prompt_body)
        return prefix_prompt
    
    def get_formatted_prompt(self, start_prompt, qs):
        return start_prompt.format(question=qs)
    
    def get_mc_formatted_prompt(self, prompt: str, qs: str, choices: list):
        qs_options = f"{qs}\nOptions:\n"
        for index, choice in enumerate(choices):
            qs_options = qs_options + f"({string.ascii_uppercase[index]}) " + choice + ".\n"
        prompt = prompt.format(question=qs_options)
        return prompt
    
    def get_bbeh_eval_prompt(self, question: str, generated_answer: str, gold_answer: str) -> str:
        eval_messages = [
            {"role": "system", "content": "You are an expert in answer correctness evaluation."},
            {"role": "user", "content": f"Given a question, its reference answer, and a generated answer, please evaluate the correctness of the generated answer based on the question and the reference answer.\n\nHere are the question, reference answer, and generated answer:\n\n- Question: {question}\n- Reference Answer: {gold_answer}\n- Generated Answer: {generated_answer}\n\nPlease assess the correctness of the generated answer by considering the question and comparing it against the reference answer.\n- Return yes if the generated answer is completely correct, otherwise, return 'no'. The final answer must only be 'yes' or 'no', corresponding to the correctness of the generated answer."}
        ]
        return eval_messages