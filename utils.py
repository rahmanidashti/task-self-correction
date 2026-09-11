
class Util:

    def get_file_name(self, model, method, task, start_prompt, iterate_prompt, iteration, sampling, temp, top_k, top_p):
        return f"{model}_method-{method}_task-{task}_sp-{start_prompt}_ip-{iterate_prompt}_iter-{iteration}_sampling-{sampling}_temp-{temp}_topk-{top_k}_topp-{top_p}"