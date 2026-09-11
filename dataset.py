import json
from datasets import load_dataset

class Dataset:
    def __init__(self, dataset: str = "bbeh-disambiguation-qa", task: str = "generation"):
        self.ds_name = dataset
        self.ds_task = task

    def load_data(self):
        if self.ds_task == "generation":
            ds = load_dataset(f"rahmanidashti/{self.ds_name}", "generation")['validation']
        elif self.ds_task == "multiple-choice":
            ds = load_dataset(f"rahmanidashti/{self.ds_name}", "multiple-choice")['validation']
        elif self.ds_task == "genmcq":
            # 1. Load both the generation and multiple-choice datasets
            gen_ds = load_dataset(f"rahmanidashti/{self.ds_name}", "generation")['validation']
            mc_ds = load_dataset(f"rahmanidashti/{self.ds_name}", "multiple-choice")['validation']

            # --- Safety Check (Recommended) ---
            # Ensure both datasets have the same number of rows before combining
            if len(gen_ds) != len(mc_ds):
                raise ValueError(
                    "The 'generation' and 'multiple-choice' datasets must have the same "
                    "number of rows to be combined."
                )
            
            # 2. Extract the nested 'choices' data from the multiple-choice dataset
            # We use a list comprehension to iterate through each row of mc_ds and
            # pull out the 'choices' list from the nested 'mc1_targets' dictionary.
            choices_column = [row['mc1_targets']['choices'] for row in mc_ds]

            # 3. Add the extracted list as a new column to the generation dataset
            # The .add_column() method is a clean and efficient way to do this.
            ds = gen_ds.add_column(name="choices", column=choices_column)
            
        else:
            raise ValueError(f"Unknown ds_task: {self.ds_task}")
        return ds