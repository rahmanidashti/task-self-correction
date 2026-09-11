from tqdm import tqdm
import string

class Label:

    def __init__(self):
        pass

    def get_mc1_labels(self, ds):
        alphabet = string.ascii_uppercase
        mc1_labels = dict()
        for index, eachline in enumerate(tqdm(ds, total=len(ds))):
            question = eachline['question']
            labels = eachline['mc1_targets']['labels']
            try:
                label_index = labels.index(1)
            except:
                print(index, labels)
            output = alphabet[label_index]
            mc1_labels[str(index)] = (question, output)
        return mc1_labels