import numpy as np
import re

class MCAnswerExtract:

    def find_match(self, response):
        patterns = [
            (r"([Tt]he (final )?(correct )?(option|answer) is[:\s]*\n*\(([A-Z])\))", 4),
            (r"(I must answer \(([A-Z])\))", 1),
            (r"(the (correct )?answer to (your|the)? question is( option)? \(([A-Z])\))", 4), # should be check later
            (r"(option \(([A-Z])\) is the correct answer.)", 1),
            (r"(the answer is option \(([A-Z])\))", 1),
            (r"([Tt]he (correct )?(answer|option)( to the question)? is[:\s]*\n*([A-Z]))", 4),
            (r"(is[:\s]*\n*([A-Z]))", 1),
            (r"(answer[:\s]*\n*([A-Z]))", 1),
            (r"option[:\s]*\n*\(?([A-Z])\)?", 1),
            (r"([:\s]*\n*([A-Z]))", 1),
        ]
        for re_pattern, match_index in patterns:
            matches = re.findall(re_pattern, response)
            if matches:
                return matches, match_index
        return None, None
    
    def search_match(self, response):
        patterns = [(r"The final answer is:\s*(.*)", 1), (r"the final answer is:\s*(.*)", 1), (r"Final Answer is:\s*(.*)", 1), 
                    (r"Final answer is:\s*(.*)", 1), (r"Final Answer:\s*(.*)", 1), (r"Final Answer is:\*\*\s*(.*)", 1), 
                    (r"Final Answer:\*\*\s*(.*)", 1), (r"Final answer is:\*\*\s*(.*)", 1),
                    (r"([Tt]he (final )?(correct )?(option|answer) is[:\s]*\n*\(([A-Z])\))", 5),
                    (r'the (correct )?answer to the question ".*?" is[:\s]*\n*([A-Z]):', 2),
                    (r"I must answer \(([A-Z])\)", 1),
                    (r"the (correct )?answer to (your|the)? question is( option)? \(([A-Z])\)", 4),
                    (r"option \(([A-Z])\) is the correct answer.", 1),
                    (r"the answer is option \(([A-Z])\)", 1),
                    (r"the (correct )?(answer|option)( to the question)? is[:\s]*\n*([A-Z])", 4),
                    (r"answer[:\s]*\n*([A-Z])", 1),
                    (r"option[:\s]*\n*\(?([A-Z])\)?", 1),
                    (r"([A-Z]):\s", 1)
                ]
        
        for re_pattern, match_index in patterns:
            match = re.search(re_pattern, response)
            if match:
                return match.group(match_index), re_pattern  # Return extracted letter
        return response, "NoPattern"

    def extract_options(self, matches: list, match_index: int):
        extracted_options = [match[match_index] for match in matches]
        extracted_options = np.unique(extracted_options).tolist()
        return extracted_options
    
    # Function to extract unique letters in parentheses
    def extract_unique_letter(self, response):
        matches = re.findall(r"\(([A-Z])\)", response)
        unique_letters = set(matches)
        return unique_letters