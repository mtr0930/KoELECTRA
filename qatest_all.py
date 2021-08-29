from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import AutoConfig
import torch
import argparse
import json
import sys
from collections import OrderedDict
sys.stdout = open('output.txt', 'w')

class QuestionAnswering():
    """I-manual question answering class.
    Attributes:
        model (AutoModelForQuestionAnswering): transformers model.
        tokenizer (AutoTokenizer): transformers tokenizer.
        pipeline (QuestionAnsweringPipeline): transformers pipeline.
    """

    def __init__(self, pretrained_model_name_or_path=None):
        """Inits QuestionAnswering, and load a pre-trained model.

         Args:
            pretrained_model_name_or_path (str):
                Pretrained huggingface model name or model path.
        """

        self.model = AutoModelForQuestionAnswering.from_pretrained(
            pretrained_model_name_or_path,
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path)

    def __call__(self, question, context):
        """Returns answer(s) for given question, context.
        Args:
            question: question of interest.
            context: context where appropriate answer can be found.
        Returns:
            A dictionary containing answer and other informations.
        """

        inputs = self.tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
        input_ids = inputs["input_ids"].tolist()[0]

        text_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        answer_start_scores, answer_end_scores = self.model(**inputs, return_dict=False)

        answer_start = torch.argmax(answer_start_scores)  # Get the most likely beginning of answer with the argmax of the score
        answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score
        
        answer = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

        return { "question" : question,
                 "answer" : answer,
                 "input_ids" : input_ids,
                 "answer_start_scores" : answer_start_scores,
                 "answer_end_scores" : answer_end_scores,
                 }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True
    )


    args = parser.parse_args()
    file_data = OrderedDict()
    with open("data/i-manual_changed.json", "r", encoding="utf-8") as reader:
        input_data = json.load(reader)["data"]    
    
    res_c = open("epoch-100-predictions/answers.txt", "w")

    qa = QuestionAnswering(args.model_name_or_path)

    for i in range(0, len(input_data)):
        data = input_data[i]

        context = data['paragraph']
        qas = data['qas']

        for k in range(0, len(qas)):
                question = qas[k]['question_c']
                answer = qa(question=question, context=context)
                text = answer['answer']
                tmp_id = qas[k]['id']
                file_data[tmp_id] = text
    with open('epoch-100-predictions/ckpt3600_predictions.json', 'a+') as make_file:
        json.dump(file_data, make_file, indent="\t" )
if __name__ == "__main__":
    main()
