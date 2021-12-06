from transformers import BartForConditionalGeneration, T5ForConditionalGeneration
from transformers import BartForSequenceClassification, BertForSequenceClassification, RobertaForSequenceClassification


def get_model(model_name: str, task_name: str):
    if task_name == 'cls':
        if 'bart' in model_name:
            return BartForSequenceClassification.from_pretrained(model_name)
        elif 'roberta' in model_name:
            return RobertaForSequenceClassification.from_pretrained(model_name)
        elif 'bert' in model_name:
            return BertForSequenceClassification.from_pretrained(model_name)
        else:
            raise KeyError("expected bart, bert, roberta, but got other")

    elif task_name == 'seq2seq':
        if 'bart' in model_name:
            return BartForConditionalGeneration.from_pretrained(model_name)
        elif 't5' in model_name:
            return T5ForConditionalGeneration.from_pretrained(model_name)
        else:
            raise KeyError("expected bart, t5, but got other")

    else:
        raise KeyError("expected cls, seq2seq, but got other")
