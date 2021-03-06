from transformers import BartForConditionalGeneration, T5ForConditionalGeneration
from transformers import BartForSequenceClassification, BertForSequenceClassification, RobertaForSequenceClassification, XLNetForSequenceClassification


def get_model(model_name: str, task_name: str):
    if task_name == 'cls':
        if 'bart' in model_name:
            model = BartForSequenceClassification.from_pretrained(model_name, num_labels=3)
        elif 'roberta' in model_name:
            model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=3)
        elif 'bert' in model_name:
            model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)
        elif 'xlnet' in model_name:
            model = XLNetForSequenceClassification.from_pretrained(model_name, num_labels=3)
        else:
            raise KeyError("expected bart, bert, roberta, but got other")
        print(model)
        return model

    elif task_name == 'seq2seq':
        if 'bart' in model_name:
            return BartForConditionalGeneration.from_pretrained(model_name)
        elif 't5' in model_name:
            return T5ForConditionalGeneration.from_pretrained(model_name)
        else:
            raise KeyError("expected bart, t5, but got other")

    else:
        raise KeyError("expected cls, seq2seq, but got other")

