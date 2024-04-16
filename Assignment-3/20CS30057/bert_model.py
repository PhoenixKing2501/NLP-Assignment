import torch.nn as nn
from transformers import RobertaForSequenceClassification


# BERT Model
class BERT_Model(nn.Module):
    def __init__(self, model_name, output_dim):
        super(BERT_Model, self).__init__()
        self.bert = RobertaForSequenceClassification.from_pretrained(
            model_name,
            num_labels=output_dim,
            problem_type="single_label_classification"
        )

    def forward(self, **inputs):
        return self.bert(**inputs)  # type: ignore
