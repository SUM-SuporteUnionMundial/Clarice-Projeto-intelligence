import torch
from torch.nn import Linear, Softmax

class CombinedModel(torch.nn.Module):
    def __init__(self, models):
        super(CombinedModel, self).__init__()
        self.models = models
        self.combination_layer = Linear(len(models), 1)
        self.softmax = Softmax(dim=1)

    def forward(self, inputs):
        outputs = []
        for model in self.models:
            output = model(**inputs)
            outputs.append(output.logits)
        combined_outputs = torch.cat(outputs, dim=-1)
        combined_outputs = self.combination_layer(combined_outputs)
        combined_outputs = self.softmax(combined_outputs)
        return combined_outputs