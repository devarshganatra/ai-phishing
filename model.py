import torch
import torch.nn as nn
from transformers import DistilBertModel

class PhishingDistilBERT(nn.Module):
    def __init__(self, num_extra_features=0):
        super(PhishingDistilBERT, self).__init__()
        # Load pre-trained DistilBERT
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        # DistilBERT hidden size is 768
        self.hidden_size = self.distilbert.config.hidden_size
        self.num_extra_features = num_extra_features
        
        # Classifier head
        # We concatenate the [CLS] token embedding (768) with our extra features
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size + self.num_extra_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1) # Binary classification output
        )
        
    def forward(self, input_ids, attention_mask, extra_features=None):
        # Pass through DistilBERT
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Extract the [CLS] token representation (first token of the sequence)
        # For DistilBERT, the first token of the last hidden state is used as the pooled output
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Concatenate handcrafted features if provided
        if self.num_extra_features > 0 and extra_features is not None:
            combined_output = torch.cat((cls_output, extra_features), dim=1)
        else:
            combined_output = cls_output
            
        # Pass through classification head
        logits = self.classifier(combined_output)
        
        return logits
