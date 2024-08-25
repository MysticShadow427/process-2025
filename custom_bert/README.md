# Training a BERT on the given text data

Organize the dataset in the csv, one column of text and another column of the corresponding label.
<hr>
To load the model after training - 
```python
from transformers import AutoModel
model = AutoModel.from_pretrained("/content/drive/MyDrive/process2025/bert_model")
```