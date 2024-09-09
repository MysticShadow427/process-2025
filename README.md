# process-2025

## To-do 
- how to integrate bert without positional embeddings and token embeddings - *done*
- miniGPT from karpathy and use it as decoder in asr_task, we need to add some blank token to predict the blank stuff, we need to preporocess the text data likewise.
- include various features from the papers and use them in cross attention. - *done*, the thing is that we can either do something pooling with all these features rather than fusing with cross attention
    - need to modify dataloader - *done* and 
    - model architecture (explicitly add 1d cnn layers there). 
- also for each feature we need to add a 1d cnn layer to adjust the feature dimension. - *done*
- whether we need to unsqueeze check after running a batch coz we need 3d features.
- check the input dimension of each feature to add the parameter in 1d cnn.
- add phonetic feature loading script.
- need to see shape issues by running a batch :)
- fbank features 512 karne ka try karo varna usko bhi projection se paas karna hoga
- max lenght of text data
- think more multitask learning objectives : mainly focusing on semantic fluency and phonetic fluency.
- come up with a loss function that makes text embeddings and speech embeddings similar. - *done*
- does adding layernorm after all conformer blocks and gated cross attention blocks makes life good?
- need to add a script to preprocess text data and audio data, we need to check distributions of of each of the lenghts. -*done*
- currently for phoneme features i am using this `vitouphy/wav2vec2-xls-r-300m-phoneme` but what we can do is to use `facebook/wav2vec2-xlsr-53-espeak-cv-ft` to generate phonetic transcriptions and then use `vinai/xphonebert-base` to produce some speech features.
- what augmentations can we apply together, what effect do they have if aplied together.
- also we need to check whether 1d cnn feture projections are better or onoe layer nueral nets are better.


`512` should be the dimension and before sending to bert we need to project it to 768.

<hr>
## Training a BERT on the given text data

Organize the dataset in the csv, one column of text and another column of the corresponding label.
<hr>
To load the model after training - 
```python
from transformers import AutoModel
model = AutoModel.from_pretrained("/content/drive/MyDrive/process2025/bert_model")