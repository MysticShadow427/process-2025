# process-2025

## To-do 
- how to integrate bert without positional embeddings and token embeddings - *done*
- miniGPT from karpathy and use it as decoder in asr_task
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

`512` should be the dimension and before sending to bert we need to project it to 768.