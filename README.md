# NER task
It currently supports only English.

## DEPENDENCIES

```bash
pip install -r requirements.txt
```
## Model description
<img src="./ner_lstm_diagram.svg" width="300">


## LAUNCHING
(optional)
 ```bash
jupyter notebook 01_train_models.ipynb
```

Main program
```bash
python infer_model.py
```

## Script outputs
The script `infer_model.py` is an interactive CLI program.

After launching , you will be prompted to insert a sentence. The script outputs each inferred named entity for each token.
<img src="https://github.com/user-attachments/assets/6f5fa8cd-6fc4-46bc-b0f5-a07599583028" width="500">

It includes a straightforward algorithm to reconstruct the original tokens by combining word-pieces, ensuring that each token is mapped to its corresponding entity.

<img src="https://github.com/user-attachments/assets/c9dfb280-d334-4e48-b5fb-819e7451e238" width="500">



## NOTES
`ray_gridsearch.py` was created for optimizing hyperparameters.

Always run your projects in a virtual environment.

## AUTHOR
bianca.ciobanica@student.uclouvain.be
