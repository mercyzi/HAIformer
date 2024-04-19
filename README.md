## HAIformer

Code for our submission: ECAI24_paper_249 

### Data Source

MDD: MTDiag: an effective multi-task framework for automatic diagnosis (AAAI 2023)

mz10: DxFormer: a decoupled automatic diagnostic system based on decoder--encoder transformer with dense symptom representations (Bioinformatics 2022)

dxy: End-to-end knowledge-routed relational dialogue system for automatic diagnosis (AAAI 2019)

muzhi (mz4): Task-oriented dialogue system for automatic diagnosis (ACL 2018)

### Environment Setup:
```yaml
conda env create --name HAIformer --file environment.yml
```

### Train Symptom-Graph
```yaml
# dataset: Either MDD, mz10, dxy, muzhi.
python train_gnn.py \
-data MDD 
```

### Pretrain
```yaml
python pretrain.py \
-data MDD 
```

### Train
```yaml
python train.py \
-d MDD 
```


