## pytorch_ner
`Classic NER Model Implementations based on Pytorch`

### Read the Docs

### Architecture

```
├── dataset # ner dataset
├── docs # rst docs
├── examples # speicific experiments
│   ├── cluener_bert_crf
│   ├── cluener_bert_softmax
│   ├── cluener_lstm_crf
│   └── cluener_lstm_softmax # cluener + lstm_softmax
│       ├── __init__.py
│       ├── cluener_lstmsoftmax_exp.py # experiment start
│       └── results
│           ├── Exp_LR0.002_Batch32_LossSeqWiseCrossEntropyLoss
│           │   ├── Cluener_LstmSoftmax_Experiment_20220223_1338.png
│           │   ├── Epoch_Statstics_Time20220223_1338.csv
│           │   └── predicts
│           └── cluener_bioes.mlb # cluener multi label binarizer, encoded with bioes 
├── log
├── ner # source code
│   ├── callbacks
│   ├── config.py 
│   ├── dataloaders 
│   │   ├── base_loader.py
│   │   └── cluener_loader.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── modeling_bert_crf.py
│   │   ├── modeling_bert_softmax.py
│   │   ├── modeling_idcnn_softmax.py
│   │   ├── modeling_lstm_crf.py
│   │   └── modeling_lstm_softmax.py
│   ├── modules
│   │   ├── __init__.py
│   │   ├── analyzer.py
│   │   ├── eval_metrics.py
│   │   ├── ner_loss.py
│   │   ├── trainer.py
│   │   ├── utils.py
│   │   └── visualizer.py
│   └── pretrain_models
│       └── bert-base-chinese
├── test 
```

[pytorch_ner's document]()

### First Experiment
```angular2html
# fit the lstm_softmax with cluener dataset

>>> python examples/cluener_lstm_softmax/cluener_lstmsoftmax_exp.py
```

### Models

- [IDCNN]()

- [LSTM + Softmax](https://github.com/bannima/pytorch_ner/tree/master/ner/models/lstm_softmax)

- [LSTM + CRF]()

- [Bert + Softmax]()

- [Bert + CRF ]()

- [Bert + LSTM + CRF]()

- [Bert + Span]()

- [MRC]()

### NER Datasets
1. zh 
   - [cluener](https://www.cluebenchmarks.com/introduce.html)

   - [msra]()

   - [pd1998]()
   
   - [pd2014]()

2. en
   - [conll03]()

   - [ace2004]()
   
   - [ace2004]()

### Experiment Results



### References

- [流水的NLP铁打的NER：命名实体识别实践与探索](https://zhuanlan.zhihu.com/p/166496466)
- [Named entity recognition | NLP-progress](https://nlpprogress.com/english/named_entity_recognition.html)
- [pytorch-crf](https://github.com/kmkurn/pytorch-crf)

