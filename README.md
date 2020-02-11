# Super-Resolved Image Generation

### Working Environment
* Ubuntu 18.04 LTS
* CUDA 10.1
* GPU: RTX 2080 (8GB)

### Data
You can download data from: 
* https://kelvins.esa.int/proba-v-super-resolution/data/
* Unzip it under `${PROJECT_HOME}/data`

### Instruction
Clone the source
```
git clone git@github.com:kbu9299/sr_image.git
```

Move to the project home
```
cd sr_image
```

Install dependencies
```
pip install -r requiremnents.txt 
```

Train model
```
python src/train.py --config=config/config.json
```
* You can increase `batch_size` in `config/config.json` if your GPU has more than 8GB.
* The maximum value of `seq_len` is 8.
* Training will take several hours depending on your GPU power. 
* If you want to skip the training then you can download pre-trained models.  

View training logs with TensorboardX
```
tensorboard --logdir='logs'
```

Evaluate model
```
python src/evaluate.py --config=config/config.json --checkpoint_file=checkpoints/pretrained_batch_6_8/fus_model.pth
```

Generate a submission file
```
python src/generate_submission.py --config=config/config.json --checkpoint_file=checkpoints/pretrained_batch_6_8/fus_model.pth
```
* You can find SR images generated from the test dataset in `submission/` with one zipped file `submission.zip`.

### Pre-trained model

| Name  | Configuration | Link | GPU Memory for training |
| ------------- | ------------- |------------- |---| 
| pretrained_batch_6_8  | batch size:6, seq len: 8  | [Download](https://drive.google.com/file/d/1viRJW33LwickTLqNI1aT6rC-nFZUlU0V/view?usp=sharing)| 8GB 
| pretrained_batch_8_8 | batch size:8, seq len: 8  | [Download](https://drive.google.com/file/d/1Rs4jjV1TMeokTQDwB4yuVUWOHFq2WpCR/view?usp=sharing)  | 11GB
