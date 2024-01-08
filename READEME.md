# Readme

## Env setup
```bash
todo
```

## Run networks
```bash
python run.py --batch_size 32 --keep_prob 0.0 --learning_rate 0.001 --momentum 0.9 --epochs 150 --dataset cifar10 --model conv4 --seeds "1 2 3 4 5" --finetuning_size 5000 --pre_trained_dataset cifar10 --degree_of_randomness 0 --pre_trained_size 50000
python run.py --batch_size 32 --keep_prob 0.0 --learning_rate 0.001 --momentum 0.9 --epochs 150 --dataset cifar10 --model conv4 --seeds "1 2 3 4 5" --finetuning_size 5000 --pre_trained_dataset cifar10 --degree_of_randomness 1 --pre_trained_size 50000
python run.py --batch_size 32 --keep_prob 0.0 --learning_rate 0.001 --momentum 0.9 --epochs 150 --dataset cifar10 --model conv4 --seeds "1 2 3 4 5" --finetuning_size 5000 --pre_trained_dataset cifar10 --degree_of_randomness 2 --pre_trained_size 50000
python run.py --batch_size 32 --keep_prob 0.0 --learning_rate 0.001 --momentum 0.9 --epochs 150 --dataset cifar10 --model conv4 --seeds "1 2 3 4 5" --finetuning_size 5000 --pre_trained_dataset cifar10 --degree_of_randomness 4 --pre_trained_size 50000
python run.py --batch_size 32 --keep_prob 0.0 --learning_rate 0.001 --momentum 0.9 --epochs 150 --dataset cifar10 --model conv4 --seeds "1 2 3 4 5" --finetuning_size 5000 --pre_trained_dataset cifar10 --degree_of_randomness 9 --pre_trained_size 50000
```


## Run stats
```bash
python stats.py
```

## Run plotter_stats
```bash
python stats.py
```
