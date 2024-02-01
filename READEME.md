# Readme


## Run experiements
```bash
source run_conv4.sh
source run_resnet18.sh
source run_vgg16.sh
```

## Run single experiment (example)

```bash
python run.py --batch_size 32 --keep_prob 0.0 --learning_rate 0.001 --momentum 0.9 --epochs 25 --dataset imagenet --model vgg16 --seeds 1 2 3 4 5 --finetuning_size 5000 --finetuning --pre_trained_dataset imagenet --degree_of_randomness 2
```

## Plot all results

```bash
python plot.py
```

