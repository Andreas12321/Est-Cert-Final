# Estimating Certainty of Deep Learning
This repository contains the code for our project "Estimating Certainty of Deep Learning". 

The dependencies necessary to run the code are listed in the file _requirements.txt_.

To run the code specify the desired network, dataset and method to apply to the file _runme.py_. See the following example:
```
python runme.py --network vgg16 --dataset mnist --method baseline
```

The available networks are the following:

- LeNet-5 (lenet)
- VGG-16 (vgg16)
- ResNet-50 (resnet) (not available for SWAG)
- MLP (only available for the ensemble methods)

The available dataset is the following:
- MNIST (mnist)

The calibration methods that can be applied to the networks are:
- Baseline (baseline)
- Temperature scaling (temp_scale)
- Ensemble methods (ensemble)

To apply SWAG to a network first create the following directories: 
- plots
- weights

Specify the training and test size and preprocess the data by running the file swag_preprocess_data.py_. See the following example:
```
python swag_preprocess_data.py --save_path data_processed --data_set MNIST --train_size 5000 --test_size 40000 
```

To train and test a network specify the hyperparameters and run the files _train_swag.py_ and _test_swag.py_. See the following example. 
```
python train_swag.py --data_path data_processed/ --save_param_path weights/ --save_checkpoint_path checkpoints/ --save_plots_path plots/ --epochs 120 --swag_start_epochs 61 --K 20 --network vgg16
python test_swag.py --data_path data_processed/ --load_param_file weights/swag_params.npz --mode swag  --network vgg16
```

To create the reliability diagrams specify the desired network and run the file plot.py. See the following example:
```
python plot.py --network vgg16
```

## References:
### Original Papers:
- Chuan Guo et al. “On Calibration of Modern Neural Networks”. In:CoRRabs/1706.04599 (2017). arXiv:1706. 04599. url: http://arxiv.org/abs/1706.04599.
- Balaji  Lakshminarayanan,  Alexander  Pritzel,  and  Charles  Blundell.“Simple  and  Scalable  Predictive  Uncertainty  Estimation  using  DeepEnsembles”.  In:Advances  in  Neural  Information  Processing  Systems30.  Ed.  by  I.  Guyon  et  al.  Curran  Associates,  Inc.,  2017,  pp.  6402–6413.url:http://papers.nips.cc/paper/7219-simple-and-scalable-predictive-uncertainty-estimation-using-deep-ensembles.pdf.
- Wesley Maddox et al. “A Simple Baseline for Bayesian Uncertainty inDeep Learning”. In:arXiv preprint arXiv:1902.02476(2019).

### Implementations:
- https://github.com/timgaripov/swa/
- https://github.com/DD2412-Final-Projects/swag-reproduced/tree/authors-config
