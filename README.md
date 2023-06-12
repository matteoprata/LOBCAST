# LOBCAST for Stock Trend Forcasting 

LOBCAST is a Python-based framework developed for stock market trend forecasting using LOB data. 
LOBCAST is an open-source framework that enables users to test DL models for the Stock Price Trend Prediction (SPTP) task. 

The framework provides data **pre-processing** functionalities, which include **normalization**, **splitting**, and **labelling**.
LOBCAST also offers a comprehensive training environment for DL models implemented in PyTorch Lightning. 
It integrates interfaces with the popular hyperparameter tuning framework WANDB, which allows users to tune and optimize 
model performance efficiently. The framework generates detailed reports for the trained models, including performance 
metrics regarding the learning task (F1, Accuracy, Recall, etc.). LOBCAST supports backtesting for profit analysis, 
utilizing the Backtesting.py external library. This feature enables users to assess the profitability of their models in
simulated trading scenarios.

LOBCAST will soon support (i) training and testing with different LOB representations, and (ii) test on adversarial
perturbations to evaluate the representations' robustness. We believe that LOBCAST, along with the advancements in DL 
models and the utilization of LOB data, has the potential to improve the state of the art on trend forecasting in the 
financial domain.

### Getting Started
#### Installation 
1. Download the LOBCAST source code, either directly from GitHub or with git:
    ```
    git clone https://github.com/xxx/xxx
    ```
2. Install the dependencies:
    ```
    pip install -r requirements.txt
    ```

#### Execution
###### Training
Train the models providing an execution plan as in `src.main_run_fi.py`, similarly for `src.main_run_lob.py`:

```
EXE_PLAN = {
    cst.Servers.ANY: [
        (cst.Models.MLP,      {'k': [cst.FI_Horizons.K5], 'seed': [500]}),
        (cst.Models.BINCTABL, {'k': [cst.FI_Horizons.K1], 'seed': [0, 1]})
    ]
}
experiment_fi(EXE_PLAN)
```

An execution list of all the desired sequential lunches to do.
Mapping the running servers to: the models to lunch, relative seeds and horizons. The snipped above will launch sequentially training on 
_(1) MLP model, horizon 5, seed 500; (2) BINCTABL model, horizon 1, seed 0; (3) BINCTABL model, horizon 1, seeds 1._

The output of a simulation will be the Pytorch model saved in the directory `data.saved_models` + the name of the simulation 
input of `experiment_fi`, if `None` is passed, then current date and time will be used as the name of the folder.

###### Testing
From `src.main_testing.py` it is possible to lunch testing of the saved models in `data.saved_models`, results of the analysis 
are saved in .json files containing the following desirable statistics:

```
"testing_FI_f1": 0.7122599109744749,
"testing_FI_f1_w": 0.7549878538983881,
"testing_FI_precision": 0.7256973918648287,
"testing_FI_precision_w": 0.7538648925354426,
"testing_FI_recall": 0.7019571806837437,
"testing_FI_recall_w": 0.7589772621410708,
"testing_FI_accuracy": 0.7589772621410708,
"testing_FI_mcc": 0.5818397649926562,
"testing_FI_cohen-k": 0.5801910886880164,
"testing_loss": 1370.2157062944025,
"cm": [[], [], []]
[...]
```

###### Plotting
From `src.metrics.metrics_plotting.py` it is possible to generate the plots in the paper. 

A much stabler version is in progress and will be released for the camera ready. 