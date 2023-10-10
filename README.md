# LOBCAST for Stock Price Trend Forecasting

🟢 🚀 _Get ready for LOBCAST 2.0 – Unleashing automation for the SPTP task, a huge leap in efficiency!_ 🌟  🟢

📌 _Scheduled for release on 01-11-2023 #LOBCASTv2_

LOBCAST is a Python-based open-source framework developed for stock market trend forecasting using Limit Order Book (LOB) data. The framework enables users to test deep learning models for the task of Stock Price Trend Prediction (SPTP). It is the official repository for [LOB-Based Deep Learning Models for Stock Price Trend Prediction: A Benchmark Study](https://arxiv.org/abs/2308.01915).



## Key Features
- Data pre-processing functionalities including **normalization**, **splitting**, and **labeling**.
- Implementation of 15 SOTA deep learning models for LOB data located in the src/models DIR. 
- Comprehensive training and testing environment for deep learning models using LOB data implemented in PyTorch Lightning.
- Integration with the hyperparameter tuning framework WANDB for efficient model performance optimization.
- Generation of detailed reports for trained models, including performance metrics such as F1, Accuracy, Recall, etc.
- Support for backtesting and profit analysis using the Backtesting.py external library.
- Future support for training and testing with different LOB representations.
- Planned support for testing adversarial perturbations to evaluate the robustness of representations.


## Getting Started
### Installation 
1. Download the LOBCAST source code, either directly from GitHub or with git:
    ```
    git clone https://github.com/matteoprata/LOBCAST
    ```
2. Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```

### Execution
#### Training
To train the models, provide an execution plan in `src.main_run_fi.py` for tests over FI-2010 dataset or in `src.main_run_lob.py` for tests over LOBSTER data. 
Here's an example execution plan:

```
EXE_PLAN = {
    cst.Servers.ANY: [
        (cst.Models.MLP,      {'k': [cst.FI_Horizons.K5], 'seed': [500]}),
        (cst.Models.BINCTABL, {'k': [cst.FI_Horizons.K1], 'seed': [0, 1]})
    ]
}
experiment_fi(EXE_PLAN)
```

The execution plan specifies the models to be trained, along with their respective seeds and horizons. 
The above example will sequentially launch training for:

- MLP model with horizon 5 and seed 500.
- BINCTABL model with horizon 1 and seed 0.
- BINCTABL model with horizon 1 and seed 1.

The trained models will be saved in the `data.saved_models` directory with the name of the simulation input of experiment_fi function. 
If no name is provided, the current date and time will be used as the folder name.

Make sure to have the [FI-2010 dataset](https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649/data) downloaded in `data.FI-2010.BenchmarkDatasets`. Now you can run the execution plan as:
```
python -m src.main_run_fi
```

#### Testing
To perform testing on the saved models in `data.saved_models`, use `src.main_testing.py`. 
The results of the analysis will be saved in .json files, which include the following statistics:

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

#### Plotting
To generate plots from the data in the JSON files, you can use `src.main_metrics_plots.py`. This will help in visualizing the results as mentioned in the paper.

Please note that a more stable version is currently in progress and will be released soon.

## Reference
If you find the code useful for your research, please consider citing
```bib
@misc{prata2023lobbased,
      title={LOB-Based Deep Learning Models for Stock Price Trend Prediction: A Benchmark Study}, 
      author={Matteo Prata and Giuseppe Masi and Leonardo Berti and Viviana Arrigoni and Andrea Coletta and Irene Cannistraci and Svitlana Vyetrenko and Paola Velardi and Novella Bartolini},
      year={2023},
      eprint={2308.01915},
      archivePrefix={arXiv},
      primaryClass={q-fin.TR}
}
```


## Acknowledgments
LOBCAST was developed by [Matteo Prata](https://github.com/matteoprata), [Giuseppe Masi](https://github.com/giuseppemasi99), [Leonardo Berti](https://github.com/LeonardoBerti00), [Andrea Coletta](https://github.com/Andrea94c), [Irene Cannistraci](https://github.com/icannistraci), Viviana Arrigoni. 
