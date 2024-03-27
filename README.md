# LOBCAST  Stock Price Trend Forecasting with Python

### 📈 mini-LOBCAST _(v2.0 coming up)_
README under construction.

### Installing LOBCAST 

You can install LOBCAST by cloning the repository and navigate into the directory:
```
git clone https://github.com/matteoprata/LOBCAST.git
cd LOBCAST
```

Install all the required dependencies.
```
pip install -r requirements.txt
```

Running LOBCAST locally with an MLP model and FI-2010 dataset: 
```
pip -m src.run 
```

The ```run.py``` scripts accepts the following arguments.

```
LOBCAST 
optional arguments:
  -h, --help            show this help message and exit
  --SEED 
  --DATASET_NAME 
  --N_TRENDS 
  --PREDICTION_MODEL 
  --PREDICTION_HORIZON_UNIT 
  --PREDICTION_HORIZON_FUTURE 
  --PREDICTION_HORIZON_PAST 
  --OBSERVATION_PERIOD 
  --IS_SHUFFLE_TRAIN_SET 
  --EPOCHS_UB 
  --TRAIN_SET_PORTION 
  --VALIDATION_EVERY 
  --IS_TEST_ONLY 
  --TEST_MODEL_PATH 
  --DEVICE 
  --N_GPUs 
  --N_CPUs 
  --DIR_EXPERIMENTS 
  --IS_WANDB 
  --WANDB_SWEEP_METHOD 
  --IS_SANITY_CHECK 
```

This will execute the LOBCAST simulator with seed 42 on FI-2010 dataset, with BINCTABL prediction model, on an observation period of 10 events, 20 epochs, running locally (no WANDB).
```
python -m src.run --SEED 42 --PREDICTION_MODEL BINCTABL --OBSERVATION_PERIOD 10 --EPOCHS_UB 20 --IS_WANDB 0
```

### References
#### LOB-based Deep Learning Models for Stock Price Trend Prediction: A Benchmark Study.

> _The recent advancements in Deep Learning (DL) research have notably influenced the finance sector. We examine the 
> robustness and generalizability of fifteen state-of-the-art DL models focusing on Stock Price Trend Prediction (SPTP) 
> based on Limit Order Book (LOB) data. To carry out this study, we developed LOBCAST, an open-source framework that 
> incorporates data preprocessing, DL model training, evaluation and profit analysis. Our extensive experiments reveal 
> that all models exhibit a significant performance drop when exposed to new data, thereby raising questions about their 
> real-world market applicability. Our work serves as a benchmark, illuminating the potential and the limitations of current 
> approaches and providing insight for innovative solutions._
 
Link: https://arxiv.org/abs/2308.01915 (to appear on 2024 Artificial Intelligence Review journal).