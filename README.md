# VitalBench

The repo is the official implementation for the paper: VitalBench: A Rigorous Multi-Center Benchmark for Long-Term Vital Sign Prediction in Intraoperative Care (arxiv version will be available soon).

## Usage

1. Install Python 3.11. For convenience, execute the following command.

    ```
    pip install -r requirements.txt
    ```

2. For setting up the Mamba environment, please refer to https://github.com/state-spaces/mamba. Here is a simple instruction on Linux system,

    ```
    pip install causal-conv1d>=1.2.0
    pip install mamba-ssm
    ```

3. Prepare Data. 

For this benchmark, we use two publicly available datasets:

- VitalDB: A multi-center, real-time monitoring dataset of intraoperative vital signs. It consists of data from over 4,000 surgeries across multiple hospitals, providing time-series data on various physiological variables such as heart rate, blood pressure, and oxygen saturation. You can access the dataset at VitalDB.

- MOVER: A dataset from the MOVER-SIS project, containing vital sign time series and corresponding clinical labels, designed for anesthesia monitoring tasks. The dataset includes data from multiple hospitals and is freely accessible at MOVER.

The processed dataset, including the necessary preprocessing steps, will be made publicly available in the future to facilitate reproducibility and extend the impact of our work.

4. Train and evaluate model. We provide the experiment scripts for all baselines under the folder `./scripts/`. You can reproduce the experiment results as the following examples:

    ```
    # VitalDB
    bash ./scripts/vital_benchmark/VitalDB@SplitByTime_script/Track1.sh
    bash ./scripts/vital_benchmark/VitalDB@SplitByTime_script/Track2.sh
    bash ./scripts/vital_benchmark/VitalDB@SplitByTime_script/Track3.sh
   
    # MOVER-SIS
    bash ./scripts/vital_benchmark/MOVER@SplitByTime_script/Track1.sh
    bash ./scripts/vital_benchmark/MOVER@SplitByTime_script/Track2.sh
    bash ./scripts/vital_benchmark/MOVER@SplitByTime_script/Track3.sh
    ```

## Acknowledgement

This library is constructed based on the following repos:

- Time-Series-Library: https://github.com/thuml/Time-Series-Library
- MambaTS: https://github.com/XiudingCai/MambaTS-pytorch

We extend our sincere thanks for their excellent work and repositories!

