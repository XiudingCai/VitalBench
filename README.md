# MambaTS

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

3. Prepare Data. You can obtain the well pre-processed datasets from public channel like [[Google Drive]](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing) or [[Tsinghua Cloud]](https://cloud.tsinghua.edu.cn/f/2ea5ca3d621e4e5ba36a/), Then place the downloaded data in the folder`./dataset`. 

4. Train and evaluate model. We provide the experiment scripts for MambaTS under the folder `./scripts/`. You can reproduce the experiment results as the following examples:

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

```