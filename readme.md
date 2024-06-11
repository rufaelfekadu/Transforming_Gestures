# Transforming Gestures

This repository contains the code for transforming gestures. Follow the instructions below to set up and run the code.

## Prerequisites

- Python 3.10 or higher
- pip package manager

## Setup

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/transforming-gestures.git
    ```

2. Create and activate a Python virtual environment:

    ```bash
    python3 -m venv env
    source env/bin/activate
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Data Preparation

1. Ensure the dataset folder is organized in the following structure:

    ```bash
    ├── dataset
    │   ├── 003
    │   │   ├── S1
    │   │   │   ├── P1
    │   │   │   │   ├── fpe_pos1_SSS_S1_rep0_BT_full.csv
    │   │   │   │   ├── fpe_pos1_SSS_S1_rep0_BT_full.csv
    │   │   │   │   └── log.json
    │   │   │   ├── P2
    │   │   │   │   ├── fpe_pos2_SSS_S1_rep0_BT_full.csv
    │   │   │   │   ├── fpe_pos2_SSS_S1_rep0_BT_full.csv
    │   │   │   │   └── log.json
    │   │   │   ├── ...
    │   │   ├── └── P4
    │   ├── ...
    │   └── 005
    ```

2. Run the following command to prepare the data:

    ```bash
    python prepare_data.py --DATA.PATH path/to/dataset
    ```

    This script will preprocess the data and prepare it for training. An .npz file will be created as a result of this process.

### Model Training

1. Run the following command to train the model:

    ```bash
    python train.py --VIS.SAVE_TEST_SET
    ```

    This script will train the model using the preprocessed data and save the test set for visualization.

## Visualization

### Unity Setup

1. Follow the instructions in the [unity_hand/readme.md](unity_hand/readme.md) file to set up the Unity project. Then, run the following command:

    ```bash
    python visualize.py --SOLVER.PRETRAINED_PATH /path/to/pretrained/directory
    ```

    This will send RPC calls to the Unity project running in the background using the test set saved during training.

## License

This project is licensed under the [MIT License](LICENSE).
