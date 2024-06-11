# Transforming Gestures

This repository contains code for transforming gestures. Follow the steps below to run the code:

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

1. Prepare the data:
    make sure the dataset folder is organised in the following manner

    ```bash
    .
    ├── dataset
    │   ├── 003
    │   │   ├── S1
    |   |   |   ├── P1
    |   |   |   |   ├── fpe_pos1_SSS_S1_rep0_BT_full.csv
    |   |   |   |   ├── fpe_pos1_SSS_S1_rep0_BT_full.csv
    |   |   |   |   └── log.json
    |   |   |   ├── P2
    |   |   |   |   ├── fpe_pos2_SSS_S1_rep0_BT_full.csv
    |   |   |   |   ├── fpe_pos2_SSS_S1_rep0_BT_full.csv
    |   |   |   |   └── log.json
    |   |   |   ├── ...
    |   |   └── └── P4
    │   ├── ...
    |   └── 005
    |
    ```

    Then run the following command to prepare the data:

    ```bash
    python prepare_data.py --DATA.PATH path/to/dataset
    ```

    This script will preprocess the data and prepare it for training. An .npz files will be created as a reuslt of this processes.

2. Train the model:

    ```bash
    python train.py
    ```

This script will train the model using the preprocessed data.

 ## Visualization:

1. Setup unity:
    
Follow the instruction in [here](hand project/readme.md)

    ```bash
    python evaluate.py
    ```

    This script will evaluate the trained model on a test dataset.

## License

This project is licensed under the [MIT License](LICENSE).