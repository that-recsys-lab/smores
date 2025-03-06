# SMORES

This repository contains scripts and data for running SMORES experiments on different datasets. Follow the instructions below to set up your environment and execute experiments.

## Installation

### 1. Create a Virtual Environment
It is recommended to use a virtual environment to manage dependencies. Run the following commands:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 2. Install Dependencies
Once the virtual environment is activated, install the required packages:

```bash
pip install -r requirements.txt
```

## Running an Experiment
To run an experiment, use one of the provided experiment scripts.

### Example: Running an Experiment on the AMBAR Dataset
```bash
python smores-ambar-run.py
```

### Example: Running an Experiment on the ML1M Dataset
```bash
python smores-ml1m-run.py
```

## Project Structure
```
smores/
│── data/                # Raw and processed data files
│   ├── raw/             # Contains original datasets
│── smores-ambar-run.py  # Script for running AMBAR dataset experiment
│── smores-ml1m-run.py   # Script for running ML1M dataset experiment
│── requirements.txt     # Python dependencies
│── LICENSE              # License information
│── README.md            # This file
```

## Additional Notes
- Ensure that the dataset files are properly placed in the `data/raw/` directory before running experiments.
- If additional preprocessing is needed, modify the scripts accordingly.

## License
This project is licensed under the terms specified in the `LICENSE` file.

