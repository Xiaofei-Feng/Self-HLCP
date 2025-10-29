# Self-HLCP

Author implementation of the paper: Self-supervised clustering algorithm based on hierarchy of local density clusters and constraint propagation.

## Requirements

```
python==3.12.2
numpy==2.1.3
networkx==3.3
scikit-learn==1.7.2
scipy==1.14.0
```

## Description

- `main.py`: Main script to run the algorithm
- `src/Self-HLCP.py`: Algorithm implementation
- `src/evaluation.py`: Metrics calculation
- `datasets/`: Directory containing the datasets

## Run

1. Change the dataset in `main.py`;
2. Adjust the parameters in `main.py` if needed;
3. Run the command in terminal: `python main.py`
