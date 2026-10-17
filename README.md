# Adaptive cross-catchment learning improves high- and low-flow prediction

This repository provides the implementation of the models used in this study.

The repository contains four model configurations:

- **Global LSTM** — one LSTM model trained across all catchments.
- **Cluster-specific LSTMs** — independent LSTMs trained for predefined catchment clusters.
- **Fixed-routing MoE** — jointly trained experts with fixed cluster-based routing.
- **MoE** — jointly trained experts with catchment-specific Top-2 routing learned from static catchment information.

## Repository structure

```text
.
├── train.py
├── evaluate.py
├── flow_interval_analysis.py
├── data.py
├── training.py
├── metrics.py
├── configs/
│   ├── global_lstm.json
│   ├── cluster_specific_lstms.json
│   ├── fixed_routing_moe.json
│   └── moe.json
├── models/
│   ├── __init__.py
│   ├── lstm.py
│   ├── fixed_routing_moe.py
│   └── moe.py
└── metadata/
    └── basin_cluster_assignments.csv
```

## Environment

```bash
conda env create -f environment.yml -n moe
conda activate moe
```

## Data

The experiments use daily hydrometeorological data from **GRDC-Caravan v0.6** and 96 static catchment attributes, comprising 64 AlphaEarth Foundation embedding features and 32 conventional catchment attributes.

- **GRDC-Caravan v0.6:** https://doi.org/10.5281/zenodo.15349031
- **AlphaEarth Foundations Satellite Embedding V1:** https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL

Set the data paths in the corresponding configuration file:

```json
"data_dir": "PATH/TO/BASIN_TIMESERIES",
"attributes_file": "PATH/TO/STATIC_ATTRIBUTES.xlsx"
```

`metadata/basin_cluster_assignments.csv` contains the 2,293 catchment IDs and the nine cluster assignments used by the cluster-specific and fixed-routing experiments.

## Training

Run from the repository root:

```bash
python train.py --config configs/global_lstm.json
python train.py --config configs/cluster_specific_lstms.json
python train.py --config configs/fixed_routing_moe.json
python train.py --config configs/moe.json
```

Each training run writes its model checkpoint, feature scaler, training history and resolved configuration to the configured output directory.

## Evaluation

Evaluate a trained model by providing its configuration and training output directory:

```bash
python evaluate.py \
    --config configs/moe.json \
    --run-dir PATH/TO/RUN_DIRECTORY
```

By default, evaluation outputs are written to:

```text
PATH/TO/RUN_DIRECTORY/evaluation/
```

The evaluation reports catchment-level NSE, KGE, PBIASHF and PBIASLF. High and low flows correspond to the highest 2% and lowest 30% of observed daily streamflow, respectively.

To save daily predictions for additional flow-interval analysis:

```bash
python evaluate.py \
    --config configs/moe.json \
    --run-dir PATH/TO/RUN_DIRECTORY \
    --save-daily-predictions
```

The six flow intervals analysed for the Global LSTM and MoE can then be evaluated with:

```bash
python flow_interval_analysis.py \
    --global-predictions PATH/TO/GLOBAL_LSTM/evaluation/daily_predictions.csv \
    --moe-predictions PATH/TO/MOE/evaluation/daily_predictions.csv \
    --output-dir PATH/TO/FLOW_INTERVAL_ANALYSIS
```

## Citation

If you use this code, please cite the accompanying paper. Citation information will be updated following publication.

## License

See `LICENSE`.
