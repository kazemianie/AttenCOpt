# Generating AttenCOpt Data

`generate_data.py` builds wind farm O&M scheduling instances used to train/test
AttenCOpt (see paper Section IV-A, Multi-Step Embeddings). Each instance
encodes, per turbine and period, a **maintenance cost embedding** `x_{i,t}`
(Equation 12) and a **location**, aligned across `T x M` (periods x
maintenance capacity), with idle/depot turbines zeroed out.

`x_{i,t}` is computed by averaging, over all failure scenarios of a sampled
preventive-cost profile row, the cost of maintaining before failure
(preventive cost + lost production revenue) or after failure (failure cost
`C^f` + lost production revenue), using the scenario's market price `pi_t^s`
and production `P_{i,t}^s`.

## Prerequisites

- A folder of preventive maintenance cost profile CSVs named
  `replacement_cost_profile<p>.csv` for `p = 1..num_cost_profiles`, each row
  being a candidate cost trajectory over periods.
- `failure_scenarios_file` (JSON): `profile<p> -> row_index -> {scenario: failure_time}`.
- `production_file` (JSON): `[{"WT_<i>": {scenario: [production_per_period, ...]}}]`.
- `market_price_file` (JSON): `[{"CPC": {scenario: [price_per_period, ...]}}]`.

## Usage

```bash
python generate_data.py --name train \
    --cost_profile_dir /path/to/cost_profiles \
    --failure_scenarios_file /path/to/Failure_time_Scenarios.json \
    --production_file /path/to/WTs_Production_Data.json \
    --market_price_file /path/to/Market_price_Data.json \
    --dataset_size 1280 --num_turbines 50 --num_idle_turbines 10 \
    --num_periods 15 --maintenance_capacity 2 --num_cost_profiles 59 --seed 1234
```

## Key arguments

| Argument | Meaning (paper notation) |
|---|---|
| `--num_turbines` | Number of real turbines (`I`) |
| `--num_idle_turbines` | Idle/depot turbines (`I'`) |
| `--num_periods` | Planning horizon (`T`) |
| `--maintenance_capacity` | Max maintenance ops per period (`M`) |
| `--cost_profile_dir` | Folder with `replacement_cost_profile<p>.csv` preventive cost profiles (`C_{i,t}`) |
| `--failure_scenarios_file` | Per-scenario failure times (`F_i^s`) |
| `--production_file` | Per-turbine, per-scenario production (`P_{i,t}^s`) |
| `--market_price_file` | Per-scenario market price (`pi_t^s`) |
| `--failure_cost` | Unexpected failure cost (`C^f`) |
| `--preventive_cost_scale` | Scale applied to the sampled preventive cost |
| `--dataset_size` | Number of instances to generate |
| `--seed` | Random seed |

Output is saved as a pickle file under `data/maintenance_scheduling/` (or
`--filename` if given), shaped `(dataset_size, T*M, I+I', 2)` with the last
dimension holding `(maintenance_cost, location)`.
