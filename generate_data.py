import argparse
import os
import json
import random

import numpy as np
import pandas as pd

from utils.data_utils import check_extension, save_dataset


def generate_maintenance_scheduling_data(dataset_size, num_turbines,
                                          cost_profile_dir, failure_scenarios_file,
                                          production_file, market_price_file,
                                          num_periods=15, maintenance_capacity=2,
                                          num_idle_turbines=5, num_cost_profiles=59,
                                          failure_cost=400.0, preventive_cost_scale=5.0):
    """
    Generates AttenCOpt wind farm O&M scheduling instances following the
    Multi-Step Embedding process (Section IV-A of the paper):
      - Maintenance Cost Embedding x_{i,t} (Equation 12): for each sampled
        preventive maintenance cost profile and scenario-specific failure
        time F_i^s, combines preventive cost + lost production revenue
        before failure, or failure cost + lost production revenue after
        failure, then averages across all failure scenarios of that row.
      - Idle Period Embedding: idle turbines (the maintenance crew depot)
        incur zero maintenance cost and zero location.
      - Crew Schedule Embedding: the cost embedding is repeated
        `maintenance_capacity` (M) times per period to allow scheduling up
        to M maintenance activities per period.
      - Location Dimension Alignment: each turbine's location l_i is
        broadcast to the same (T x M) shape as the cost embedding.

    Args:
        dataset_size: number of instances to generate.
        num_turbines: number of real turbines (I).
        cost_profile_dir: directory containing `replacement_cost_profile<p>.csv`
            preventive maintenance cost profiles (rows are candidate cost
            trajectories, columns are periods).
        failure_scenarios_file: JSON mapping `profile<p>` -> row index ->
            {scenario: failure time F_i^s}.
        production_file: JSON with per-turbine, per-scenario production P_i,t^s.
        market_price_file: JSON with per-scenario market price pi_t^s.
        num_periods: number of maintenance periods (T).
        maintenance_capacity: maintenance operations allowed per period (M).
        num_idle_turbines: number of idle/depot turbines (I').
        num_cost_profiles: number of available cost profile files.
        failure_cost: unexpected failure cost (C^f).
        preventive_cost_scale: scale applied to the sampled preventive
            maintenance cost (C_{i,t}).

    Returns:
        np.ndarray of shape (dataset_size, T * M, I + I', 2), where the last
        dimension holds (maintenance_cost, location) pairs.
    """
    num_nodes = num_turbines + num_idle_turbines

    with open(failure_scenarios_file) as f:
        failure_time_scenarios = json.load(f)
    with open(production_file) as f:
        production = json.load(f)
    with open(market_price_file) as f:
        market_price = json.load(f)

    cost_profile_files = [f'replacement_cost_profile{p}.csv' for p in range(1, num_cost_profiles + 1)]

    dataset = np.zeros((dataset_size, num_nodes, num_periods * maintenance_capacity, 2))

    for instance in range(dataset_size):
        for node in range(num_nodes):
            selected_file = random.choice(cost_profile_files)
            df = pd.read_csv(os.path.join(cost_profile_dir, selected_file), header=None)

            # Sample a row, excluding the last 10 rows (reserved for testing)
            min_rows = max(int(0.2 * (len(df) - 10)), 1)
            max_rows = min(int(0.8 * (len(df) - 10)), len(df) - 10)
            n_rows = random.randint(min_rows, max_rows)
            sampled_rows = df.iloc[:-10].sample(n=n_rows)
            preventive_cost = sampled_rows.values[0, :num_periods]
            sampled_row_index = sampled_rows.index[0] + 1

            profile_number = int(selected_file.replace('replacement_cost_profile', '').replace('.csv', ''))
            scenario_failure_times = failure_time_scenarios[f'profile{profile_number}'][f'{sampled_row_index}']

            # Equation 12: average maintenance cost x_{i,t} across failure scenarios
            scenario_costs = []
            for scenario, failure_time in scenario_failure_times.items():
                price = market_price[0]["CPC"][scenario]
                turbine_production = production[0][f"WT_{node + 1}"][scenario]
                cost_t = []
                for t, preventive_cost_t in enumerate(preventive_cost):
                    if t < failure_time:
                        cost_t.append(preventive_cost_scale * preventive_cost_t + price[t] * turbine_production[t])
                    else:
                        lost_revenue = sum(price[l] * turbine_production[l] for l in range(failure_time, t + 1))
                        cost_t.append(failure_cost + lost_revenue)
                scenario_costs.append(cost_t)

            maintenance_cost = np.mean(scenario_costs, axis=0)

            # Crew Schedule Embedding: repeat cost M times per period
            maintenance_cost_embedding = np.repeat(maintenance_cost, maintenance_capacity)

            turbine_location = random.randint(1, 3) if node < num_turbines else 0
            for t in range(num_periods * maintenance_capacity):
                dataset[instance, node, t, 0] = maintenance_cost_embedding[t]
                dataset[instance, node, t, 1] = turbine_location

    # Idle Period Embedding: idle turbines incur zero cost and are located at the depot
    for instance in range(dataset_size):
        for idle_turbine in range(num_turbines, num_nodes):
            dataset[instance, idle_turbine, :, :] = 0

    dataset = np.transpose(dataset, (0, 2, 1, 3))

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate AttenCOpt wind farm O&M scheduling instances")
    parser.add_argument("--filename", help="Filename of the dataset to create (ignores data_dir/name)")
    parser.add_argument("--data_dir", default='data', help="Create dataset in data_dir/maintenance_scheduling (default 'data')")
    parser.add_argument("--name", type=str, required=True, help="Name to identify dataset")
    parser.add_argument("--dataset_size", type=int, default=100, help="Number of instances to generate")
    parser.add_argument("--num_turbines", type=int, default=50, help="Number of turbines (I)")
    parser.add_argument("--num_idle_turbines", type=int, default=10, help="Number of idle/depot turbines (I')")
    parser.add_argument("--num_periods", type=int, default=15, help="Number of maintenance periods (T)")
    parser.add_argument("--maintenance_capacity", type=int, default=2, help="Maintenance operations per period (M)")
    parser.add_argument("--num_cost_profiles", type=int, default=59, help="Number of preventive cost profile files")
    parser.add_argument("--cost_profile_dir", type=str, required=True,
                        help="Directory containing replacement_cost_profile<p>.csv preventive maintenance cost profiles")
    parser.add_argument("--failure_scenarios_file", type=str, required=True,
                        help="JSON file mapping profile/row to per-scenario failure times F_i^s")
    parser.add_argument("--production_file", type=str, required=True,
                        help="JSON file with per-turbine, per-scenario production P_i,t^s")
    parser.add_argument("--market_price_file", type=str, required=True,
                        help="JSON file with per-scenario market price pi_t^s")
    parser.add_argument("--failure_cost", type=float, default=400.0, help="Unexpected failure cost (C^f)")
    parser.add_argument("--preventive_cost_scale", type=float, default=5.0,
                        help="Scale applied to the sampled preventive maintenance cost (C_i,t)")
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument('--seed', type=int, default=54321, help="Random seed")

    opts = parser.parse_args()

    datadir = os.path.join(opts.data_dir, 'maintenance_scheduling')
    os.makedirs(datadir, exist_ok=True)

    if opts.filename is None:
        filename = os.path.join(datadir, "maintenance_scheduling{}_{}_seed{}.pkl".format(
            opts.num_turbines, opts.name, opts.seed))
    else:
        filename = check_extension(opts.filename)

    assert opts.f or not os.path.isfile(check_extension(filename)), \
        "File already exists! Try running with -f option to overwrite."

    np.random.seed(opts.seed)
    random.seed(opts.seed)

    dataset = generate_maintenance_scheduling_data(
        opts.dataset_size,
        opts.num_turbines,
        opts.cost_profile_dir,
        opts.failure_scenarios_file,
        opts.production_file,
        opts.market_price_file,
        num_periods=opts.num_periods,
        maintenance_capacity=opts.maintenance_capacity,
        num_idle_turbines=opts.num_idle_turbines,
        num_cost_profiles=opts.num_cost_profiles,
        failure_cost=opts.failure_cost,
        preventive_cost_scale=opts.preventive_cost_scale,
    )

    print(dataset[0])

    save_dataset(dataset, filename)

