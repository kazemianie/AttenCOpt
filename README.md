# AttenCOpt: Attention is All You Need to Optimize Wind Farm Operations and Maintenance

Implementation of AttenCOpt, a multi-head attention (MHA) model that learns to solve the wind farm
operations and maintenance (O&M) scheduling problem, embedding the mixed integer programming (MIP)
objective, constraints, and problem definition directly into the attention architecture. This code
base builds on top of [attention-learn-to-route](https://github.com/wouterkool/attention-learn-to-route).

## Installation

Install following libraries

* Python>=3.8
* NumPy
* SciPy
* [PyTorch](http://pytorch.org/)>=1.7
* tqdm
* [tensorboard_logger](https://github.com/TeamHG-Memex/tensorboard_logger)

Or use [env.yml](env.yml) for the complete list of libraries. You can use Anaconda for the installation.

## Generate Data

Use the following script to generate wind farm O&M scheduling instances (see [DATA_GENERATION.md](DATA_GENERATION.md) for details).

```bash
python generate_data.py
  --name <dataset_name>
  --seed 4321
  --num_turbines 25
  --dataset_size 100
  --cost_profile_dir <folder with replacement_cost_profile<p>.csv files>
  --failure_scenarios_file <Failure_time_Scenarios.json>
  --production_file <WTs_Production_Data.json>
  --market_price_file <Market_price_Data.json>
```

## Train AttenCOpt

Use the following script to start training AttenCOpt for the wind farm O&M scheduling problem with

```bash
python run.py
  --problem maintenance_scheduling
  --model attencopt
  --graph_size 20 
  --baseline rollout 
  --run_name 'maintenance_scheduling' 
  --val_dataset <location of the generated validation dataset>
```

If the --val_dataset is not provided, the validation dataset will be automatically generated.

To enable AttenCOpt in real-time mode use  "--use_single_time"

Refer to [options.py](options.py) for the complete list of parameters

## Pretrained models

Trained AttenCOpt models are saved under the `--output_dir` folder (default [outputs/icde](outputs/icde)),
e.g. `outputs/icde/maintenance_scheduling_20/<run_name>` for the wind farm O&M scheduling problem with 20 turbines.

## Testing

Use the following script to evaluate the trained model.

```bash
python eval.py <location of the generated dataset> 
  --model outputs/icde/maintenance_scheduling_20/<run_name>/
  --decode_strategy <decode strategy> 
  --eval_batch_size 1
```

Possible options for --decode_strategy are "greedy" and "bs" (for beam search). Use --width <int> with "bs" option.

## Visualization
Use the following script to visualize the solution.
  
  
```bash
python eval.py <location of the generated dataset>
  --model outputs/icde/maintenance_scheduling_20/<run_name>/
  --decode_strategy <decode strategy> 
  --eval_batch_size 1
  --plot
  --plot_index <index of the batch to plot>
```
Use --use_gurobi to additionally solve the same instance with the Gurobi MIP benchmark and plot it at the same plot_index.
  
## Acknowledgements
We thank attention learning to route [https://github.com/wouterkool/attention-learn-to-route] for an easily extendable codebase. 

