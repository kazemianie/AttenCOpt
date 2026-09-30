from torch.utils.data import Dataset
import torch
import os
import pickle
from problems.scheduling.state_scheduling import StateScheduling
from utils.beam_search import beam_search
import random
import numpy as np
import json

class SchedulingProblem(object):

    NAME = 'maintenance_scheduling'

    @staticmethod
    def get_costs(dataset, pi):
        # Check that tours are valid, i.e. contain 0 to n -1
        assert (
            torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
            pi.data.sort(1)[0]
        ).all(), "Invalid tour"

        # Gather dataset in order of tour
        if len(dataset.size()) == 4:
            d = dataset.gather(2, pi.unsqueeze(-1).unsqueeze(1).expand_as(dataset)).diagonal(dim1=1, dim2=2)
            d = d.permute(0,2,1)
            #dataset = dataset[:,0,:,:]
            #d1 = dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset))
        else:
            d = dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset))
        
        # Length is distance (L2-norm of difference) from each next location from its prev and of last from first
        return (d[:, :, 0].sum(dim=1) + (d[:, 1:, 1] != d[:, :-1, 1]).to(torch.float32).sum(dim=1) * 400) , None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return SchedulingDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateScheduling.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096, dynamic=False):

        assert model is not None, "Provide model"

        if dynamic:
            def propose_expansions(beam, fixed):
                return model.propose_expansions(
                    beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
                )

            return beam_search(dynamic, SchedulingProblem, input, model, beam_size, propose_expansions)
        else:

            state = SchedulingProblem.make_state(
                input, visited_dtype=torch.int64 if compress_mask else torch.uint8
            )
            fixed = model.precompute_fixed(input)
            def propose_expansions(beam):
                return model.propose_expansions(
                    beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
                )

            return beam_search(dynamic, state, beam_size, propose_expansions)

class SchedulingDataset(torch.utils.data.Dataset):
    def __init__(self, filename=None, size=30, num_samples=100, offset=0, distribution=None, is_dynamic=False):
        super(SchedulingDataset, self).__init__()


        
        self.size = size
        self.num_turbines = 25
        self.num_periods = 15
        self.maintenance_capacity = 2
        self.profile_dir = os.environ.get('ATTENCOPT_PROFILE_DIR', "/scratch/imanka/GT-RLnew/dataset/data")
        self.num_profiles = int(os.environ.get('ATTENCOPT_NUM_PROFILES', 59))

        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'
            with open(filename, 'rb') as fa:
                data = pickle.load(fa)
                self.data = [torch.FloatTensor(row) for row in (data[offset:offset+num_samples])]
        else:
            if is_dynamic:
                self.data = [self.get_dynamic_data(size, 0.1) for i in range(num_samples)]
            else:
                self.data = [torch.FloatTensor(size, 2).uniform_(0, 1) for i in range(num_samples)]
        #self.dataframes = None
        #self.production = None
        #self.failure_time_scenarios = None
        #self.price = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def as_tensor(self):
        return torch.stack(self.data, dim=0)

    def get_dynamic_data(self, size, strength=0.01):
        turbine_embeddings = []
        for turbine in range(self.size):
            if turbine >= self.num_turbines:
                turbine_data = torch.zeros(self.num_periods * self.maintenance_capacity, 2)
            else:
                # Select a random cost profile for the turbine
                random_profile = torch.randint(1, self.num_profiles + 1, (1,)).item()
                
                # Construct the file name
                file_name = f'Node{turbine+1}_Profile{random_profile}.json'
                file_path = os.path.join(self.profile_dir, file_name)
                
                # Load the JSON data from the file
                with open(file_path) as f:
                    data = json.load(f)
                
                # Extract all rows from the data
                rows = list(data.values())
                
                # Select a random row from the data
                random_row = random.choice(rows)
                
                # Extract the first `num_periods` elements
                maintenance_cost = random_row[:self.num_periods]
                
                # Crew Schedule Embedding: repeat cost M times per period
                maintenance_cost_embedding = np.repeat(maintenance_cost, self.maintenance_capacity)
                
                # Assign values to the turbine_data
                turbine_data = torch.zeros(self.num_periods * self.maintenance_capacity, 2)
                turbine_data[:, 0] = torch.tensor(maintenance_cost_embedding, dtype=torch.float32)
                
                # Set turbine location
                turbine_location = torch.randint(1, 4, (1,)).item()
                turbine_data[:, 1] = turbine_location
            
            turbine_embeddings.append(turbine_data)
        
        final_data = torch.stack(turbine_embeddings, dim=0)
        final_data = final_data.permute(1, 0, 2)
        return final_data

class MaintenanceScheduling(SchedulingProblem):
    @staticmethod
    def make_dataset(*args, **kwargs):
        kwargs['is_dynamic'] = True
        return SchedulingDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateScheduling.initialize(*args, **kwargs)
