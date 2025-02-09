#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import torch
from pathlib import Path

from sal.utils.llm import LLM 
from sal.config import Config
from sal.models.reward_models import load_prm
from sal.utils.data import get_dataset, save_dataset
from sal.utils.parser import H4ArgumentParser
from sal.utils.score import score
from sal.utils.profiler import ComputeProfiler, profile_dataset_generation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def main():
    parser = H4ArgumentParser(Config)
    config = parser.parse()
    print(config)
    
    # Initialize profiler with hardcoded path
    profiler = ComputeProfiler(
        model_name=config.model_path.split('/')[-1],
        method=config.approach,
        output_dir="/n/netscratch/pehlevan_lab/Lab/mkulkarni/compute_profiles"
    )

    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs")

    hybrid = config.hybrid if hasattr(config, 'hybrid') else False
    print(config.model_path, hybrid)
    
    llm = LLM(
        model=config.model_path,
        seed=config.seed,
        trust_remote_code=True,
        device_map="auto",
        hybrid=hybrid
    )
    print("Model Loaded on device", llm)
    
    prm = load_prm(config)
    print("PRM Loaded")

    dataset = get_dataset(config)
    print("Dataset Loaded")
    
    # Run generation with profiling
    dataset = profile_dataset_generation(dataset, config, llm, prm, profiler)
    print("Dataset Generation Completed")
    
    # Score and save results
    dataset = score(dataset, config)
    save_dataset(dataset, config)
    
    # Save profiling statistics
    profiler.save_stats()
    
    logger.info("Done 🔥!")

if __name__ == "__main__":
    main()
