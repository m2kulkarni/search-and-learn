#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.

import logging
import torch
import time
import json
import os
import psutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

class ComputeProfiler:
    def __init__(self, model_name: str, method: str, output_dir: str):
        # Get detailed SLURM array job information
        self.job_id = os.environ.get('SLURM_ARRAY_JOB_ID', 'local')
        self.array_task_id = os.environ.get('SLURM_ARRAY_TASK_ID', '0')
        self.array_task_count = int(os.environ.get('SLURM_ARRAY_TASK_COUNT', '1'))
        self.node_name = os.environ.get('SLURMD_NODENAME', 'local')
        
        # Create job-specific output directory
        self.output_dir = Path(output_dir) / f"job_{self.job_id}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset processing range
        self.dataset_start = None
        self.dataset_end = None
        
        self.stats = {
            # Job info
            "job_info": {
                "job_id": self.job_id,
                "array_job_id": os.environ.get('SLURM_ARRAY_JOB_ID', 'local'),
                "array_task_id": self.array_task_id,
                "array_task_count": self.array_task_count,
                "node_name": self.node_name,
                "gpu_id": os.environ.get('CUDA_VISIBLE_DEVICES', '0'),
                "dataset_partition": {
                    "start_idx": None,
                    "end_idx": None,
                    "total_samples": None,
                },
                "job_start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            
            # Model info
            "model_name": model_name,
            "method": method,
            "num_gpus": torch.cuda.device_count(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
            
            # Time metrics
            "total_cuda_time_ns": 0,
            "total_cpu_time_ns": 0,
            "avg_generation_time_ns": 0,
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            
            # Memory metrics
            "peak_memory_mb": 0,
            "peak_memory_percent": 0,
            "avg_memory_mb": 0,
            "memory_samples": [],
            
            # GPU metrics
            "gpu_utilization_samples": [],
            "avg_gpu_utilization": 0,
            "peak_gpu_utilization": 0,
            
            # Generation metrics
            "generations": 0,
            "tokens_generated": 0,
            "tokens_per_second": 0,
            "generations_per_second": 0,
            
            # Batch metrics
            "num_batches": 0,
            "avg_batch_size": 0,
            "total_batch_time_ns": 0,
            
            # Error tracking
            "num_failed_generations": 0,
            "error_messages": [],
            
            # Per-partition metrics
            "partition_metrics": {
                "samples_processed": 0,
                "partition_progress": 0.0,
                "time_per_sample": [],
                "memory_per_sample": [],
                "samples_per_second": 0,
            },
            
            # Resource utilization
            "resource_usage": {
                "cpu_percent": [],
                "ram_usage": [],
                "gpu_memory_usage": [],
                "gpu_utilization": [],
                "process_memory": None,
            },
        }
        
        # Initialize CUDA events for timing
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.batch_start_time = None
    
    def set_dataset_range(self, start: int, end: int, total: int):
        """Set the dataset range this job is processing"""
        self.dataset_start = start
        self.dataset_end = end
        self.stats["job_info"]["dataset_partition"].update({
            "start_idx": start,
            "end_idx": end,
            "total_samples": total,
        })
    
    def update_partition_progress(self, samples_processed: int):
        """Update progress for this partition"""
        self.stats["partition_metrics"]["samples_processed"] = samples_processed
        if self.dataset_end and self.dataset_start:
            total_partition = self.dataset_end - self.dataset_start
            self.stats["partition_metrics"]["partition_progress"] = \
                (samples_processed / total_partition) * 100 if total_partition > 0 else 0
    
    def sample_resource_usage(self):
        """Sample current resource usage"""
        try:
            process = psutil.Process()
            
            # CPU and RAM usage
            self.stats["resource_usage"]["cpu_percent"].append(process.cpu_percent())
            self.stats["resource_usage"]["ram_usage"].append(process.memory_info().rss / 1024 / 1024)  # MB
            
            # GPU stats
            if torch.cuda.is_available():
                self.stats["resource_usage"]["gpu_memory_usage"].append(
                    torch.cuda.memory_allocated() / 1024 / 1024  # MB
                )
                try:
                    self.stats["resource_usage"]["gpu_utilization"].append(
                        torch.cuda.utilization()
                    )
                except:
                    pass
        except Exception as e:
            logger.warning(f"Failed to sample resource usage: {e}")
    
    def start(self):
        """Start profiling a batch"""
        torch.cuda.reset_peak_memory_stats()
        self.start_event.record()
        self.batch_start_time = time.time()
        
        # Sample GPU utilization at start
        if torch.cuda.is_available():
            try:
                gpu_util = torch.cuda.utilization()
                self.stats["gpu_utilization_samples"].append(gpu_util)
            except:
                pass
    
    def end(self, num_generations: int, num_tokens: int, failed_generations: int = 0, error_msg: Optional[str] = None):
        """End profiling a batch"""
        if torch.cuda.is_available():
            self.end_event.record()
            torch.cuda.synchronize()
            cuda_time_ns = self.start_event.elapsed_time(self.end_event) * 1e6  # ms to ns
        else:
            cuda_time_ns = 0
            
        batch_end_time = time.time()
        cpu_time_ns = (batch_end_time - self.batch_start_time) * 1e9  # s to ns
        
        # Memory sampling
        current_memory = torch.cuda.memory_allocated() / 1024 / 1024  # bytes to MB
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
            memory_percent = (current_memory / total_memory) * 100 if total_memory > 0 else 0
        else:
            memory_percent = 0
        
        # Update batch statistics
        self.stats["num_batches"] += 1
        self.stats["total_batch_time_ns"] += cuda_time_ns
        
        # Update generation statistics
        self.stats["generations"] += num_generations
        self.stats["tokens_generated"] += num_tokens
        self.stats["num_failed_generations"] += failed_generations
        if error_msg:
            self.stats["error_messages"].append(error_msg)
        
        # Update memory statistics
        self.stats["memory_samples"].append(current_memory)
        self.stats["peak_memory_mb"] = max(
            self.stats["peak_memory_mb"],
            torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        )
        self.stats["peak_memory_percent"] = max(
            self.stats["peak_memory_percent"],
            memory_percent
        )
        
        # Sample GPU utilization at end
        if torch.cuda.is_available():
            try:
                gpu_util = self.get_gpu_utilization()
                if gpu_util is not None:
                    self.stats["gpu_utilization_samples"].append(gpu_util)
                    self.stats["peak_gpu_utilization"] = max(
                        self.stats["peak_gpu_utilization"],
                        gpu_util
                    )
            except:
                pass
        
        # Update cumulative statistics
        self.stats["total_cuda_time_ns"] += cuda_time_ns
        self.stats["total_cpu_time_ns"] += cpu_time_ns
        
        # Calculate averages
        if self.stats["generations"] > 0:
            self.stats["avg_generation_time_ns"] = self.stats["total_cuda_time_ns"] / self.stats["generations"]
            self.stats["tokens_per_second"] = self.stats["tokens_generated"] / (self.stats["total_cuda_time_ns"] / 1e9)
            self.stats["generations_per_second"] = self.stats["generations"] / (self.stats["total_cuda_time_ns"] / 1e9)
        
        if len(self.stats["memory_samples"]) > 0:
            self.stats["avg_memory_mb"] = sum(self.stats["memory_samples"]) / len(self.stats["memory_samples"])
        
        if len(self.stats["gpu_utilization_samples"]) > 0:
            self.stats["avg_gpu_utilization"] = sum(self.stats["gpu_utilization_samples"]) / len(self.stats["gpu_utilization_samples"])
        
        if self.stats["num_batches"] > 0:
            self.stats["avg_batch_size"] = self.stats["generations"] / self.stats["num_batches"]
            
    def save_stats(self):
        """Save statistics with job ID in filename"""
        self.stats["job_info"]["job_end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Calculate final metrics
        if self.stats["partition_metrics"]["samples_processed"] > 0:
            total_time = self.stats["total_cuda_time_ns"] / 1e9  # Convert to seconds
            self.stats["partition_metrics"]["samples_per_second"] = \
                self.stats["partition_metrics"]["samples_processed"] / total_time
        
        # Create filename with job info
        filename = (f"compute_stats_{self.stats['model_name']}_{self.stats['method']}_"
                   f"job{self.job_id}_array{self.array_task_id}_"
                   f"range{self.dataset_start}-{self.dataset_end}.json")
        output_file = self.output_dir / filename
        
        # Save stats
        with open(output_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        logger.info(f"Saved compute statistics to {output_file}")

def profile_dataset_generation(dataset, config, llm, prm, profiler):
    """Wrapper to profile the dataset generation process"""
    # Set dataset range in profiler
    profiler.set_dataset_range(
        start=config.dataset_start,
        end=config.dataset_end,
        total=len(dataset)
    )
    
    samples_processed = 0
    
    def profiled_approach(x, config, llm, prm):
        nonlocal samples_processed
        from sal.search import beam_search, best_of_n, dvts
        
        APPROACHES = {
            "beam_search": beam_search,
            "dvts": dvts,
            "best_of_n": best_of_n,
        }
        
        approach_fn = APPROACHES[config.approach]
        
        profiler.start()
        profiler.sample_resource_usage()
        
        try:
            result = approach_fn(x, config, llm, prm)
            total_tokens = sum(sum(tokens) for tokens in result["completion_tokens"])
            failed_gens = 0
            error_msg = None
        except Exception as e:
            result = x  # Return original input on failure
            total_tokens = 0
            failed_gens = len(x[config.problem]) * config.n
            error_msg = str(e)
        
        # Update progress
        samples_processed += len(x[config.problem])
        profiler.update_partition_progress(samples_processed)
        
        profiler.end(
            num_generations=len(x[config.problem]) * config.n,
            num_tokens=total_tokens,
            failed_generations=failed_gens,
            error_msg=error_msg
        )
        
        return result
    
    return dataset.map(
        profiled_approach,
        batched=True,
        batch_size=config.search_batch_size,
        fn_kwargs={"config": config, "llm": llm, "prm": prm},
        desc="Running search with profiling",
        load_from_cache_file=True,
    ) 
