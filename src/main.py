import sys
import os
from pathlib import Path

# Ensure project root is in path for imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from data import get_dataset_fn
from tqdm import tqdm
from litellm import completion, acompletion
import argparse
import json
import time
import logging
from multiprocessing import Pool, current_process
from functools import partial
import glob
import random
from typing import Dict, Any
import asyncio

def parse_sample_indices(indices_string):
    """
    Parse sample indices string into a list of integers.
    Supports comma-separated values and ranges (e.g., '0,1,2,5-10,15').
    
    Args:
        indices_string (str): String representation of indices
        
    Returns:
        list: List of integer indices, sorted and deduplicated
    """
    if not indices_string or not indices_string.strip():
        return []
    
    indices = []
    parts = indices_string.split(',')
    
    for part in parts:
        part = part.strip()
        if '-' in part:
            # Handle range like "5-10"
            try:
                start, end = part.split('-', 1)
                start, end = int(start.strip()), int(end.strip())
                if start <= end:
                    indices.extend(range(start, end + 1))
            except ValueError:
                print(f"Warning: Invalid range format '{part}', skipping")
        else:
            # Handle single number
            try:
                indices.append(int(part))
            except ValueError:
                print(f"Warning: Invalid number '{part}', skipping")
    
    # Remove duplicates and sort
    return sorted(list(set(indices)))


def setup_logging(save_dir=None, process_id=None):
    """Configure elegant logging with console and optional file output."""
    logger_name = f"{__name__}.{process_id}" if process_id else __name__
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # Remove any existing handlers
    
    # Console handler with clean formatting
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s │ %(levelname)-8s │ %(processName)-12s │ %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler if save directory is provided
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        log_filename = f'run_process_{process_id}.log' if process_id else 'run.log'
        file_handler = logging.FileHandler(
            os.path.join(save_dir, log_filename),
            mode='a',  # Changed to append mode for resume
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s │ %(levelname)-8s │ %(processName)-12s │ %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="USAMO2025")
    parser.add_argument("--model", type=str, default="openrouter/openai/gpt-4o-mini")
    parser.add_argument("--global_config", type=str, default="global_config.json",
                        help="Path to global configuration file")
    parser.add_argument("--sample_indices", type=str, default=None,
                        help="Comma-separated list of sample indices to process (e.g., '0,1,2,5-10')")
    parser.add_argument("--resume_from", type=str, default=None, 
                        help="Path to previous run directory to resume from")
    parser.add_argument("--async", action="store_true", 
                        help="Use async mode for inference instead of multiprocessing")
    return parser.parse_args()

def query_model(model, prompt, sampling_params):
    response = completion(
        model=model,
        messages=[
            {"role": "system", "content": prompt['system_prompt']},
            {"role": "user", "content": prompt['user_prompt']}
        ],
        **sampling_params
    )
    return response

async def query_model_async(model, prompt, sampling_params):
    response = await acompletion(
        model=model,
        messages=[
            {"role": "system", "content": prompt['system_prompt']},
            {"role": "user", "content": prompt['user_prompt']}
        ],
        **sampling_params
    )
    return response

def get_completed_ids(save_dir):
    """Get set of IDs that have already been processed."""
    completed_ids = set()
    json_files = glob.glob(os.path.join(save_dir, "*.json"))
    
    for json_file in json_files:
        # Skip summary.json
        if os.path.basename(json_file) == 'summary.json':
            continue
        if os.path.basename(json_file) == 'merged_samples.json':
            continue
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                # Extract ID from the data or filename
                if 'id' in data:
                    completed_ids.add(data['id'])
                else:
                    # Extract from filename (without .json extension)
                    file_id = os.path.splitext(os.path.basename(json_file))[0]
                    completed_ids.add(file_id)
        except Exception as e:
            logging.warning(f"Could not read {json_file}: {e}")
    
    return completed_ids

def load_resume_config(resume_dir):
    """Load configuration from previous run if available."""
    summary_path = os.path.join(resume_dir, 'summary.json')
    
    if os.path.exists(summary_path):
        try:
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            return summary
        except Exception as e:
            logging.warning(f"Could not load summary from {summary_path}: {e}")
    
    return None

def run_one_example_worker(args_tuple):
    """Worker function that processes a single example. Takes a tuple of arguments."""
    data_item, model, dataset_name, save_dir, sampling_params = args_tuple
    
    # Get process name for logging
    process_name = current_process().name
    
    # Setup logging for this worker
    logger = setup_logging(save_dir, process_name)
    
    try:
        logger.info(f"Processing example: {data_item['id']}")
        
        # Get dataset instance (each worker needs its own)
        dataset = get_dataset_fn[dataset_name]
        
        prompt = dataset.parse_prompt(data_item)
        response = query_model(model, prompt, sampling_params)
        item_to_save = dataset.parse_response(response, data_item)
        
        output_path = os.path.join(save_dir, f"{data_item['id']}.json")
        with open(output_path, 'w') as f:
            json.dump(item_to_save, f, indent=2)
        
        logger.debug(f"Saved result to: {output_path}")
        return {'success': True, 'id': data_item['id'], 'result': item_to_save}
        
    except Exception as e:
        logger.error(f"Failed to process example {data_item['id']}: {e}", exc_info=True)
        return {'success': False, 'id': data_item['id'], 'error': str(e)}

async def run_one_example_async(semaphore, data_item, model, dataset_name, save_dir, sampling_params, logger):
    """Async worker function that processes a single example with concurrency control."""
    async with semaphore:
        try:
            logger.info(f"Processing example: {data_item['id']}")
            
            # Get dataset instance
            dataset = get_dataset_fn[dataset_name]
            
            prompt = dataset.parse_prompt(data_item)
            response = await query_model_async(model, prompt, sampling_params)
            item_to_save = dataset.parse_response(response, data_item)
            
            output_path = os.path.join(save_dir, f"{data_item['id']}.json")
            with open(output_path, 'w') as f:
                json.dump(item_to_save, f, indent=2)
            
            logger.debug(f"Saved result to: {output_path}")
            return {'success': True, 'id': data_item['id'], 'result': item_to_save}
            
        except Exception as e:
            logger.error(f"Failed to process example {data_item['id']}: {e}", exc_info=True)
            return {'success': False, 'id': data_item['id'], 'error': str(e)}

async def process_examples_async(remaining_data, args, save_dir, sampling_params, logger, max_concurrent=10):
    """Process examples asynchronously with concurrency control."""
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Create tasks for all examples
    tasks = [
        run_one_example_async(
            semaphore, data_item, args.model, args.dataset, 
            save_dir, sampling_params, logger
        )
        for data_item in remaining_data
    ]
    
    logger.info(f"Starting async processing with max {max_concurrent} concurrent requests")
    
    # Process with tqdm progress bar
    results = []
    with tqdm(total=len(tasks), desc="Processing examples") as pbar:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            pbar.update(1)
    
    return results

def main():
    args = parse_arguments()
    
    # Load configuration
    try:
        with open(args.global_config, 'r') as f:
            global_config = json.load(f)
        print(f"Loaded global config from: {args.global_config}")
    except FileNotFoundError:
        print(f"Error: Global config file not found: {args.global_config}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in global config file {args.global_config}: {e}")
        sys.exit(1)   

    # Setup run directory
    if args.resume_from:
        save_dir = args.resume_from
        run_name = os.path.basename(save_dir)
        resuming = True
        
        # Verify the directory exists
        if not os.path.exists(save_dir):
            raise ValueError(f"Resume directory does not exist: {save_dir}")
        
        # Load previous configuration if available
        previous_summary = load_resume_config(save_dir)
        if previous_summary:
            # Optionally override args with previous run's settings
            if 'dataset' in previous_summary:
                args.dataset = previous_summary['dataset']
            if 'model' in previous_summary:
                args.model = previous_summary['model']
    else:
        run_name = f"{args.dataset}_{args.model.split('/')[-1]}_{int(time.time())}"
        save_dir = os.path.join(global_config['run_config']['save_dir'], run_name)
        resuming = False
    
    # Initialize main process logging
    logger = setup_logging(save_dir)
    
    # Get number of processes
    num_processes = global_config['run_config'].get('num_processes', 1)
    
    # Log run configuration
    logger.info("=" * 60)
    if resuming:
        logger.info(f"Resuming run: {run_name}")
    else:
        logger.info(f"Starting run: {run_name}")
    logger.info("=" * 60)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Model: {args.model}")
    if getattr(args, 'async', False):
        logger.info(f"Mode: Async (max concurrent: {num_processes})")
    else:
        logger.info(f"Mode: Multiprocessing (processes: {num_processes})")
    logger.info(f"Save directory: {save_dir}")
    sampling_params = global_config['sampling_params']
    logger.info(f"Sampling params: {json.dumps(sampling_params, indent=2)}")
    logger.info("=" * 60)

    dataset = get_dataset_fn[args.dataset]
    logger.info(f"Loaded dataset with {len(dataset.data)} examples")

    # Filter by sample indices if specified (from command line or config)
    sample_indices = None
    
    # Check command line argument first
    if args.sample_indices:
        sample_indices = parse_sample_indices(args.sample_indices)
        logger.info(f"Using sample indices from command line: {len(sample_indices)} samples")
    # Check global config for sample indices
    elif 'sample_indices' in global_config.get('run_config', {}):
        config_indices = global_config['run_config']['sample_indices']
        if config_indices:  # Only use if not empty
            sample_indices = config_indices
            logger.info(f"Using sample indices from config: {len(sample_indices)} samples")
    
    # Apply sample indices filtering if specified
    if sample_indices is not None:
        # Filter dataset to only include specified sample indices
        filtered_data = []
        available_ids = {item['id'] for item in dataset.data}
        
        for idx in sample_indices:
            if idx in available_ids:
                # Find the item with this id
                item = next(item for item in dataset.data if item['id'] == idx)
                filtered_data.append(item)
            else:
                logger.warning(f"Sample index {idx} not found in dataset (max index: {max(available_ids) if available_ids else 'N/A'})")
        
        dataset.data = filtered_data
        logger.info(f"Filtered dataset to {len(dataset.data)} examples based on sample indices")
        
        if len(dataset.data) == 0:
            logger.error("No valid samples found after filtering by sample indices")
            return

    # Filter out already completed examples if resuming
    if resuming:
        completed_ids = get_completed_ids(save_dir)
        logger.info(f"Found {len(completed_ids)} already completed examples")
        
        remaining_data = [item for item in dataset.data if item['id'] not in completed_ids]
        logger.info(f"Remaining examples to process: {len(remaining_data)}")
        
        if len(remaining_data) == 0:
            logger.info("All examples already completed. Nothing to do.")
            return
    else:
        remaining_data = dataset.data
        completed_ids = set()

    # Prepare arguments for each worker
    worker_args = [
        (data_item, args.model, args.dataset, save_dir, sampling_params)
        for data_item in remaining_data
    ]

    # Process examples based on mode
    if getattr(args, 'async', False):
        logger.info("Running in async mode")
        logger.info(f"Max concurrent async requests: {num_processes}")
        
        # Run async processing
        results = asyncio.run(process_examples_async(
            remaining_data, args, save_dir, sampling_params, logger, num_processes
        ))
        
        # Summarize results
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful
        logger.info(f"Async processing complete: {successful} successful, {failed} failed")
        
    elif num_processes > 1:
        logger.info(f"Starting parallel processing with {num_processes} workers")
        
        # Use multiprocessing pool
        with Pool(processes=num_processes) as pool:
            # Process with progress bar
            results = list(tqdm(
                pool.imap(run_one_example_worker, worker_args),
                total=len(worker_args),
                desc="Processing examples"
            ))
        
        # Summarize results
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful
        logger.info(f"Processing complete: {successful} successful, {failed} failed")
        
    else:
        logger.info("Running in sequential mode (single process)")
        results = []
        for worker_arg in tqdm(worker_args, desc="Processing examples"):
            result = run_one_example_worker(worker_arg)
            results.append(result)

    # Calculate total statistics (including previously completed)
    total_successful = len(completed_ids) + sum(1 for r in results if r['success'])
    total_failed = len(dataset.data) - total_successful

    # Save summary
    summary = {
        'run_name': run_name,
        'dataset': args.dataset,
        'model': args.model,
        'num_processes': num_processes,
        'total_examples': len(dataset.data),
        'successful': total_successful,
        'failed': total_failed,
        'resumed': resuming,
        'previously_completed': len(completed_ids) if resuming else 0,
        'newly_processed': len(results),
        'sample_indices_used': sample_indices if sample_indices is not None else 'all',
        'sample_indices_source': 'command_line' if args.sample_indices else ('config' if sample_indices is not None else 'all'),
        'results': results
    }
    
    summary_path = os.path.join(save_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Summary saved to: {summary_path}")
    logger.info("=" * 60)
    logger.info("Run completed successfully")
    if resuming:
        logger.info(f"Total completed: {total_successful}/{len(dataset.data)}")
        logger.info(f"Previously completed: {len(completed_ids)}")
        logger.info(f"Newly processed: {len(results)}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()