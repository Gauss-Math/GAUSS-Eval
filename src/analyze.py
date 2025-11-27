import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Type
import logging
import sys
import csv
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure project root is in path for imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import abstract base classes (avoid importing data.__init__ which instantiates datasets)
import importlib.util
spec = importlib.util.spec_from_file_location("base", project_root / "data" / "base.py")
base_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base_module)
AbstractResultEntry = base_module.AbstractResultEntry
AbstractResultSet = base_module.AbstractResultSet

# Dynamic dataset discovery and import
def discover_datasets() -> List[str]:
    """Discover all available datasets by scanning the data directory."""
    datasets = []
    data_dir = project_root / "data"
    
    for item in data_dir.iterdir():
        if item.is_dir() and not item.name.startswith('__'):
            # Check if it has a result_entry.py file
            result_entry_file = item / "result_entry.py"
            if result_entry_file.exists():
                datasets.append(item.name)
    
    logger.info(f"Discovered datasets: {datasets}")
    return datasets

def safe_import_dataset_classes(dataset_name: str):
    """Safely import dataset-specific classes without triggering dataset initialization."""
    try:
        spec = importlib.util.spec_from_file_location(
            f"{dataset_name}_result_entry", 
            project_root / "data" / dataset_name / "result_entry.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        entry_class = getattr(module, f"{dataset_name}ResultEntry")
        set_class = getattr(module, f"{dataset_name}ResultSet")
        return entry_class, set_class
    except Exception as e:
        logger.warning(f"Failed to import {dataset_name} classes: {e}")
        return AbstractResultEntry, AbstractResultSet

# Dynamically build dataset class mapping
def build_dataset_classes() -> Dict[str, Tuple[Type[AbstractResultEntry], Type[AbstractResultSet]]]:
    """Build the dataset class mapping dynamically."""
    dataset_classes = {}
    discovered_datasets = discover_datasets()
    
    for dataset_name in discovered_datasets:
        entry_class, set_class = safe_import_dataset_classes(dataset_name)
        dataset_classes[dataset_name] = (entry_class, set_class)
    
    return dataset_classes

# Build the dataset classes mapping dynamically
DATASET_CLASSES = build_dataset_classes()


def get_dataset_classes(dataset_name: str) -> Tuple[Type[AbstractResultEntry], Type[AbstractResultSet]]:
    """Get the appropriate ResultEntry and ResultSet classes for a dataset."""
    # If dataset not found, try to refresh the mapping in case it's a new dataset
    if dataset_name not in DATASET_CLASSES:
        logger.info(f"Dataset '{dataset_name}' not found in cache, refreshing dataset classes...")
        refresh_dataset_classes()
    
    return DATASET_CLASSES.get(dataset_name, (AbstractResultEntry, AbstractResultSet))

def refresh_dataset_classes():
    """Refresh the dataset classes mapping by re-discovering datasets."""
    global DATASET_CLASSES
    DATASET_CLASSES = build_dataset_classes()
    logger.info(f"Refreshed dataset classes. Available datasets: {list(DATASET_CLASSES.keys())}")

def get_available_datasets() -> List[str]:
    """Get a list of all available datasets."""
    return list(DATASET_CLASSES.keys())


class ResultsLoader:
    """Simple loader for evaluation results with support for single directory or batch processing."""
    
    def __init__(self, results_root: Union[str, Path] = "results"):
        """
        Initialize the results loader.
        
        Args:
            results_root: Root directory containing result folders
        """
        self.results_root = Path(results_root)
        if not self.results_root.exists():
            raise FileNotFoundError(f"Results directory not found: {self.results_root}")
        
        logger.info(f"Initialized ResultsLoader with root: {self.results_root}")
    
    def load_single_directory(self, directory: Union[str, Path]) -> AbstractResultSet:
        """
        Load results from a single directory containing JSON files named {id}.json.
        
        Args:
            directory: Path to directory containing result JSON files
            
        Returns:
            Dataset-specific ResultSet object containing all loaded entries
            
        Raises:
            FileNotFoundError: If directory doesn't exist
            ValueError: If no valid JSON files found
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")
        
        logger.info(f"Loading results from single directory: {dir_path}")
        
        # Determine dataset type from directory name
        dataset_name = self._extract_dataset_name(dir_path.name)
        entry_class, result_set_class = get_dataset_classes(dataset_name)
        
        # Find all JSON files that match the pattern {id}.json (numeric IDs)
        json_files = []
        for file_path in dir_path.glob("*.json"):
            if file_path.name != "summary.json":  # Skip summary file
                try:
                    # Check if filename (without extension) is numeric
                    int(file_path.stem)
                    json_files.append(file_path)
                except ValueError:
                    # Skip non-numeric JSON files
                    continue
        
        if not json_files:
            raise ValueError(f"No valid result JSON files found in {dir_path}")
        
        # Sort by numeric ID
        json_files.sort(key=lambda x: int(x.stem))
        
        # Load entries using dataset-specific class
        entries = []
        for file_path in json_files:
            try:
                entry = self._load_single_entry(file_path, entry_class)
                entries.append(entry)
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
                continue
        
        # Load summary if available
        summary = None
        summary_path = dir_path / "summary.json"
        if summary_path.exists():
            try:
                with open(summary_path, 'r', encoding='utf-8') as f:
                    summary = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load summary from {summary_path}: {e}")
        
        # Create dataset-specific result set
        result_set = result_set_class(
            name=dir_path.name,
            directory=dir_path,
            entries=entries,
            summary=summary
        )
        
        logger.info(f"Loaded {len(entries)} entries from {dir_path.name} using {dataset_name} classes")
        return result_set
    
    def load_batch(self, pattern: Optional[str] = None) -> Dict[str, AbstractResultSet]:
        """
        Load results from all subdirectories in the results root.
        
        Args:
            pattern: Optional glob pattern to filter directories (e.g., "USAMO*", "*gpt*")
            
        Returns:
            Dictionary mapping directory names to dataset-specific ResultSet objects
        """
        logger.info(f"Loading batch results from: {self.results_root}")
        if pattern:
            logger.info(f"Using pattern filter: {pattern}")
        
        result_sets = {}
        
        # Find all subdirectories
        directories = []
        if pattern:
            directories = list(self.results_root.glob(pattern))
            # Filter to only directories
            directories = [d for d in directories if d.is_dir()]
        else:
            directories = [d for d in self.results_root.iterdir() if d.is_dir()]
        
        logger.info(f"Found {len(directories)} directories to process")
        
        for dir_path in directories:
            try:
                result_set = self.load_single_directory(dir_path)
                result_sets[dir_path.name] = result_set
            except Exception as e:
                logger.error(f"Failed to load directory {dir_path.name}: {e}")
                continue
        
        logger.info(f"Successfully loaded {len(result_sets)} result sets")
        return result_sets
    
    def _extract_dataset_name(self, directory_name: str) -> str:
        """
        Extract dataset name from directory name.
        
        Handles various naming patterns:
        - DATASET_model_timestamp
        - DATASET_model_timestamp_extra
        - single_example directories: DATASET_model_timestamp_single_example
        """
        parts = directory_name.split('_')
        if len(parts) >= 1:
            potential_dataset = parts[0]
            # Check if this is a known dataset
            if potential_dataset in DATASET_CLASSES:
                return potential_dataset
            
            # If not found, try to match against available datasets (case-insensitive)
            available_datasets = get_available_datasets()
            for dataset in available_datasets:
                if potential_dataset.upper() == dataset.upper():
                    return dataset
        
        # If no match found, log warning and return the first part
        logger.warning(f"Could not identify dataset from directory name '{directory_name}'. Available datasets: {get_available_datasets()}")
        return parts[0] if parts else "UNKNOWN"
    
    def _load_single_entry(self, file_path: Path, entry_class: Type[AbstractResultEntry] = AbstractResultEntry) -> AbstractResultEntry:
        """Load a single result entry from a JSON file using the specified entry class."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Use the dataset-specific class to create the entry
        return entry_class.from_json_data(data, str(file_path))


# Convenience functions for easy usage
def load_results(directory: Union[str, Path]) -> AbstractResultSet:
    """
    Convenience function to load results from a single directory.
    
    Args:
        directory: Path to directory containing result JSON files
        
    Returns:
        Dataset-specific ResultSet object containing all loaded entries
    """
    loader = ResultsLoader()
    return loader.load_single_directory(directory)


def load_all_results(results_root: Union[str, Path] = "results", pattern: Optional[str] = None) -> Dict[str, AbstractResultSet]:
    """
    Convenience function to load all results from the results directory.
    
    Args:
        results_root: Root directory containing result folders
        pattern: Optional glob pattern to filter directories
        
    Returns:
        Dictionary mapping directory names to dataset-specific ResultSet objects
    """
    loader = ResultsLoader(results_root)
    return loader.load_batch(pattern)


def list_available_results(results_root: Union[str, Path] = "results") -> List[str]:
    """
    List all available result directories.
    
    Args:
        results_root: Root directory containing result folders
        
    Returns:
        List of directory names that contain results
    """
    results_path = Path(results_root)
    if not results_path.exists():
        return []
    
    directories = []
    for item in results_path.iterdir():
        if item.is_dir():
            # Check if it contains JSON result files
            json_files = list(item.glob("*.json"))
            if json_files:
                directories.append(item.name)
    
    return sorted(directories)


def get_dataset_info() -> Dict[str, Dict[str, Any]]:
    """
    Get information about all available datasets.
    
    Returns:
        Dictionary with dataset information including available classes
    """
    info = {}
    for dataset_name, (entry_class, set_class) in DATASET_CLASSES.items():
        info[dataset_name] = {
            "entry_class": entry_class.__name__,
            "set_class": set_class.__name__,
            "module_path": f"data/{dataset_name}/result_entry.py"
        }
    
    return info


def validate_dataset_structure(dataset_name: str) -> Dict[str, Any]:
    """
    Validate that a dataset has the required structure and classes.
    
    Args:
        dataset_name: Name of the dataset to validate
        
    Returns:
        Dictionary with validation results
    """
    validation = {
        "dataset_name": dataset_name,
        "valid": False,
        "errors": [],
        "warnings": []
    }
    
    # Check if dataset directory exists
    dataset_dir = project_root / "data" / dataset_name
    if not dataset_dir.exists():
        validation["errors"].append(f"Dataset directory not found: {dataset_dir}")
        return validation
    
    # Check if result_entry.py exists
    result_entry_file = dataset_dir / "result_entry.py"
    if not result_entry_file.exists():
        validation["errors"].append(f"result_entry.py not found in {dataset_dir}")
        return validation
    
    # Try to import classes
    try:
        entry_class, set_class = safe_import_dataset_classes(dataset_name)
        if entry_class == AbstractResultEntry:
            validation["warnings"].append("Using fallback AbstractResultEntry class")
        if set_class == AbstractResultSet:
            validation["warnings"].append("Using fallback AbstractResultSet class")
        
        validation["entry_class"] = entry_class.__name__
        validation["set_class"] = set_class.__name__
        validation["valid"] = True
        
    except Exception as e:
        validation["errors"].append(f"Failed to import classes: {e}")
    
    return validation


def find_csv_files(results_root: Union[str, Path] = "results") -> List[Dict[str, Any]]:
    """
    Find all CSV files in results directories and extract metadata.
    
    Args:
        results_root: Root directory containing result folders
        
    Returns:
        List of dictionaries containing CSV file info and metadata
    """
    results_path = Path(results_root)
    if not results_path.exists():
        logger.error(f"Results directory not found: {results_path}")
        return []
    
    csv_files = []
    
    # Search for CSV files in all subdirectories
    for csv_file in results_path.rglob("*.csv"):
        # Skip if it's in the root directory (like full_analysis.csv, filtered_analysis.csv)
        if csv_file.parent == results_path:
            continue
            
        result_dir = csv_file.parent
        summary_file = result_dir.parent / "summary.json"
        
        csv_info = {
            "csv_path": csv_file,
            "result_dir": result_dir,
            "csv_filename": csv_file.name,
            "model_name": None,
            "dataset_name": None,
            "judge_model_name": "GPT-4",  # default
            "problem_set_name": "USAMO",  # default
        }
        
        # Try to extract metadata from summary.json
        if summary_file.exists():
            try:
                with open(summary_file, 'r', encoding='utf-8') as f:
                    summary = json.load(f)
                
                # Extract model name from summary
                if "model" in summary:
                    model_full = summary["model"]
                    # Extract the last part after the last slash (e.g., "openrouter/qwen/qwen3-235b-a22b-thinking-2507" -> "qwen3-235b-a22b-thinking-2507")
                    csv_info["model_name"] = model_full.split("/")[-1].lower()
                    csv_info["judge_model_name"] = csv_info["model_name"]
                
                # Extract dataset name from summary
                if "dataset" in summary:
                    csv_info["dataset_name"] = summary["dataset"]
                    csv_info["problem_set_name"] = summary["dataset"]
                
                logger.info(f"Found CSV: {csv_file.name} - Model: {csv_info['model_name']}, Dataset: {csv_info['dataset_name']}")
                
            except Exception as e:
                logger.warning(f"Failed to read summary from {summary_file}: {e}")
                # Try to extract from directory name as fallback
                dir_name = result_dir.name
                parts = dir_name.split('_')
                if len(parts) >= 2:
                    csv_info["dataset_name"] = parts[0]
                    csv_info["problem_set_name"] = parts[0]
                    # Try to extract model name from directory name
                    if len(parts) >= 3:
                        csv_info["model_name"] = parts[1].lower()
                        csv_info["judge_model_name"] = parts[1].lower()
        
        csv_files.append(csv_info)
    
    logger.info(f"Found {len(csv_files)} CSV files in results directories")
    return csv_files


def generate_reports_from_csvs(results_root: Union[str, Path] = "results", pattern: Optional[str] = None, save_figs: bool = False) -> None:
    """
    Generate reports for all CSV files found in results directories.
    
    Args:
        results_root: Root directory containing result folders
        pattern: Optional pattern to filter CSV files
        save_figs: Whether to save figures in the reports
    """
    try:
        # Import the report function - try different import paths
        try:
            from report_fn import eval_model_human
        except ImportError:
            # If we're running from src directory, try relative import
            import sys
            from pathlib import Path
            src_dir = Path(__file__).parent
            sys.path.insert(0, str(src_dir))
            from report_fn import eval_model_human
    except ImportError as e:
        logger.error(f"Failed to import eval_model_human from report_fn.py: {e}")
        return
    
    csv_files = find_csv_files(results_root)
    
    if pattern:
        # Filter CSV files based on pattern
        csv_files = [info for info in csv_files if pattern.lower() in str(info["csv_path"]).lower()]
        logger.info(f"Filtered to {len(csv_files)} CSV files matching pattern: {pattern}")
    
    if not csv_files:
        logger.warning("No CSV files found to process")
        return
    
    for csv_info in csv_files:
        try:
            logger.info(f"Processing CSV: {csv_info['csv_path']}")
            
            # Load the CSV file
            df = pd.read_csv(csv_info["csv_path"])
            logger.info(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
            
            # Prepare parameters for the report function
            judge_model_name = csv_info["judge_model_name"] or "GPT-4"
            problem_set_name = csv_info["problem_set_name"] or "USAMO"
            
            # Set output directory to the parent folder of the CSV
            result_dir = csv_info["result_dir"]
            analysis_dir = result_dir
            
            # Create analysis directory if it doesn't exist
            analysis_dir.mkdir(exist_ok=True)
            
            logger.info(f"Generating report with judge_model_name='{judge_model_name}', problem_set_name='{problem_set_name}'")
            logger.info(f"Output directory: {result_dir}")
            logger.info(f"Reports and figures will be saved to: {analysis_dir}")
            
            
            # Generate the report in the analysis directory
            eval_model_human(
                df=df,
                judge_model_name=judge_model_name,
                problem_set_name=problem_set_name,
                save_figs=save_figs,
                output_dir=str(analysis_dir)
            )
            
            logger.info(f"Successfully generated report for {csv_info['csv_filename']}")
            
        except Exception as e:
            logger.error(f"Failed to generate report for {csv_info['csv_path']}: {e}")
            continue
    
    logger.info("Finished processing all CSV files")


if __name__ == "__main__":
    # Example usage with improved functionality
    import sys
    
    # Show available datasets and results
    print("=== Gauss-Judge Results Analyzer ===")
    print(f"Available datasets: {get_available_datasets()}")
    print(f"Available results: {list_available_results()}")
    print()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "--info":
            # Show dataset information
            print("Dataset Information:")
            info = get_dataset_info()
            for dataset_name, dataset_info in info.items():
                print(f"  {dataset_name}:")
                print(f"    Entry Class: {dataset_info['entry_class']}")
                print(f"    Set Class: {dataset_info['set_class']}")
                print(f"    Module Path: {dataset_info['module_path']}")
            
        elif command == "--validate":
            # Validate all datasets
            print("Dataset Validation:")
            for dataset_name in get_available_datasets():
                validation = validate_dataset_structure(dataset_name)
                status = "✓" if validation["valid"] else "✗"
                print(f"  {status} {dataset_name}")
                if validation["errors"]:
                    for error in validation["errors"]:
                        print(f"    Error: {error}")
                if validation["warnings"]:
                    for warning in validation["warnings"]:
                        print(f"    Warning: {warning}")
        
        elif command == "--find-csvs":
            # Find and list all CSV files
            print("Finding CSV files in results directories...")
            csv_files = find_csv_files()
            if csv_files:
                print(f"Found {len(csv_files)} CSV files:")
                for csv_info in csv_files:
                    print(f"  {csv_info['csv_filename']}")
                    print(f"    Path: {csv_info['csv_path']}")
                    print(f"    Model: {csv_info['model_name']}")
                    print(f"    Dataset: {csv_info['dataset_name']}")
                    print()
            else:
                print("No CSV files found.")
        
        elif command == "--generate-reports":
            # Generate reports from all CSV files
            print("Generating reports from CSV files...")
            
            # Parse additional arguments
            pattern = None
            save_figs = False
            
            # Check for additional arguments
            for i in range(2, len(sys.argv)):
                arg = sys.argv[i]
                if arg == "--figure":
                    save_figs = True
                    print("Figures will be saved in reports")
                elif not arg.startswith("--"):
                    # Assume it's a pattern if it doesn't start with --
                    pattern = arg
                    print(f"Using pattern filter: {pattern}")
            
            generate_reports_from_csvs(pattern=pattern, save_figs=save_figs)
        
        elif command.startswith("--"):
            print("Available commands:")
            print("  --info            Show dataset information")
            print("  --validate        Validate dataset structures")
            print("  --find-csvs       Find and list all CSV files in results directories")
            print("  --generate-reports [pattern] [--figure]  Generate reports from CSV files")
            print("                    Optional arguments:")
            print("                      pattern: Filter CSV files by pattern")
            print("                      --figure: Save figures in reports")
            print("  <directory>       Load specific result directory")
            
        else:
            # Load specific directory
            directory = command
            print(f"Loading results from: {directory}")
            try:
                result_set = load_results(directory)
                print(f"Loaded {len(result_set.entries)} entries")
                print(f"Dataset: {result_set.metadata.get('dataset', 'unknown')}")
                print(f"Model: {result_set.metadata.get('model', 'unknown')}")
                analysis = result_set.analyze()
                print(f"Analysis: {analysis}")
            except Exception as e:
                print(f"Error loading results: {e}")
    else:
        # Load all results
        print("Loading all results...")
        all_results = load_all_results()
        print(f"Loaded {len(all_results)} result sets:")
        for name, result_set in all_results.items():
            print(f"  {name}: {len(result_set.entries)} entries, dataset: {result_set.metadata.get('dataset', 'unknown')}")
            analysis = result_set.analyze()
            print(f"  Analysis: {analysis}")