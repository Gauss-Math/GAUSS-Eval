from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import importlib.util
import re
import numpy as np
import json
import pandas as pd
from collections import Counter, defaultdict

# Import abstract base classes directly to avoid triggering data.__init__.py
base_path = Path(__file__).parent.parent / "base.py"
spec = importlib.util.spec_from_file_location("base", base_path)
base_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base_module)
AbstractResultEntry = base_module.AbstractResultEntry
AbstractResultSet = base_module.AbstractResultSet


@dataclass
class IMO2025ResultEntry(AbstractResultEntry):
    """IMO2025 dataset result entry."""
    
    # Core data fields
    response: str
    pred_score: Optional[float]
    problem_idx: str
    problem: str
    model_name: str
    model_config: str
    answer: str
    points_judge_1: float
    points_judge_2: float
    max_points_judge_1: float
    max_points_judge_2: float
    grading_details_judge_1: List[Dict[str, Any]]
    grading_details_judge_2: List[Dict[str, Any]]
    error_judge_1: Optional[str]
    error_judge_2: Optional[str]
    input_tokens: int
    output_tokens: int
    cost: float
    source_file: str = ""
    
    @classmethod
    def from_json_data(cls, data: Dict[str, Any], source_file: str = "") -> 'IMO2025ResultEntry':
        """Create an IMO2025ResultEntry from JSON data."""
        return cls(
            response=data.get('response', ''),
            pred_score=None,
            problem_idx=str(data.get('problem_idx', data.get('id', ''))),
            problem=data.get('problem', ''),
            model_name=data.get('model_name', ''),
            model_config=data.get('model_config', ''),
            answer=data.get('answer', ''),
            points_judge_1=float(data.get('points_judge_1') or 0),
            points_judge_2=float(data.get('points_judge_2') or 0),
            max_points_judge_1=float(data.get('max_points_judge_1') or 0),
            max_points_judge_2=float(data.get('max_points_judge_2') or 0),
            grading_details_judge_1=data.get('grading_details_judge_1', []),
            grading_details_judge_2=data.get('grading_details_judge_2', []),
            error_judge_1=data.get('error_judge_1'),
            error_judge_2=data.get('error_judge_2'),
            input_tokens=int(data.get('input_tokens') or 0),
            output_tokens=int(data.get('output_tokens') or 0),
            cost=float(data.get('cost') or 0),
            source_file=source_file
        )


@dataclass
class IMO2025ResultSet(AbstractResultSet):
    """IMO2025 dataset result set."""
    
    name: str
    directory: Path
    entries: List[IMO2025ResultEntry] = field(default_factory=list)
    summary: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    unmatched_indices: List[int] = field(default_factory=list)
    
    def __post_init__(self):
        """Extract metadata from directory name."""
        parts = self.name.split('_')
        if len(parts) >= 2:
            self.metadata['dataset'] = parts[0]
            self.metadata['model'] = '_'.join(parts[1:-1]) if len(parts) > 2 else parts[1]
            if len(parts) > 2:
                self.metadata['timestamp'] = parts[-1]
    
    def _extract_predicted_scores(self) -> None:
        """Extract predicted scores from response texts using \\boxed{} pattern."""
        self.unmatched_indices = []  # Reset unmatched indices list
        
        for i, entry in enumerate(self.entries):
            matches = re.findall(r'\\boxed{([0-9.]+)}', entry.response)
            if matches:
                entry.pred_score = float(matches[-1])  # Take the last match
            else:
                entry.pred_score = 0.0
                self.unmatched_indices.append(i)
        
        print(f"Found {len(self.unmatched_indices)} entries without \\boxed{{}} pattern: {self.unmatched_indices}")
        if len(self.unmatched_indices) > 0:
            print(f"Unmatched percentage: {len(self.unmatched_indices)/len(self.entries)*100:.2f}%")
    
    def _compute_basic_metrics(self, entries: List[IMO2025ResultEntry]) -> Dict[str, Any]:
        """Compute accuracy and error metrics."""
        solved = [entry.pred_score == entry.points_judge_1 for entry in entries]
        accuracy = sum(solved) / len(solved) if len(solved) > 0 else 0
        
        # Normalized L1 and L2 scores
        if len(entries) > 0:
            l1_score = sum([
                abs(entry.pred_score - entry.points_judge_1) / entry.max_points_judge_1 
                for entry in entries
            ]) / len(entries)
            
            l2_score = sum([
                ((entry.pred_score - entry.points_judge_1) / entry.max_points_judge_1) ** 2 
                for entry in entries
            ]) / len(entries)
        else:
            l1_score = 0
            l2_score = 0
        
        print(f"Accuracy: {accuracy}")
        print(f"L1 score: {l1_score}")
        print(f"L2 score: {l2_score}")
        
        return {
            'accuracy': accuracy,
            'l1_score': l1_score,
            'l2_score': l2_score,
            'solved': solved
        }
    
    def _merge_sample_jsons(self, analysis_dir: Path) -> None:
        """Merge all individual JSON files into one consolidated file with sample_idx."""
        # Create merged data list
        merged_data = []
        
        # Get all JSON files in the directory (excluding summary.json and analysis files)
        json_files = []
        for file_path in self.directory.glob("*.json"):
            if file_path.name not in ["summary.json"]:  # Skip summary and analysis files
                try:
                    # Check if filename (without extension) is numeric (sample ID)
                    sample_idx = int(file_path.stem)
                    json_files.append((sample_idx, file_path))
                except ValueError:
                    # Skip non-numeric JSON files
                    continue
        
        # Sort by sample index
        json_files.sort(key=lambda x: x[0])
        
        # Load and merge each JSON file
        for sample_idx, file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Add sample_idx to the data
                data['sample_idx'] = sample_idx
                merged_data.append(data)
                
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")
                continue
        
        # Save merged data to analysis directory
        merged_file_path = analysis_dir / "merged_samples.json"
        with open(merged_file_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, indent=2, ensure_ascii=False)
        
        print(f"Merged {len(merged_data)} samples into {merged_file_path}")

    def _generate_csv_report(self, analysis_dir: Path) -> None:
        """Generate CSV report compatible with eval_model_human function."""
        # Create DataFrame with required columns
        data = []
        for i, entry in enumerate(self.entries):
            data.append({
                'problem_idx': entry.problem_idx,
                'idx_answer': i,  # Sequential answer index
                'model_name': entry.model_name,
                'points_judge_1': entry.points_judge_1,
                'model_answers': entry.pred_score if entry.pred_score is not None else 0.0,
                # Additional columns that might be useful
                'points_judge_2': entry.points_judge_2,
                'max_points_judge_1': entry.max_points_judge_1,
                'max_points_judge_2': entry.max_points_judge_2,
                'response': entry.response,
                'answer': entry.answer,
                'problem': entry.problem,
                'model_config': entry.model_config,
                'input_tokens': entry.input_tokens,
                'output_tokens': entry.output_tokens,
                'cost': entry.cost,
                'source_file': entry.source_file
            })
        
        df = pd.DataFrame(data)
        
        # Save CSV report to analysis directory
        csv_path = analysis_dir / f"{self.name}_report.csv"
        df.to_csv(csv_path, index=False)
        print(f"CSV report saved to: {csv_path}")

    def _save_unmatched_responses(self, analysis_dir: Path) -> None:
        """Save unmatched responses for visualization."""
        unmatched_responses = []
        
        for idx in self.unmatched_indices:
            if idx < len(self.entries):
                entry = self.entries[idx]
                # Extract sample index from source_file (e.g., "/path/to/4.json" -> 4)
                try:
                    sample_idx = int(Path(entry.source_file).stem)
                except (ValueError, AttributeError):
                    # Fallback to list index if source_file parsing fails
                    sample_idx = idx
                
                unmatched_responses.append({
                    "id": sample_idx,
                    "response": entry.response
                })
        
        # Save unmatched responses
        unmatched_path = analysis_dir / "unmatched_responses.json"
        with open(unmatched_path, 'w', encoding='utf-8') as f:
            json.dump(unmatched_responses, f, indent=2, ensure_ascii=False)
        
        print(f"Unmatched responses saved to: {unmatched_path}")

    def analyze(self) -> Dict[str, Any]:
        """Analyze results: extract scores, generate merged JSON and CSV."""
        print(self.directory)
        
        # Create analysis directory first
        analysis_dir = self.directory / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Extract predicted scores
        self._extract_predicted_scores()
        
        # Step 2: Merge individual JSON files into consolidated file
        self._merge_sample_jsons(analysis_dir)
        
        # Step 3: Generate CSV report for eval_model_human compatibility
        self._generate_csv_report(analysis_dir)
        
        # Step 4: Save unmatched responses for visualization
        self._save_unmatched_responses(analysis_dir)
        
        # Step 5: Compute basic metrics
        basic_metrics = self._compute_basic_metrics(self.entries)
        
        # Create summary with basic information
        summary = {
            'num_samples': len(self.entries),
            'unmatched_indices': self.unmatched_indices,
            'num_unmatched': len(self.unmatched_indices),
            'unmatched_percentage': len(self.unmatched_indices) / len(self.entries) * 100 if len(self.entries) > 0 else 0,
            **basic_metrics
        }
        
        # Remove 'solved' from final summary (internal use only)
        summary.pop('solved', None)
        
        # Step 6: Save basic summary
        with open(analysis_dir / 'basic_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nBasic analysis summary saved to {analysis_dir / 'basic_summary.json'}")
        
        return summary