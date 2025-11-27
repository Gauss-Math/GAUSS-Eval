from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import importlib.util
import re
import numpy as np
import matplotlib.pyplot as plt
import json
from collections import Counter, defaultdict

# Import abstract base classes directly to avoid triggering data.__init__.py
base_path = Path(__file__).parent.parent / "base.py"
spec = importlib.util.spec_from_file_location("base", base_path)
base_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base_module)
AbstractResultEntry = base_module.AbstractResultEntry
AbstractResultSet = base_module.AbstractResultSet


@dataclass
class DEBUGResultEntry(AbstractResultEntry):
    """DEBUG dataset result entry."""
    
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
    def from_json_data(cls, data: Dict[str, Any], source_file: str = "") -> 'DEBUGResultEntry':
        """Create a DEBUGResultEntry from JSON data."""
        return cls(
            response=data.get('response', ''),
            pred_score=None,
            problem_idx=str(data.get('problem_idx', data.get('id', ''))),
            problem=data.get('problem', ''),
            model_name=data.get('model_name', ''),
            model_config=data.get('model_config', ''),
            answer=data.get('answer', ''),
            points_judge_1=float(data.get('points_judge_1', 0)),
            points_judge_2=float(data.get('points_judge_2', 0)),
            max_points_judge_1=float(data.get('max_points_judge_1', 0)),
            max_points_judge_2=float(data.get('max_points_judge_2', 0)),
            grading_details_judge_1=data.get('grading_details_judge_1', []),
            grading_details_judge_2=data.get('grading_details_judge_2', []),
            error_judge_1=data.get('error_judge_1'),
            error_judge_2=data.get('error_judge_2'),
            input_tokens=int(data.get('input_tokens', 0)),
            output_tokens=int(data.get('output_tokens', 0)),
            cost=float(data.get('cost', 0)),
            source_file=source_file
        )


@dataclass
class DEBUGResultSet(AbstractResultSet):
    """DEBUG dataset result set."""
    
    name: str
    directory: Path
    entries: List[DEBUGResultEntry] = field(default_factory=list)
    summary: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
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
        for entry in self.entries:
            pred_score = re.search(r'\\boxed{([0-9.]+)}', entry.response)
            entry.pred_score = float(pred_score.group(1)) if pred_score else 0.0
    
    def _compute_basic_metrics(self, entries: List[DEBUGResultEntry]) -> Dict[str, Any]:
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
    
    def _identify_divergent_samples(self, entries: List[DEBUGResultEntry]) -> Dict[str, Any]:
        """Identify samples with different types of errors."""
        solved = [entry.pred_score == entry.points_judge_1 for entry in entries]
        wrong_sample_idx = [i for i, s in enumerate(solved) if not s]
        
        # Diverged samples by different thresholds
        diverged_by_values = {}
        for diff in [3, 4, 5, 6, 7]:
            diverged_idx = [
                i for i, entry in enumerate(entries) 
                if abs(entry.pred_score - entry.points_judge_1) >= diff
            ]
            diverged_by_values[diff] = diverged_idx
        
        return {
            'wrong_sample_idx': wrong_sample_idx,
            'diverged_by_values': diverged_by_values
        }
    
    def _create_bin_structure(self, all_pred_scores: List[float], 
                             all_actual_scores: List[float]) -> Tuple[np.ndarray, List[str], int]:
        """Create bins for score distribution analysis."""
        unique_scores = sorted(set(all_pred_scores + all_actual_scores))
        
        # Integer bins
        if all(score == int(score) for score in unique_scores):
            min_score = int(min(unique_scores))
            max_score = int(max(unique_scores))
            num_bins = max_score - min_score + 1
            bin_edges = np.arange(min_score, max_score + 1.5, 1)
            bin_labels = [str(int(i)) for i in range(min_score, max_score + 1)]
        else:
            # Float bins
            min_score = min(unique_scores)
            max_score = max(unique_scores)
            
            if len(unique_scores) <= 20:
                bin_labels = [f"{score:.1f}" for score in unique_scores]
                num_bins = len(unique_scores)
                bin_edges = []
                for i in range(len(unique_scores)):
                    if i == 0:
                        bin_edges.append(unique_scores[i] - 0.05)
                    else:
                        bin_edges.append((unique_scores[i-1] + unique_scores[i]) / 2)
                bin_edges.append(unique_scores[-1] + 0.05)
                bin_edges = np.array(bin_edges)
            else:
                num_bins = 20
                bin_edges = np.linspace(min_score - 0.05, max_score + 0.05, num_bins + 1)
                bin_labels = [f"{edge:.1f}" for edge in bin_edges[:-1]]
        
        return bin_edges, bin_labels, num_bins
    
    def _create_agreement_matrix(self, entries: List[DEBUGResultEntry], 
                                 bin_edges: np.ndarray, num_bins: int) -> np.ndarray:
        """Create agreement matrix between predicted and actual scores."""
        agreement_matrix = np.zeros((num_bins, num_bins))
        
        for entry in entries:
            pred = entry.pred_score if entry.pred_score is not None else 0
            actual = entry.points_judge_1
            
            pred_bin = np.digitize([pred], bin_edges)[0] - 1
            actual_bin = np.digitize([actual], bin_edges)[0] - 1
            
            pred_bin = max(0, min(num_bins - 1, pred_bin))
            actual_bin = max(0, min(num_bins - 1, actual_bin))
            
            agreement_matrix[actual_bin, pred_bin] += 1
        
        return agreement_matrix
    
    def _plot_heatmap(self, matrix: np.ndarray, bin_labels: List[str], 
                     filename: str, title: str, xlabel: str, ylabel: str,
                     output_dir: Path) -> None:
        """Generic heatmap plotting function."""
        num_bins = len(bin_labels)
        fig_width = max(10, num_bins * 0.8)
        fig_height = max(8, num_bins * 0.6)
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        im = ax.imshow(matrix, cmap='YlOrRd', origin='lower', aspect='auto')
        
        # Add text annotations
        for i in range(num_bins):
            for j in range(num_bins):
                count = int(matrix[i, j])
                if count > 0:
                    color = 'white' if matrix[i, j] > matrix.max()/2 else 'black'
                    fontsize = max(6, min(10, 200 // num_bins))
                    ax.text(j, i, str(count), ha='center', va='center', 
                           color=color, fontsize=fontsize, fontweight='bold')
        
        plt.colorbar(im, ax=ax, label='Count')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(range(num_bins))
        ax.set_xticklabels(bin_labels, rotation=45 if num_bins > 10 else 0, 
                           ha='right' if num_bins > 10 else 'center')
        ax.set_yticks(range(num_bins))
        ax.set_yticklabels(bin_labels)
        ax.grid(False)
        plt.tight_layout()
        
        output_path = output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Heatmap saved to {output_path}")
    
    def _plot_scatter(self, human_scores: List[float], model_scores: List[float], 
                     counts: List[int], filename: str, output_dir: Path) -> None:
        """Plot scatter plot with bubble sizes."""
        max_count = max(counts)
        sizes = [200 * (count / max_count) ** 0.7 for count in counts]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(human_scores, model_scores, s=sizes, alpha=0.6, 
                            c=counts, cmap='Blues', edgecolors='black', linewidth=0.5)
        
        # Diagonal reference line
        score_min = min(min(human_scores), min(model_scores))
        score_max = max(max(human_scores), max(model_scores))
        ax.plot([score_min, score_max], [score_min, score_max], 
                'r--', alpha=0.5, linewidth=2, label='Perfect Agreement')
        
        plt.colorbar(scatter, ax=ax, label='Number of Samples')
        ax.set_xlabel('Human Score', fontsize=12)
        ax.set_ylabel('Model Score (numerator)', fontsize=12)
        ax.set_title('Distribution of Model Scores by Human Score (Scatter Plot)', 
                     fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Scatter plot saved to {output_path}")
    
    def _plot_distribution(self, score_counts: Counter, filename: str, 
                          title: str, xlabel: str, color: str, output_dir: Path) -> None:
        """Generic distribution bar plot function."""
        scores = sorted(score_counts.keys())
        counts = [score_counts[score] for score in scores]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(scores, counts, alpha=0.7, color=color, 
               edgecolor='black', linewidth=1.2)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Handle x-axis labels
        if len(scores) <= 20:
            ax.set_xticks(scores)
        else:
            tick_indices = np.linspace(0, len(scores)-1, min(20, len(scores)), dtype=int)
            ax.set_xticks([scores[i] for i in tick_indices])
            ax.set_xticklabels([scores[i] for i in tick_indices], rotation=45, ha='right')
        
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        output_path = output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Distribution plot saved to {output_path}")
    
    def _plot_average_bar_chart(self, averages: Dict[str, float], filename: str,
                               title: str, xlabel: str, ylabel: str, color: str,
                               output_dir: Path) -> None:
        """Generic bar chart for average scores."""
        labels = list(averages.keys())
        values = list(averages.values())
        
        # Determine figure size based on number of bars
        fig_width = max(10, len(labels) * 0.5)
        fig, ax = plt.subplots(figsize=(fig_width, 6))
        
        bars = ax.bar(range(len(labels)), values, alpha=0.7, color=color,
                      edgecolor='black', linewidth=1.2)
        
        # Add value labels on top of bars
        for i, (bar, value) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45 if len(labels) > 10 else 0, 
                           ha='right' if len(labels) > 10 else 'center')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        output_path = output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Bar chart saved to {output_path}")
    
    def _compute_average_scores_by_model(self, entries: List[DEBUGResultEntry]) -> Dict[str, float]:
        """Compute average human scores per model."""
        model_scores = defaultdict(list)
        
        for entry in entries:
            model_scores[entry.model_name].append(entry.points_judge_1)
        
        return {model: np.mean(scores) for model, scores in model_scores.items()}
    
    def _compute_average_scores_by_problem(self, entries: List[DEBUGResultEntry]) -> Dict[str, float]:
        """Compute average human scores per problem."""
        problem_scores = defaultdict(list)
        
        for entry in entries:
            problem_scores[entry.problem_idx].append(entry.points_judge_1)
        
        # Sort by problem index (convert to int if possible for proper sorting)
        try:
            sorted_problems = sorted(problem_scores.keys(), key=lambda x: int(x))
        except ValueError:
            sorted_problems = sorted(problem_scores.keys())
        
        return {problem: np.mean(problem_scores[problem]) for problem in sorted_problems}
    
    def _plot_bias_distribution(self, entries: List[DEBUGResultEntry], 
                               output_dir: Path) -> Dict[str, Any]:
        """Plot distribution of score differences (Model - Human)."""
        score_diffs = [
            (entry.pred_score if entry.pred_score is not None else 0) - entry.points_judge_1
            for entry in entries
        ]
        
        diff_counts = Counter(score_diffs)
        diffs = sorted(diff_counts.keys())
        counts = [diff_counts[diff] for diff in diffs]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Color bars based on bias direction
        colors = ['red' if diff < 0 else 'green' if diff > 0 else 'gray' for diff in diffs]
        bars = ax.bar(diffs, counts, alpha=0.7, edgecolor='black', linewidth=1.2, color=colors)
        
        # Add value labels on top of bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            if count > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        str(int(count)), ha='center', va='bottom', fontsize=8)
        
        # Add vertical line at zero
        ax.axvline(x=0, color='black', linestyle='--', linewidth=2, alpha=0.5, label='No Bias')
        
        ax.set_xlabel('Score Difference (Model - Human)', fontsize=12)
        ax.set_ylabel('Sample Count', fontsize=12)
        ax.set_title('Bias Distribution: Model Score - Human Score', fontsize=14, fontweight='bold')
        
        # Handle x-axis labels
        if len(diffs) <= 30:
            ax.set_xticks(diffs)
        else:
            tick_indices = np.linspace(0, len(diffs)-1, min(30, len(diffs)), dtype=int)
            ax.set_xticks([diffs[i] for i in tick_indices])
            ax.set_xticklabels([diffs[i] for i in tick_indices], rotation=45, ha='right')
        
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()
        plt.tight_layout()
        
        output_path = output_dir / 'bias_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Bias distribution plot saved to {output_path}")
        
        # Compute and print bias statistics
        mean_bias = np.mean(score_diffs)
        median_bias = np.median(score_diffs)
        std_bias = np.std(score_diffs)
        
        print(f"Mean bias (Model - Human): {mean_bias:.4f}")
        print(f"Median bias (Model - Human): {median_bias:.4f}")
        print(f"Std deviation of bias: {std_bias:.4f}")
        
        return {
            'mean_bias': mean_bias,
            'median_bias': median_bias,
            'std_bias': std_bias,
            'bias_distribution': dict(diff_counts)
        }
    
    def _analyze_subset(self, entries: List[DEBUGResultEntry], 
                       output_dir: Path, subset_name: str) -> Dict[str, Any]:
        """Analyze a subset of entries and generate visualizations."""
        print(f"\n{'='*60}")
        print(f"Analyzing subset: {subset_name}")
        print(f"Number of samples: {len(entries)}")
        print(f"{'='*60}\n")
        
        if len(entries) == 0:
            print(f"Warning: No entries in subset '{subset_name}'. Skipping analysis.")
            return {}
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Compute basic metrics
        metrics = self._compute_basic_metrics(entries)
        
        # Step 2: Identify divergent samples
        divergence_info = self._identify_divergent_samples(entries)
        
        # Step 3: Prepare score data
        all_pred_scores = [entry.pred_score if entry.pred_score is not None else 0 
                          for entry in entries]
        all_actual_scores = [entry.points_judge_1 for entry in entries]
        
        # Step 4: Create bin structure
        bin_edges, bin_labels, num_bins = self._create_bin_structure(
            all_pred_scores, all_actual_scores
        )
        
        # Step 5: Create and plot agreement matrix
        agreement_matrix = self._create_agreement_matrix(entries, bin_edges, num_bins)
        self._plot_heatmap(
            agreement_matrix, bin_labels, 'agreement_matrix.png',
            'Agreement Matrix: Predicted vs Actual Scores',
            'Predicted Score', 'Actual Score (Judge 1)', output_dir
        )
        
        # Step 6: Calculate agreement accuracies
        total_sum = agreement_matrix.sum()
        diagonal_accuracy = np.trace(agreement_matrix) / total_sum if total_sum > 0 else 0
        off_by_one = sum(agreement_matrix[i, j] 
                        for i in range(num_bins) 
                        for j in range(num_bins) 
                        if abs(i - j) <= 1)
        off_by_one_accuracy = off_by_one / total_sum if total_sum > 0 else 0
        
        print(f"Diagonal accuracy (exact matches): {diagonal_accuracy:.4f}")
        print(f"Off-by-one accuracy: {off_by_one_accuracy:.4f}")
        
        # Step 7: Create scatter plot
        score_pairs = [(entry.points_judge_1, 
                       entry.pred_score if entry.pred_score is not None else 0) 
                      for entry in entries]
        pair_counts = Counter(score_pairs)
        unique_pairs = list(pair_counts.keys())
        human_scores = [pair[0] for pair in unique_pairs]
        model_scores = [pair[1] for pair in unique_pairs]
        counts = [pair_counts[pair] for pair in unique_pairs]
        
        self._plot_scatter(human_scores, model_scores, counts, 
                          'score_distribution_scatter.png', output_dir)
        
        # Step 8: Create distribution plots
        human_score_counts = Counter(all_actual_scores)
        pred_score_counts = Counter(all_pred_scores)
        
        self._plot_distribution(
            human_score_counts, 'human_score_distribution.png',
            'Distribution of Human Scores', 'Human Score (Judge 1)', 'steelblue',
            output_dir
        )
        
        self._plot_distribution(
            pred_score_counts, 'predicted_score_distribution.png',
            'Distribution of Predicted Scores', 'Predicted Score', 'coral',
            output_dir
        )
        
        # Step 9: Plot average scores by model
        avg_scores_by_model = self._compute_average_scores_by_model(entries)
        self._plot_average_bar_chart(
            avg_scores_by_model, 'average_score_by_model.png',
            'Average Human Score by Model', 'Model Name', 'Average Human Score (Judge 1)',
            'royalblue', output_dir
        )
        
        # Step 10: Plot average scores by problem
        avg_scores_by_problem = self._compute_average_scores_by_problem(entries)
        self._plot_average_bar_chart(
            avg_scores_by_problem, 'average_score_by_problem.png',
            'Average Human Score by Problem', 'Problem Index', 'Average Human Score (Judge 1)',
            'mediumseagreen', output_dir
        )
        
        # Step 11: Plot bias distribution
        bias_stats = self._plot_bias_distribution(entries, output_dir)
        
        # Step 12: Compile analysis summary
        analysis_summary = {
            'subset_name': subset_name,
            'num_samples': len(entries),
            **metrics,
            'diagonal_accuracy': diagonal_accuracy,
            'off_by_one_accuracy': off_by_one_accuracy,
            'num_bins': num_bins,
            'bin_labels': bin_labels,
            'human_score_distribution': dict(human_score_counts),
            'pred_score_distribution': dict(pred_score_counts),
            'average_scores_by_model': avg_scores_by_model,
            'average_scores_by_problem': avg_scores_by_problem,
            **bias_stats,
            **divergence_info
        }
        
        # Add divergence statistics
        for diff in [3, 4, 5, 6, 7]:
            diverged_idx = divergence_info['diverged_by_values'][diff]
            analysis_summary[f'diverged_sample_idx_by_values_{diff}'] = diverged_idx
            analysis_summary[f'diverged_sample_count_by_values_{diff}'] = len(diverged_idx)
            analysis_summary[f'diverged_sample_percentage_by_values_{diff}'] = (
                len(diverged_idx) / len(entries) if len(entries) > 0 else 0
            )
        
        # Remove 'solved' from final summary (internal use only)
        analysis_summary.pop('solved', None)
        
        # Step 13: Save analysis summary
        with open(output_dir / 'analysis_summary.json', 'w') as f:
            json.dump(analysis_summary, f, indent=2)
        
        print(f"Analysis summary saved to {output_dir / 'analysis_summary.json'}")
        
        return analysis_summary
    
    def _merge_sample_jsons(self) -> None:
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
        
        # Save merged data
        merged_file_path = self.directory / "merged_samples.json"
        with open(merged_file_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, indent=2, ensure_ascii=False)
        
        print(f"Merged {len(merged_data)} samples into {merged_file_path}")

    def analyze(self) -> Dict[str, Any]:
        """Analyze results and generate visualizations."""
        print(self.directory)
        
        # Step 1: Extract predicted scores
        self._extract_predicted_scores()
        
        # Step 2: Merge individual JSON files into consolidated file
        self._merge_sample_jsons()
        
        # Step 3: Analyze full dataset
        analysis_dir = self.directory / "analysis"
        full_summary = self._analyze_subset(
            self.entries, 
            analysis_dir, 
            "Full Dataset"
        )
        
        # Step 4: Analyze filtered dataset (points_judge_1 > 0)
        filtered_entries = [entry for entry in self.entries if entry.points_judge_1 > 0]
        filtered_dir = analysis_dir / "human_larger_0"
        filtered_summary = self._analyze_subset(
            filtered_entries,
            filtered_dir,
            "Filtered Dataset (Human Score > 0)"
        )
        
        # Step 5: Combine summaries
        combined_summary = {
            'full_dataset': full_summary,
            'filtered_dataset_human_gt_0': filtered_summary
        }
        
        # Save combined summary
        with open(analysis_dir / 'combined_analysis_summary.json', 'w') as f:
            json.dump(combined_summary, f, indent=2)
        
        print(f"\nCombined analysis summary saved to {analysis_dir / 'combined_analysis_summary.json'}")
        
        return combined_summary