import sys
import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog

# Ensure project root is in path for imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
try:
    from litellm import completion
except ImportError:
    print("Warning: litellm not available. Model querying will not work.")
    def completion(*args, **kwargs):
        raise ImportError("litellm is not installed. Please install it to use model querying functionality.")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data import get_dataset_fn
from src.prompt_utils import get_current_prompt, set_current_prompt


class RunOneExampleGUI:
    """GUI for running individual examples from datasets."""
    
    def __init__(self, root=None):
        """Initialize the GUI."""
        if root is None:
            self.root = tk.Tk()
            self._root_owner = True
        else:
            self.root = root
            self._root_owner = False
            
        self.root.title("Run One Example")
        self.root.geometry("1200x900")
        
        # Load global config
        self.global_config = self._load_global_config()
        
        # State variables
        self.selected_dataset = None
        self.selected_example_id = None
        self.selected_problem_idx = None
        self.current_example_data = None
        self.current_examples_data = []  # For multiple examples when using problem_idx
        self.model_name = tk.StringVar(value="openrouter/openai/gpt-4o-mini")
        
        self._setup_ui()
        self._load_datasets()
        
    def _load_global_config(self) -> Dict[str, Any]:
        """Load global configuration."""
        config_path = project_root / "global_config.json"
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load global config: {e}")
            return {
                "sampling_params": {
                    "temperature": 0.1,
                    "max_tokens": 32768,
                    "top_p": 1.0,
                    "seed": 1234
                }
            }
    
    def _setup_ui(self):
        """Set up the user interface."""
        # Main container with scrollable frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        
        # Title
        title_label = ttk.Label(
            main_frame, 
            text="Run One Example", 
            font=("Arial", 16, "bold")
        )
        title_label.grid(row=0, column=0, pady=(0, 15))
        
        # Dataset selection section
        self._setup_dataset_section(main_frame, row=1)
        
        # Example selection section
        self._setup_example_section(main_frame, row=2)
        
        # Example details section
        self._setup_details_section(main_frame, row=3)
        
        # Prompt configuration section
        self._setup_prompt_section(main_frame, row=4)
        
        # Model configuration section
        self._setup_model_section(main_frame, row=5)
        
        # Run button section
        self._setup_run_section(main_frame, row=6)
        
        # Results section
        self._setup_results_section(main_frame, row=7)
        
    def _setup_dataset_section(self, parent, row):
        """Set up dataset selection section."""
        dataset_frame = ttk.LabelFrame(parent, text="Dataset Selection", padding="10")
        dataset_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        dataset_frame.columnconfigure(1, weight=1)
        
        ttk.Label(dataset_frame, text="Dataset:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        
        self.dataset_combo = ttk.Combobox(dataset_frame, state="readonly", width=30)
        self.dataset_combo.grid(row=0, column=1, sticky=(tk.W, tk.E))
        self.dataset_combo.bind('<<ComboboxSelected>>', self._on_dataset_selected)
        
        # Dataset info label
        self.dataset_info_label = ttk.Label(dataset_frame, text="", foreground="blue")
        self.dataset_info_label.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))
        
    def _setup_example_section(self, parent, row):
        """Set up example selection section."""
        example_frame = ttk.LabelFrame(parent, text="Example Selection", padding="10")
        example_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        example_frame.columnconfigure(1, weight=1)
        
        # Selection mode radio buttons
        self.selection_mode = tk.StringVar(value="example_id")
        
        ttk.Radiobutton(
            example_frame, 
            text="By Example ID", 
            variable=self.selection_mode, 
            value="example_id",
            command=self._on_selection_mode_changed
        ).grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        ttk.Radiobutton(
            example_frame, 
            text="By Problem Index", 
            variable=self.selection_mode, 
            value="problem_idx",
            command=self._on_selection_mode_changed
        ).grid(row=0, column=1, sticky=tk.W, pady=(0, 5))
        
        # Example ID input
        ttk.Label(example_frame, text="Example ID:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10))
        
        self.example_id_entry = ttk.Entry(example_frame, width=20)
        self.example_id_entry.grid(row=1, column=1, sticky=tk.W, padx=(0, 10))
        self.example_id_entry.bind('<KeyRelease>', self._on_example_id_changed)
        
        ttk.Button(
            example_frame, 
            text="Load Example", 
            command=self._load_example
        ).grid(row=1, column=2, padx=(10, 0))
        
        # Problem index input
        ttk.Label(example_frame, text="Problem Index:").grid(row=2, column=0, sticky=tk.W, padx=(0, 10))
        
        self.problem_idx_entry = ttk.Entry(example_frame, width=20, state=tk.DISABLED)
        self.problem_idx_entry.grid(row=2, column=1, sticky=tk.W, padx=(0, 10))
        self.problem_idx_entry.bind('<KeyRelease>', self._on_problem_idx_changed)
        
        ttk.Button(
            example_frame, 
            text="Load by Problem Index", 
            command=self._load_by_problem_idx,
            state=tk.DISABLED
        ).grid(row=2, column=2, padx=(10, 0))
        self.load_problem_idx_button = example_frame.grid_slaves(row=2, column=2)[0]
        
        # Example range info
        self.example_range_label = ttk.Label(example_frame, text="", foreground="gray")
        self.example_range_label.grid(row=3, column=0, columnspan=3, sticky=tk.W, pady=(5, 0))
        
        # Problem index info
        self.problem_idx_info_label = ttk.Label(example_frame, text="", foreground="gray")
        self.problem_idx_info_label.grid(row=4, column=0, columnspan=3, sticky=tk.W, pady=(5, 0))
        
    def _setup_details_section(self, parent, row):
        """Set up example details section."""
        details_frame = ttk.LabelFrame(parent, text="Example Details", padding="10")
        details_frame.grid(row=row, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        details_frame.columnconfigure(0, weight=1)
        details_frame.rowconfigure(0, weight=1)
        
        # Scrollable text widget for example details
        self.details_text = scrolledtext.ScrolledText(
            details_frame,
            wrap=tk.WORD,
            width=80,
            height=8,
            font=("Consolas", 10),
            state=tk.DISABLED
        )
        self.details_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
    def _setup_prompt_section(self, parent, row):
        """Set up prompt configuration section."""
        prompt_frame = ttk.LabelFrame(parent, text="Prompt Configuration", padding="10")
        prompt_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        prompt_frame.columnconfigure(0, weight=1)
        
        # Current prompt display
        self.current_prompt_label = ttk.Label(
            prompt_frame, 
            text="Current prompts loaded from GUI prompting system", 
            foreground="green"
        )
        self.current_prompt_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        # Buttons for prompt and rubric editing
        button_frame = ttk.Frame(prompt_frame)
        button_frame.grid(row=1, column=0, sticky=tk.W)
        
        ttk.Button(
            button_frame, 
            text="Edit Prompts", 
            command=self._open_prompt_editor
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            button_frame, 
            text="Edit Rubrics", 
            command=self._open_rubric_editor
        ).pack(side=tk.LEFT)
        
    def _setup_model_section(self, parent, row):
        """Set up model configuration section."""
        model_frame = ttk.LabelFrame(parent, text="Model Configuration", padding="10")
        model_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        model_frame.columnconfigure(1, weight=1)
        
        ttk.Label(model_frame, text="Model:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        
        model_entry = ttk.Entry(model_frame, textvariable=self.model_name, width=50)
        model_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        
        # Sampling parameters display
        params_text = f"Sampling params: {json.dumps(self.global_config['sampling_params'], indent=2)}"
        self.params_label = ttk.Label(
            model_frame, 
            text=params_text, 
            foreground="blue",
            font=("Consolas", 9)
        )
        self.params_label.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))
        
    def _setup_run_section(self, parent, row):
        """Set up run button section."""
        run_frame = ttk.Frame(parent)
        run_frame.grid(row=row, column=0, pady=10)
        
        self.run_button = ttk.Button(
            run_frame,
            text="Run Example",
            command=self._run_example,
            state=tk.DISABLED
        )
        self.run_button.pack(side=tk.LEFT, padx=5)
        
        self.save_button = ttk.Button(
            run_frame,
            text="Save Result",
            command=self._save_result,
            state=tk.DISABLED
        )
        self.save_button.pack(side=tk.LEFT, padx=5)
        
        # Save to results directory button
        self.save_to_results_button = ttk.Button(
            run_frame,
            text="Save to Results Dir",
            command=self._save_to_results_dir,
            state=tk.DISABLED
        )
        self.save_to_results_button.pack(side=tk.LEFT, padx=5)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(run_frame, textvariable=self.status_var, foreground="blue")
        self.status_label.pack(side=tk.LEFT, padx=20)
        
    def _setup_results_section(self, parent, row):
        """Set up results display section."""
        results_frame = ttk.LabelFrame(parent, text="Results", padding="10")
        results_frame.grid(row=row, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Configure main frame to expand results section
        parent.rowconfigure(row, weight=1)
        
        # Scrollable text widget for results
        self.results_text = scrolledtext.ScrolledText(
            results_frame,
            wrap=tk.WORD,
            width=80,
            height=12,
            font=("Consolas", 10),
            state=tk.DISABLED
        )
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Store the last result for saving
        self.last_result = None
        
    def _load_datasets(self):
        """Load available datasets."""
        dataset_names = list(get_dataset_fn.keys())
        self.dataset_combo['values'] = dataset_names
        if dataset_names:
            self.dataset_combo.set(dataset_names[0])
            self._on_dataset_selected()
            
    def _on_dataset_selected(self, event=None):
        """Handle dataset selection."""
        dataset_name = self.dataset_combo.get()
        if dataset_name in get_dataset_fn:
            self.selected_dataset = get_dataset_fn[dataset_name]
            info_text = f"Dataset: {self.selected_dataset.data_name}, Examples: {len(self.selected_dataset.data)}"
            self.dataset_info_label.config(text=info_text)
            
            # Update example range info
            if len(self.selected_dataset.data) > 0:
                range_text = f"Available example IDs: 0 to {len(self.selected_dataset.data) - 1}"
                self.example_range_label.config(text=range_text)
            else:
                self.example_range_label.config(text="No examples available")
                
            # Clear current example
            self._clear_example_details()
            
            # Update problem index info
            self._update_problem_idx_info()
            
    def _on_example_id_changed(self, event=None):
        """Handle example ID entry changes."""
        # Enable/disable load button based on input
        example_id_text = self.example_id_entry.get().strip()
        if example_id_text and self.selected_dataset:
            try:
                example_id = int(example_id_text)
                if 0 <= example_id < len(self.selected_dataset.data):
                    # Valid ID - could auto-load here if desired
                    pass
            except ValueError:
                pass
    
    def _on_selection_mode_changed(self):
        """Handle selection mode radio button changes."""
        mode = self.selection_mode.get()
        if mode == "example_id":
            # Enable example ID controls, disable problem index controls
            self.example_id_entry.config(state=tk.NORMAL)
            self.problem_idx_entry.config(state=tk.DISABLED)
            self.load_problem_idx_button.config(state=tk.DISABLED)
        else:  # problem_idx
            # Disable example ID controls, enable problem index controls
            self.example_id_entry.config(state=tk.DISABLED)
            self.problem_idx_entry.config(state=tk.NORMAL)
            self.load_problem_idx_button.config(state=tk.NORMAL)
    
    def _on_problem_idx_changed(self, event=None):
        """Handle problem index entry changes."""
        # This could be used for validation or auto-completion in the future
        pass
    
    def _update_problem_idx_info(self):
        """Update the problem index information display."""
        if not self.selected_dataset:
            self.problem_idx_info_label.config(text="")
            return
            
        # Get available problem_idx values
        available_problem_idx = set()
        for example in self.selected_dataset.data:
            if 'problem_idx' in example:
                available_problem_idx.add(str(example['problem_idx']))
        
        if available_problem_idx:
            sorted_idx = sorted(available_problem_idx)
            if len(sorted_idx) <= 10:
                info_text = f"Available problem_idx values: {', '.join(sorted_idx)}"
            else:
                info_text = f"Available problem_idx values: {', '.join(sorted_idx[:10])}... ({len(sorted_idx)} total)"
            self.problem_idx_info_label.config(text=info_text)
        else:
            self.problem_idx_info_label.config(text="No problem_idx field found in dataset")
                
    def _load_example(self):
        """Load the selected example."""
        if not self.selected_dataset:
            messagebox.showwarning("Warning", "Please select a dataset first.")
            return
            
        example_id_text = self.example_id_entry.get().strip()
        if not example_id_text:
            messagebox.showwarning("Warning", "Please enter an example ID.")
            return
            
        try:
            example_id = int(example_id_text)
        except ValueError:
            messagebox.showerror("Error", "Example ID must be a number.")
            return
            
        if not (0 <= example_id < len(self.selected_dataset.data)):
            messagebox.showerror(
                "Error", 
                f"Example ID must be between 0 and {len(self.selected_dataset.data) - 1}."
            )
            return
            
        # Load the example
        self.selected_example_id = example_id
        self.current_example_data = self.selected_dataset.data[example_id]
        
        # Display example details
        self._display_example_details()
        
        # Enable run button
        self.run_button.config(state=tk.NORMAL)
        self.status_var.set(f"Example {example_id} loaded")
        
    def _load_by_problem_idx(self):
        """Load examples by problem index."""
        if not self.selected_dataset:
            messagebox.showwarning("Warning", "Please select a dataset first.")
            return
            
        problem_idx_text = self.problem_idx_entry.get().strip()
        if not problem_idx_text:
            messagebox.showwarning("Warning", "Please enter a problem index.")
            return
        
        # Find all examples with matching problem_idx
        matching_examples = []
        for example in self.selected_dataset.data:
            if str(example.get('problem_idx', '')) == str(problem_idx_text):
                matching_examples.append(example)
        
        if not matching_examples:
            messagebox.showerror("Error", f"No examples found with problem_idx '{problem_idx_text}'")
            return
        
        # Store the matching examples
        self.selected_problem_idx = problem_idx_text
        self.current_examples_data = matching_examples
        self.current_example_data = None  # Clear single example data
        
        # Display summary of loaded examples
        self._display_problem_idx_details()
        
        # Enable run button
        self.run_button.config(state=tk.NORMAL)
        self.status_var.set(f"Loaded {len(matching_examples)} examples with problem_idx '{problem_idx_text}'")
        
    def _display_example_details(self):
        """Display details of the current example."""
        if not self.current_example_data:
            return
            
        self.details_text.config(state=tk.NORMAL)
        self.details_text.delete(1.0, tk.END)
        
        # Format example data for display
        details = f"Example ID: {self.selected_example_id}\n\n"
        
        # Show key fields
        for key, value in self.current_example_data.items():
            if key in ['id', 'problem', 'answer', 'max_points_judge_1', 'grading_details_judge_1']:
                if isinstance(value, (list, dict)):
                    details += f"{key}:\n{json.dumps(value, indent=2)}\n\n"
                else:
                    # Truncate very long values
                    str_value = str(value)
                    if len(str_value) > 500:
                        str_value = str_value[:500] + "... (truncated)"
                    details += f"{key}: {str_value}\n\n"
        
        # Show all other fields in a compact format
        other_fields = {k: v for k, v in self.current_example_data.items() 
                       if k not in ['id', 'problem', 'answer', 'max_points_judge_1', 'grading_details_judge_1']}
        if other_fields:
            details += "Other fields:\n"
            for key, value in other_fields.items():
                str_value = str(value)
                if len(str_value) > 100:
                    str_value = str_value[:100] + "..."
                details += f"  {key}: {str_value}\n"
        
        self.details_text.insert(1.0, details)
        self.details_text.config(state=tk.DISABLED)
        
    def _display_problem_idx_details(self):
        """Display details of the loaded examples by problem index."""
        if not self.current_examples_data:
            return
            
        self.details_text.config(state=tk.NORMAL)
        self.details_text.delete(1.0, tk.END)
        
        # Format summary for multiple examples
        details = f"Problem Index: {self.selected_problem_idx}\n"
        details += f"Number of examples: {len(self.current_examples_data)}\n\n"
        
        # Show summary of each example
        for i, example_data in enumerate(self.current_examples_data):
            details += f"--- Example {i+1} (ID: {example_data.get('id', 'N/A')}) ---\n"
            
            # Show key fields
            for key in ['problem', 'answer']:
                if key in example_data:
                    value = example_data[key]
                    if isinstance(value, str) and len(value) > 200:
                        value = value[:200] + "... (truncated)"
                    details += f"{key}: {value}\n"
            
            details += "\n"
        
        self.details_text.insert(1.0, details)
        self.details_text.config(state=tk.DISABLED)
        
    def _clear_example_details(self):
        """Clear example details display."""
        self.details_text.config(state=tk.NORMAL)
        self.details_text.delete(1.0, tk.END)
        self.details_text.config(state=tk.DISABLED)
        self.current_example_data = None
        self.current_examples_data = []
        self.selected_example_id = None
        self.selected_problem_idx = None
        self.run_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.DISABLED)
        self.save_to_results_button.config(state=tk.DISABLED)
        
    def _open_prompt_editor(self):
        """Open the prompt editor in a new window."""
        try:
            from src.gui_prompting import PromptGUI
            
            # Create new window for prompt editor
            prompt_window = tk.Toplevel(self.root)
            prompt_window.title("Prompt Editor")
            prompt_window.geometry("900x700")
            
            # Create prompt GUI in the new window
            prompt_gui = PromptGUI(prompt_window)
        except ImportError:
            messagebox.showwarning("Feature Unavailable", 
                                 "Prompt editor GUI is not available. Please use the web UI instead.")
        
    def _open_rubric_editor(self):
        """Open the rubric editor in a new window."""
        try:
            from src.gui_rubric import RubricGUI
            
            # Create new window for rubric editor
            rubric_window = tk.Toplevel(self.root)
            rubric_window.title("Rubric Editor")
            rubric_window.geometry("1000x700")
            
            # Create rubric GUI in the new window
            rubric_gui = RubricGUI(rubric_window)
        except ImportError:
            messagebox.showwarning("Feature Unavailable", 
                                 "Rubric editor GUI is not available. Please use the web UI instead.")
        
    def _run_example(self):
        """Run the current example(s)."""
        if not self.selected_dataset:
            messagebox.showwarning("Warning", "Please select a dataset first.")
            return
            
        # Check if we have single example or multiple examples
        if not self.current_example_data and not self.current_examples_data:
            messagebox.showwarning("Warning", "Please load an example or examples first.")
            return
            
        model = self.model_name.get().strip()
        if not model:
            messagebox.showwarning("Warning", "Please enter a model name.")
            return
        
        # Determine which examples to run
        examples_to_run = []
        if self.current_example_data:
            # Single example mode
            examples_to_run = [self.current_example_data]
        elif self.current_examples_data:
            # Multiple examples mode
            examples_to_run = self.current_examples_data
            
        self.status_var.set(f"Running {len(examples_to_run)} example(s)...")
        self.run_button.config(state=tk.DISABLED)
        self.root.update()
        
        results = []
        all_prompts = []
        
        try:
            for i, example_data in enumerate(examples_to_run):
                self.status_var.set(f"Running example {i+1}/{len(examples_to_run)}...")
                self.root.update()
                
                # Parse prompt using dataset's method
                prompt = self.selected_dataset.parse_prompt(example_data)
                
                # Query the model
                sampling_params = self.global_config['sampling_params'].copy()
                
                response = completion(
                    model=model,
                    messages=[
                        {"role": "system", "content": prompt['system_prompt']},
                        {"role": "user", "content": prompt['user_prompt']}
                    ],
                    **sampling_params
                )
                
                # Parse response using dataset's method
                result = self.selected_dataset.parse_response(response, example_data.copy())
                results.append(result)
                all_prompts.append(prompt)
            
            # Display results
            if len(results) == 1:
                # Single result
                self._display_results(results[0], all_prompts[0])
                self.last_result = results[0]
            else:
                # Multiple results
                self._display_multiple_results(results, all_prompts)
                self.last_result = results  # Store all results
            
            self.save_button.config(state=tk.NORMAL)
            self.save_to_results_button.config(state=tk.NORMAL)
            
            self.status_var.set(f"Completed {len(results)} example(s) successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to run example(s): {str(e)}")
            self.status_var.set("Error occurred")
            
        finally:
            self.run_button.config(state=tk.NORMAL)
            
    def _display_results(self, result, prompt):
        """Display the results of running the example."""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        
        # Format results for display
        results_text = "=== PROMPT ===\n"
        results_text += f"System: {prompt['system_prompt'][:200]}...\n\n"
        results_text += f"User: {prompt['user_prompt'][:500]}...\n\n"
        
        results_text += "=== RESPONSE ===\n"
        if 'response' in result:
            results_text += f"{result['response']}\n\n"
        
        results_text += "=== FULL RESULT ===\n"
        results_text += json.dumps(result, indent=2, ensure_ascii=False)
        
        self.results_text.insert(1.0, results_text)
        self.results_text.config(state=tk.DISABLED)
        
    def _display_multiple_results(self, results, prompts):
        """Display the results of running multiple examples."""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        
        # Format results for display
        results_text = f"=== MULTIPLE RESULTS ({len(results)} examples) ===\n\n"
        
        for i, (result, prompt) in enumerate(zip(results, prompts)):
            example_id = result.get('id', i)
            results_text += f"--- EXAMPLE {example_id} ({i+1}/{len(results)}) ---\n"
            results_text += f"System: {prompt['system_prompt'][:100]}...\n"
            results_text += f"User: {prompt['user_prompt'][:200]}...\n\n"
            
            if 'response' in result:
                response_preview = result['response'][:300]
                if len(result['response']) > 300:
                    response_preview += "... (truncated)"
                results_text += f"Response: {response_preview}\n\n"
            
            # Show key result fields
            for key in ['points_judge_1', 'max_points_judge_1', 'problem_idx']:
                if key in result:
                    results_text += f"{key}: {result[key]}\n"
            
            results_text += "\n" + "="*50 + "\n\n"
        
        self.results_text.insert(1.0, results_text)
        self.results_text.config(state=tk.DISABLED)
        
    def _save_result(self):
        """Save the last result to a file."""
        if not self.last_result:
            messagebox.showwarning("Warning", "No result to save.")
            return
            
        # Generate default filename
        dataset_name = self.dataset_combo.get() if hasattr(self, 'dataset_combo') else "unknown"
        model_name = self.model_name.get().split('/')[-1] if self.model_name.get() else "unknown"
        timestamp = int(time.time())
        
        # Handle different naming for single vs multiple results
        if isinstance(self.last_result, list):
            # Multiple results
            identifier = f"problem_idx_{self.selected_problem_idx}" if self.selected_problem_idx else "multiple"
            default_filename = f"results_{dataset_name}_{identifier}_{model_name}_{timestamp}.json"
        else:
            # Single result
            identifier = f"example_{self.selected_example_id}" if self.selected_example_id is not None else "single"
            default_filename = f"result_{dataset_name}_{identifier}_{model_name}_{timestamp}.json"
        
        # Ask user for save location
        file_path = filedialog.asksaveasfilename(
            title="Save Result",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile=default_filename
        )
        
        if not file_path:
            return
            
        try:
            # Create a comprehensive result object
            if isinstance(self.last_result, list):
                # Multiple results
                save_data = {
                    "metadata": {
                        "dataset": dataset_name,
                        "problem_idx": self.selected_problem_idx,
                        "num_examples": len(self.last_result),
                        "model": self.model_name.get(),
                        "timestamp": timestamp,
                        "sampling_params": self.global_config.get('sampling_params', {}),
                        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    },
                    "examples_data": self.current_examples_data,
                    "results": self.last_result
                }
            else:
                # Single result
                save_data = {
                    "metadata": {
                        "dataset": dataset_name,
                        "example_id": self.selected_example_id,
                        "model": self.model_name.get(),
                        "timestamp": timestamp,
                        "sampling_params": self.global_config.get('sampling_params', {}),
                        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    },
                    "example_data": self.current_example_data,
                    "result": self.last_result
                }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            
            messagebox.showinfo("Success", f"Result saved to:\n{file_path}")
            self.status_var.set(f"Result saved to {os.path.basename(file_path)}")
            
            # Optionally open the file location
            if messagebox.askyesno("Open Location", "Would you like to open the file location?"):
                import subprocess
                import platform
                
                try:
                    if platform.system() == "Darwin":  # macOS
                        subprocess.run(["open", "-R", file_path])
                    elif platform.system() == "Windows":
                        subprocess.run(["explorer", "/select,", file_path])
                    else:  # Linux
                        subprocess.run(["xdg-open", os.path.dirname(file_path)])
                except Exception as e:
                    print(f"Could not open file location: {e}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save result: {str(e)}")
    
    def _save_to_results_dir(self):
        """Save the result to the standard results directory structure."""
        if not self.last_result:
            messagebox.showwarning("Warning", "No result to save.")
            return
            
        try:
            # Create results directory structure similar to main.py
            dataset_name = self.dataset_combo.get() if hasattr(self, 'dataset_combo') else "unknown"
            model_name = self.model_name.get().split('/')[-1] if self.model_name.get() else "unknown"
            timestamp = int(time.time())
            
            if isinstance(self.last_result, list):
                # Multiple results
                run_name = f"{dataset_name}_{model_name}_{timestamp}_problem_idx_{self.selected_problem_idx}"
                results_dir = project_root / "results" / run_name
                results_dir.mkdir(parents=True, exist_ok=True)
                
                # Save each individual result
                saved_files = []
                for result in self.last_result:
                    result_id = result.get('id', 'unknown')
                    result_file = results_dir / f"{result_id}.json"
                    
                    # Create a result in the same format as the main evaluation system
                    save_data = result.copy()
                    
                    with open(result_file, 'w', encoding='utf-8') as f:
                        json.dump(save_data, f, indent=2, ensure_ascii=False)
                    saved_files.append(result_file.name)
                
                # Create a summary file
                summary_data = {
                    "run_name": run_name,
                    "dataset": dataset_name,
                    "model": self.model_name.get(),
                    "problem_idx_mode": True,
                    "problem_idx": self.selected_problem_idx,
                    "timestamp": timestamp,
                    "sampling_params": self.global_config.get('sampling_params', {}),
                    "saved_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    "total_examples": len(self.last_result),
                    "successful": len(self.last_result),
                    "failed": 0,
                    "saved_files": saved_files
                }
            else:
                # Single result
                run_name = f"{dataset_name}_{model_name}_{timestamp}_single_example"
                results_dir = project_root / "results" / run_name
                results_dir.mkdir(parents=True, exist_ok=True)
                
                # Save the individual result
                result_file = results_dir / f"{self.selected_example_id}.json"
                
                # Create a result in the same format as the main evaluation system
                save_data = self.last_result.copy()
                
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(save_data, f, indent=2, ensure_ascii=False)
                
                # Create a summary file
                summary_data = {
                    "run_name": run_name,
                    "dataset": dataset_name,
                    "model": self.model_name.get(),
                    "single_example": True,
                    "example_id": self.selected_example_id,
                    "timestamp": timestamp,
                    "sampling_params": self.global_config.get('sampling_params', {}),
                    "saved_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    "total_examples": 1,
                    "successful": 1,
                    "failed": 0
                }
            
            summary_file = results_dir / "summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            
            if isinstance(self.last_result, list):
                messagebox.showinfo(
                    "Success", 
                    f"Results saved to results directory:\n{results_dir}\n\nFiles created:\n- {len(saved_files)} result files\n- {summary_file.name}"
                )
                self.status_var.set(f"Results saved to results/{run_name}")
            else:
                messagebox.showinfo(
                    "Success", 
                    f"Result saved to results directory:\n{results_dir}\n\nFiles created:\n- {result_file.name}\n- {summary_file.name}"
                )
                self.status_var.set(f"Result saved to results/{run_name}")
            
            # Optionally open the results directory
            if messagebox.askyesno("Open Directory", "Would you like to open the results directory?"):
                import subprocess
                import platform
                
                try:
                    if platform.system() == "Darwin":  # macOS
                        subprocess.run(["open", str(results_dir)])
                    elif platform.system() == "Windows":
                        subprocess.run(["explorer", str(results_dir)])
                    else:  # Linux
                        subprocess.run(["xdg-open", str(results_dir)])
                except Exception as e:
                    print(f"Could not open results directory: {e}")
                    
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save to results directory: {str(e)}")
            
    def run(self):
        """Run the GUI application."""
        if self._root_owner:
            self.root.mainloop()


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description="Run individual examples from datasets")
    parser.add_argument("--dataset", type=str, help="Dataset name")
    parser.add_argument("--example-id", type=int, help="Example ID to run")
    parser.add_argument("--problem-idx", type=str, help="Problem index to run (all examples with this problem_idx)")
    parser.add_argument("--model", type=str, default="openrouter/openai/gpt-4o-mini", help="Model name")
    parser.add_argument("--gui", action="store_true", help="Launch GUI mode")
    
    args = parser.parse_args()
    
    if args.gui or (not args.dataset and not args.example_id and not args.problem_idx):
        # Launch GUI
        app = RunOneExampleGUI()
        app.run()
    else:
        # Command line mode
        if not args.dataset:
            print("Error: --dataset is required for command line mode")
            return
            
        if args.example_id is None and args.problem_idx is None:
            print("Error: Either --example-id or --problem-idx is required for command line mode")
            return
            
        if args.example_id is not None and args.problem_idx is not None:
            print("Error: Cannot specify both --example-id and --problem-idx")
            return
            
        # Load dataset
        if args.dataset not in get_dataset_fn:
            print(f"Error: Dataset '{args.dataset}' not found. Available: {list(get_dataset_fn.keys())}")
            return
            
        dataset = get_dataset_fn[args.dataset]
        
        # Load global config
        config_path = project_root / "global_config.json"
        try:
            with open(config_path, 'r') as f:
                global_config = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load global config: {e}")
            global_config = {"sampling_params": {"temperature": 0.1, "max_tokens": 32768}}
        
        # Determine which examples to run
        examples_to_run = []
        
        if args.example_id is not None:
            # Single example mode
            if not (0 <= args.example_id < len(dataset.data)):
                print(f"Error: Example ID must be between 0 and {len(dataset.data) - 1}")
                return
            examples_to_run = [dataset.data[args.example_id]]
            print(f"Running example {args.example_id} from dataset {args.dataset}")
            
        elif args.problem_idx is not None:
            # Problem index mode - find all examples with matching problem_idx
            for example in dataset.data:
                if str(example.get('problem_idx', '')) == str(args.problem_idx):
                    examples_to_run.append(example)
            
            if not examples_to_run:
                print(f"Error: No examples found with problem_idx '{args.problem_idx}'")
                # Show available problem_idx values
                available_problem_idx = set()
                for example in dataset.data:
                    if 'problem_idx' in example:
                        available_problem_idx.add(str(example['problem_idx']))
                if available_problem_idx:
                    print(f"Available problem_idx values: {sorted(available_problem_idx)}")
                return
                
            print(f"Running {len(examples_to_run)} examples with problem_idx '{args.problem_idx}' from dataset {args.dataset}")
        
        print(f"Model: {args.model}")
        
        # Run all selected examples
        results = []
        for i, example_data in enumerate(examples_to_run):
            try:
                example_id = example_data.get('id', i)
                print(f"\n--- Processing example {example_id} ({i+1}/{len(examples_to_run)}) ---")
                
                # Parse prompt
                prompt = dataset.parse_prompt(example_data)
                
                # Query model
                response = completion(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": prompt['system_prompt']},
                        {"role": "user", "content": prompt['user_prompt']}
                    ],
                    **global_config['sampling_params']
                )
                
                # Parse response
                result = dataset.parse_response(response, example_data.copy())
                results.append(result)
                
                # Print result for this example
                print(f"=== RESULT for example {example_id} ===")
                print(json.dumps(result, indent=2, ensure_ascii=False))
                
            except Exception as e:
                print(f"Error processing example {example_data.get('id', i)}: {e}")
                
        print(f"\n=== SUMMARY ===")
        print(f"Processed {len(results)} examples successfully out of {len(examples_to_run)} total")


if __name__ == "__main__":
    main()
