import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

# Ensure project root is in path for imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Global variable to store rubric update callback (optional)
_rubric_update_callbacks = []


def _get_rubric_file_path():
    """
    Get the path to the persistent rubric storage file.
    
    Returns:
        Path: Path to the rubric storage file
    """
    # Store in project root as .rubric_config.json
    project_root = Path(__file__).parent.parent
    return project_root / ".rubric_config.json"


def get_current_rubrics() -> Dict[str, List[Dict[str, Any]]]:
    """
    Get the current rubrics from persistent storage.
    Returns a dictionary mapping problem_idx to rubric list.
    Falls back to empty dict if file doesn't exist or has errors.
    
    Returns:
        dict: Dictionary mapping problem_idx to list of rubric items
    """
    rubric_file = _get_rubric_file_path()
    
    # Try to read from file first
    if rubric_file.exists():
        try:
            with open(rubric_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data
        except (json.JSONDecodeError, IOError, Exception) as e:
            print(f"Warning: Could not read rubric file: {e}")
    
    # Fallback to empty dict
    return {}


def get_rubric_for_problem(problem_idx: str) -> Optional[Union[List[Dict[str, Any]], str]]:
    """
    Get the rubric for a specific problem_idx.
    
    Args:
        problem_idx: The problem index to get rubric for
    
    Returns:
        Union[list, str]: The rubric (list of items or string) for the problem, or None if not found
    """
    rubrics = get_current_rubrics()
    return rubrics.get(problem_idx)


def set_rubric_for_problem(problem_idx: str, rubric: Union[List[Dict[str, Any]], str], notify: bool = True, save_to_file: bool = True):
    """
    Set the rubric for a specific problem_idx and save to persistent storage.
    
    Args:
        problem_idx: The problem index to set rubric for
        rubric: The rubric (list of items or string) to set
        notify: Whether to notify registered callbacks
        save_to_file: Whether to save to persistent file (default: True)
    """
    # Get current rubrics
    rubrics = get_current_rubrics()
    
    # Update the specific problem's rubric
    rubrics[problem_idx] = rubric
    
    # Save to file for persistence across processes
    if save_to_file:
        rubric_file = _get_rubric_file_path()
        try:
            with open(rubric_file, 'w', encoding='utf-8') as f:
                json.dump(rubrics, f, indent=2, ensure_ascii=False)
        except (IOError, Exception) as e:
            print(f"Warning: Could not save rubric to file: {e}")
    
    if notify:
        for callback in _rubric_update_callbacks:
            try:
                callback(problem_idx, rubric)
            except Exception as e:
                print(f"Error in rubric update callback: {e}")


def delete_rubric_for_problem(problem_idx: str, notify: bool = True, save_to_file: bool = True):
    """
    Delete the rubric for a specific problem_idx.
    
    Args:
        problem_idx: The problem index to delete rubric for
        notify: Whether to notify registered callbacks
        save_to_file: Whether to save to persistent file (default: True)
    """
    # Get current rubrics
    rubrics = get_current_rubrics()
    
    # Remove the specific problem's rubric if it exists
    if problem_idx in rubrics:
        del rubrics[problem_idx]
        
        # Save to file for persistence across processes
        if save_to_file:
            rubric_file = _get_rubric_file_path()
            try:
                with open(rubric_file, 'w', encoding='utf-8') as f:
                    json.dump(rubrics, f, indent=2, ensure_ascii=False)
            except (IOError, Exception) as e:
                print(f"Warning: Could not save rubric to file: {e}")
        
        if notify:
            for callback in _rubric_update_callbacks:
                try:
                    callback(problem_idx, None)  # None indicates deletion
                except Exception as e:
                    print(f"Error in rubric update callback: {e}")


def register_rubric_update_callback(callback):
    """
    Register a callback function to be called when rubrics are updated.
    
    Args:
        callback: Function to call with (problem_idx, rubric) when rubrics change
    """
    _rubric_update_callbacks.append(callback)


class RubricGUI:
    """GUI application for managing rubrics."""
    
    def __init__(self, root=None):
        """
        Initialize the GUI application.
        
        Args:
            root: Optional tkinter root window. If None, creates a new one.
        """
        if root is None:
            self.root = tk.Tk()
            self._root_owner = True
        else:
            self.root = root
            self._root_owner = False
            
        self.root.title("Rubric Editor")
        self.root.geometry("1000x700")
        
        # Current state
        self.current_problem_idx = ""
        self.current_rubric = []
        self.rubric_mode = "structured"  # "structured" or "text"
        
        self._setup_ui()
        self._load_problem_list()
        
    def _setup_ui(self):
        """Set up the user interface components."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Title label
        title_label = ttk.Label(
            main_frame, 
            text="Rubric Editor - Edit Rubrics by Problem Index", 
            font=("Arial", 16, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 15))
        
        # Problem selection section
        problem_frame = ttk.LabelFrame(main_frame, text="Problem Selection", padding="5")
        problem_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        problem_frame.columnconfigure(1, weight=1)
        
        ttk.Label(problem_frame, text="Problem Index:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        
        self.problem_idx_var = tk.StringVar()
        self.problem_idx_combo = ttk.Combobox(problem_frame, textvariable=self.problem_idx_var, width=20)
        self.problem_idx_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        self.problem_idx_combo.bind('<<ComboboxSelected>>', self._on_problem_selected)
        
        ttk.Button(problem_frame, text="Load", command=self._load_rubric).grid(row=0, column=2, padx=(0, 5))
        ttk.Button(problem_frame, text="New", command=self._new_rubric).grid(row=0, column=3, padx=(0, 5))
        ttk.Button(problem_frame, text="Delete", command=self._delete_rubric).grid(row=0, column=4)
        
        # Mode selection
        mode_frame = ttk.Frame(problem_frame)
        mode_frame.grid(row=1, column=0, columnspan=5, sticky=(tk.W, tk.E), pady=(10, 0))
        
        ttk.Label(mode_frame, text="Rubric Mode:").pack(side=tk.LEFT, padx=(0, 10))
        
        self.mode_var = tk.StringVar(value="structured")
        ttk.Radiobutton(mode_frame, text="Structured Items", variable=self.mode_var, 
                       value="structured", command=self._on_mode_change).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Radiobutton(mode_frame, text="Plain Text", variable=self.mode_var, 
                       value="text", command=self._on_mode_change).pack(side=tk.LEFT)
        
        # Rubric editing section
        self.rubric_frame = ttk.LabelFrame(main_frame, text="Rubric Content", padding="5")
        self.rubric_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        self.rubric_frame.columnconfigure(0, weight=1)
        self.rubric_frame.rowconfigure(1, weight=1)
        
        # Create both structured and text interfaces
        self._setup_structured_ui()
        self._setup_text_ui()
        
        # Show structured mode by default
        self._show_mode("structured")
        
        # Action buttons
        action_frame = ttk.Frame(main_frame)
        action_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        ttk.Button(action_frame, text="Save Rubric", command=self._save_rubric).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(action_frame, text="Load from File", command=self._load_from_file).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(action_frame, text="Save to File", command=self._save_to_file).pack(side=tk.LEFT)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_label = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_label.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
    def _setup_structured_ui(self):
        """Set up the structured rubric editing interface."""
        # Rubric items list
        self.structured_frame = ttk.Frame(self.rubric_frame)
        
        list_frame = ttk.Frame(self.structured_frame)
        list_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        list_frame.columnconfigure(0, weight=1)
        
        self.rubric_listbox = tk.Listbox(list_frame, height=6)
        self.rubric_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E))
        self.rubric_listbox.bind('<<ListboxSelect>>', self._on_rubric_item_selected)
        
        list_scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.rubric_listbox.yview)
        list_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.rubric_listbox.configure(yscrollcommand=list_scroll.set)
        
        # Buttons for list management
        button_frame = ttk.Frame(self.structured_frame)
        button_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        ttk.Button(button_frame, text="Add Item", command=self._add_rubric_item).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Remove Item", command=self._remove_rubric_item).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Move Up", command=self._move_item_up).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Move Down", command=self._move_item_down).pack(side=tk.LEFT)
        
        # Rubric item editing section
        edit_frame = ttk.LabelFrame(self.structured_frame, text="Edit Rubric Item", padding="5")
        edit_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        edit_frame.columnconfigure(1, weight=1)
        edit_frame.rowconfigure(2, weight=1)
        
        ttk.Label(edit_frame, text="Title:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        self.title_var = tk.StringVar()
        self.title_entry = ttk.Entry(edit_frame, textvariable=self.title_var)
        self.title_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=(0, 5))
        
        ttk.Label(edit_frame, text="Max Points:").grid(row=1, column=0, sticky=tk.W, pady=(0, 5))
        self.max_points_var = tk.StringVar()
        self.max_points_entry = ttk.Entry(edit_frame, textvariable=self.max_points_var, width=10)
        self.max_points_entry.grid(row=1, column=1, sticky=tk.W, pady=(0, 5))
        
        ttk.Label(edit_frame, text="Content:").grid(row=2, column=0, sticky=(tk.W, tk.N), pady=(0, 5))
        self.content_text = scrolledtext.ScrolledText(edit_frame, height=8, wrap=tk.WORD)
        self.content_text.grid(row=2, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 5))
        
        ttk.Button(edit_frame, text="Update Item", command=self._update_rubric_item).grid(row=3, column=1, sticky=tk.E, pady=(5, 0))
        
        # Configure grid weights for structured frame
        self.structured_frame.columnconfigure(0, weight=1)
        self.structured_frame.rowconfigure(2, weight=1)
        
    def _setup_text_ui(self):
        """Set up the plain text rubric editing interface."""
        self.text_frame = ttk.Frame(self.rubric_frame)
        
        # Instructions
        instruction_label = ttk.Label(
            self.text_frame, 
            text="Enter the complete rubric as plain text. This will be used directly in the evaluation prompt.",
            foreground="blue"
        )
        instruction_label.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Text editor
        self.rubric_text = scrolledtext.ScrolledText(
            self.text_frame, 
            height=20, 
            wrap=tk.WORD,
            font=("Consolas", 11)
        )
        self.rubric_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights for text frame
        self.text_frame.columnconfigure(0, weight=1)
        self.text_frame.rowconfigure(1, weight=1)
        
    def _show_mode(self, mode):
        """Show the appropriate UI mode."""
        # Hide both frames first
        self.structured_frame.grid_remove()
        self.text_frame.grid_remove()
        
        if mode == "structured":
            self.structured_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            self.rubric_mode = "structured"
        else:  # text mode
            self.text_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            self.rubric_mode = "text"
            
    def _on_mode_change(self):
        """Handle mode change between structured and text."""
        new_mode = self.mode_var.get()
        
        # Convert current rubric to new mode if needed
        if self.current_rubric:
            if new_mode == "text" and isinstance(self.current_rubric, list):
                # Convert structured to text
                text_rubric = self._convert_structured_to_text(self.current_rubric)
                self.current_rubric = text_rubric
            elif new_mode == "structured" and isinstance(self.current_rubric, str):
                # Convert text to structured (basic conversion)
                structured_rubric = self._convert_text_to_structured(self.current_rubric)
                self.current_rubric = structured_rubric
        
        self._show_mode(new_mode)
        self._load_current_rubric()
        
    def _convert_structured_to_text(self, structured_rubric):
        """Convert structured rubric to text format."""
        text_parts = []
        for i, item in enumerate(structured_rubric, 1):
            title = item.get('title', f'Criterion {i}')
            max_points = item.get('max_points', 0)
            content = item.get('grading_scheme_desc', '')
            text_parts.append(f"{i}. {title} ({max_points} points): {content}")
        return "\n\n".join(text_parts)
        
    def _convert_text_to_structured(self, text_rubric):
        """Convert text rubric to structured format (basic parsing)."""
        # This is a simple conversion - users can refine in structured mode
        return [
            {
                "title": "Converted from Text",
                "max_points": 1,
                "grading_scheme_desc": text_rubric.strip()
            }
        ]
        
    def _load_problem_list(self):
        """Load the list of available problem indices."""
        rubrics = get_current_rubrics()
        problem_indices = list(rubrics.keys())
        self.problem_idx_combo['values'] = problem_indices
        
    def _on_problem_selected(self, event=None):
        """Handle problem selection from combobox."""
        self._load_rubric()
        
    def _load_rubric(self):
        """Load the rubric for the selected problem index."""
        problem_idx = self.problem_idx_var.get().strip()
        if not problem_idx:
            messagebox.showwarning("Warning", "Please enter a problem index!")
            return
            
        self.current_problem_idx = problem_idx
        rubric = get_rubric_for_problem(problem_idx)
        
        if rubric is None:
            # Default to structured mode for new rubrics
            self.current_rubric = []
            self.mode_var.set("structured")
            self.status_var.set(f"No rubric found for problem {problem_idx}. Creating new.")
        else:
            self.current_rubric = rubric.copy() if isinstance(rubric, list) else rubric
            # Set mode based on rubric type
            if isinstance(rubric, str):
                self.mode_var.set("text")
            else:
                self.mode_var.set("structured")
            self.status_var.set(f"Loaded rubric for problem {problem_idx}")
            
        self._show_mode(self.mode_var.get())
        self._load_current_rubric()
        
    def _load_current_rubric(self):
        """Load the current rubric into the appropriate UI."""
        if self.rubric_mode == "structured":
            self._update_rubric_list()
        else:  # text mode
            self.rubric_text.delete(1.0, tk.END)
            if isinstance(self.current_rubric, str):
                self.rubric_text.insert(1.0, self.current_rubric)
            elif isinstance(self.current_rubric, list):
                # Convert structured to text for display
                text_rubric = self._convert_structured_to_text(self.current_rubric)
                self.rubric_text.insert(1.0, text_rubric)
        
    def _new_rubric(self):
        """Create a new rubric for the entered problem index."""
        problem_idx = self.problem_idx_var.get().strip()
        if not problem_idx:
            messagebox.showwarning("Warning", "Please enter a problem index!")
            return
            
        self.current_problem_idx = problem_idx
        self.current_rubric = []
        self._update_rubric_list()
        self.status_var.set(f"Created new rubric for problem {problem_idx}")
        
    def _delete_rubric(self):
        """Delete the rubric for the current problem index."""
        if not self.current_problem_idx:
            messagebox.showwarning("Warning", "No problem selected!")
            return
            
        if messagebox.askyesno("Confirm Delete", f"Delete rubric for problem {self.current_problem_idx}?"):
            delete_rubric_for_problem(self.current_problem_idx)
            self.current_rubric = []
            self._update_rubric_list()
            self._load_problem_list()
            self.status_var.set(f"Deleted rubric for problem {self.current_problem_idx}")
            
    def _update_rubric_list(self):
        """Update the rubric items listbox."""
        self.rubric_listbox.delete(0, tk.END)
        for i, item in enumerate(self.current_rubric):
            title = item.get('title', 'Untitled')
            max_points = item.get('max_points', 0)
            self.rubric_listbox.insert(tk.END, f"{i+1}. {title} ({max_points} pts)")
            
    def _on_rubric_item_selected(self, event=None):
        """Handle selection of a rubric item."""
        selection = self.rubric_listbox.curselection()
        if selection:
            idx = selection[0]
            item = self.current_rubric[idx]
            
            self.title_var.set(item.get('title', ''))
            self.max_points_var.set(str(item.get('max_points', 0)))
            self.content_text.delete(1.0, tk.END)
            self.content_text.insert(1.0, item.get('grading_scheme_desc', ''))
            
    def _add_rubric_item(self):
        """Add a new rubric item."""
        new_item = {
            'title': 'New Rubric Item',
            'max_points': 1,
            'grading_scheme_desc': 'Enter grading criteria here...'
        }
        self.current_rubric.append(new_item)
        self._update_rubric_list()
        # Select the new item
        self.rubric_listbox.selection_set(len(self.current_rubric) - 1)
        self._on_rubric_item_selected()
        
    def _remove_rubric_item(self):
        """Remove the selected rubric item."""
        selection = self.rubric_listbox.curselection()
        if selection:
            idx = selection[0]
            del self.current_rubric[idx]
            self._update_rubric_list()
            
    def _move_item_up(self):
        """Move the selected item up in the list."""
        selection = self.rubric_listbox.curselection()
        if selection and selection[0] > 0:
            idx = selection[0]
            self.current_rubric[idx], self.current_rubric[idx-1] = self.current_rubric[idx-1], self.current_rubric[idx]
            self._update_rubric_list()
            self.rubric_listbox.selection_set(idx-1)
            
    def _move_item_down(self):
        """Move the selected item down in the list."""
        selection = self.rubric_listbox.curselection()
        if selection and selection[0] < len(self.current_rubric) - 1:
            idx = selection[0]
            self.current_rubric[idx], self.current_rubric[idx+1] = self.current_rubric[idx+1], self.current_rubric[idx]
            self._update_rubric_list()
            self.rubric_listbox.selection_set(idx+1)
            
    def _update_rubric_item(self):
        """Update the selected rubric item with the current form values."""
        selection = self.rubric_listbox.curselection()
        if selection:
            idx = selection[0]
            try:
                max_points = float(self.max_points_var.get())
            except ValueError:
                messagebox.showerror("Error", "Max points must be a number!")
                return
                
            self.current_rubric[idx] = {
                'title': self.title_var.get(),
                'max_points': max_points,
                'grading_scheme_desc': self.content_text.get(1.0, tk.END).strip()
            }
            self._update_rubric_list()
            self.rubric_listbox.selection_set(idx)  # Keep selection
            
    def _save_rubric(self):
        """Save the current rubric."""
        if not self.current_problem_idx:
            messagebox.showwarning("Warning", "No problem selected!")
            return
        
        # Get the current rubric from the appropriate UI
        if self.rubric_mode == "text":
            # Get text from text editor
            rubric_to_save = self.rubric_text.get(1.0, tk.END).strip()
            if not rubric_to_save:
                messagebox.showwarning("Warning", "Rubric text is empty!")
                return
        else:
            # Use structured rubric
            rubric_to_save = self.current_rubric
            if not rubric_to_save:
                messagebox.showwarning("Warning", "No rubric items to save!")
                return
            
        set_rubric_for_problem(self.current_problem_idx, rubric_to_save)
        self._load_problem_list()
        self.status_var.set(f"Saved rubric for problem {self.current_problem_idx}")
        messagebox.showinfo("Success", f"Rubric for problem {self.current_problem_idx} has been saved!")
        
    def _load_from_file(self):
        """Load rubric from a JSON file."""
        from tkinter import filedialog
        filename = filedialog.askopenfilename(
            title="Load Rubric from File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                if isinstance(data, list):
                    # Direct rubric list
                    self.current_rubric = data
                elif isinstance(data, dict) and self.current_problem_idx in data:
                    # Problem-indexed rubric dict
                    self.current_rubric = data[self.current_problem_idx]
                else:
                    messagebox.showerror("Error", "Invalid rubric file format!")
                    return
                    
                self._update_rubric_list()
                self.status_var.set(f"Loaded rubric from {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not load file: {e}")
                
    def _save_to_file(self):
        """Save current rubric to a JSON file."""
        from tkinter import filedialog
        filename = filedialog.asksaveasfilename(
            title="Save Rubric to File",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(self.current_rubric, f, indent=2, ensure_ascii=False)
                self.status_var.set(f"Saved rubric to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save file: {e}")
                
    def run(self):
        """Run the GUI application."""
        if self._root_owner:
            self.root.mainloop()


def main():
    """Main function to run the rubric GUI."""
    app = RubricGUI()
    app.run()


if __name__ == "__main__":
    main()
