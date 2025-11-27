import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import json
import os
import sys
from pathlib import Path

# Ensure project root is in path for imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import default prompts
from src.default_prompt import DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT

# Global variables to store the current prompts (for in-process access)
CURRENT_SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT
CURRENT_USER_PROMPT = DEFAULT_USER_PROMPT

# Global variable to store prompt update callback (optional)
_prompt_update_callbacks = []


def _get_prompt_file_path():
    """
    Get the path to the persistent prompt storage file.
    
    Returns:
        Path: Path to the prompt storage file
    """
    # Store in project root as .prompt_config.json
    project_root = Path(__file__).parent.parent
    return project_root / ".prompt_config.json"


def get_current_prompt():
    """
    Get the current prompt from persistent storage.
    Returns a tuple of (system_prompt, user_prompt).
    Falls back to defaults if file doesn't exist or has errors.
    
    Returns:
        tuple: (system_prompt: str, user_prompt: str)
    """
    prompt_file = _get_prompt_file_path()
    
    # Try to read from file first
    if prompt_file.exists():
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Read system_prompt and user_prompt
                system_prompt = data.get('system_prompt', '')
                user_prompt = data.get('user_prompt', '')
                
                if system_prompt or user_prompt:
                    global CURRENT_SYSTEM_PROMPT, CURRENT_USER_PROMPT
                    CURRENT_SYSTEM_PROMPT = system_prompt
                    CURRENT_USER_PROMPT = user_prompt
                    return (system_prompt, user_prompt)
        except (json.JSONDecodeError, IOError, Exception) as e:
            print(f"Warning: Could not read prompt file: {e}")
    
    # Fallback to defaults
    return (DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT)


def get_system_prompt():
    """
    Get the current system prompt.
    
    Returns:
        str: The system prompt string
    """
    system_prompt, _ = get_current_prompt()
    return system_prompt


def get_user_prompt():
    """
    Get the current user prompt.
    
    Returns:
        str: The user prompt string
    """
    _, user_prompt = get_current_prompt()
    return user_prompt


def set_current_prompt(system_prompt: str = None, user_prompt: str = None, notify: bool = True, save_to_file: bool = True):
    """
    Set the current prompts globally and save to persistent storage.
    
    Args:
        system_prompt: The system prompt string to set (optional)
        user_prompt: The user prompt string to set (optional)
        notify: Whether to notify registered callbacks
        save_to_file: Whether to save to persistent file (default: True)
    """
    global CURRENT_SYSTEM_PROMPT, CURRENT_USER_PROMPT
    
    if system_prompt is not None:
        CURRENT_SYSTEM_PROMPT = system_prompt
    if user_prompt is not None:
        CURRENT_USER_PROMPT = user_prompt
    
    # Save to file for persistence across processes
    if save_to_file:
        prompt_file = _get_prompt_file_path()
        try:
            data = {
                'system_prompt': CURRENT_SYSTEM_PROMPT,
                'user_prompt': CURRENT_USER_PROMPT
            }
            with open(prompt_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except (IOError, Exception) as e:
            print(f"Warning: Could not save prompt to file: {e}")
    
    if notify:
        for callback in _prompt_update_callbacks:
            try:
                callback(CURRENT_SYSTEM_PROMPT, CURRENT_USER_PROMPT)
            except Exception as e:
                print(f"Error in prompt update callback: {e}")


def register_prompt_update_callback(callback):
    """
    Register a callback function to be called when the prompt is updated.
    
    Args:
        callback: A function that takes the new prompt as an argument
    """
    _prompt_update_callbacks.append(callback)


class PromptGUI:
    """GUI application for managing prompts."""
    
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
            
        self.root.title("Prompt Ablation Tool")
        self.root.geometry("900x800")
        
        # Load initial prompts from persistent storage
        self.current_system_prompt, self.current_user_prompt = get_current_prompt()
        
        self._setup_ui()
        self._load_prompts()
        
    def _setup_ui(self):
        """Set up the user interface components."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
        main_frame.rowconfigure(5, weight=1)
        
        # Title label
        title_label = ttk.Label(
            main_frame, 
            text="Prompt Editor - System & User Prompts", 
            font=("Arial", 16, "bold")
        )
        title_label.grid(row=0, column=0, pady=(0, 15))
        
        # System prompt section
        system_frame = ttk.LabelFrame(main_frame, text="System Prompt", padding="5")
        system_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        system_frame.columnconfigure(0, weight=1)
        system_frame.rowconfigure(1, weight=1)
        
        system_label = ttk.Label(system_frame, text="Enter system prompt:")
        system_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        self.system_prompt_text = scrolledtext.ScrolledText(
            system_frame,
            wrap=tk.WORD,
            width=80,
            height=10,
            font=("Consolas", 11)
        )
        self.system_prompt_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # User prompt section
        user_frame = ttk.LabelFrame(main_frame, text="User Prompt", padding="5")
        user_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        user_frame.columnconfigure(0, weight=1)
        user_frame.rowconfigure(1, weight=1)
        
        user_label = ttk.Label(user_frame, text="Enter user prompt:")
        user_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        self.user_prompt_text = scrolledtext.ScrolledText(
            user_frame,
            wrap=tk.WORD,
            width=80,
            height=10,
            font=("Consolas", 11)
        )
        self.user_prompt_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, pady=10)
        
        # Save button
        save_btn = ttk.Button(
            button_frame,
            text="Save Prompts",
            command=self._save_prompts
        )
        save_btn.pack(side=tk.LEFT, padx=5)
        
        # Load from file button
        load_btn = ttk.Button(
            button_frame,
            text="Load from File",
            command=self._load_from_file
        )
        load_btn.pack(side=tk.LEFT, padx=5)
        
        # Save to file button
        save_file_btn = ttk.Button(
            button_frame,
            text="Save to File",
            command=self._save_to_file
        )
        save_file_btn.pack(side=tk.LEFT, padx=5)
        
        # Reset button
        reset_btn = ttk.Button(
            button_frame,
            text="Reset",
            command=self._reset_prompts
        )
        reset_btn.pack(side=tk.LEFT, padx=5)
        
        # Run One Example button
        run_one_btn = ttk.Button(
            button_frame,
            text="Run One Example",
            command=self._open_run_one_example
        )
        run_one_btn.pack(side=tk.LEFT, padx=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(
            main_frame,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        status_bar.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Current prompts display frame
        display_frame = ttk.LabelFrame(main_frame, text="Current Saved Prompts", padding="5")
        display_frame.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        display_frame.columnconfigure(0, weight=1)
        
        self.current_system_display = ttk.Label(
            display_frame,
            text="",
            wraplength=750,
            foreground="blue"
        )
        self.current_system_display.grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        self.current_user_display = ttk.Label(
            display_frame,
            text="",
            wraplength=750,
            foreground="green"
        )
        self.current_user_display.grid(row=1, column=0, sticky=tk.W)
        
    def _load_prompts(self):
        """Load the current prompts into the text widgets."""
        self.system_prompt_text.delete(1.0, tk.END)
        self.system_prompt_text.insert(1.0, self.current_system_prompt)
        
        self.user_prompt_text.delete(1.0, tk.END)
        self.user_prompt_text.insert(1.0, self.current_user_prompt)
        
        self._update_status_display()
        
    def _save_prompts(self):
        """Save the prompts from the text widgets to global variables."""
        system_prompt = self.system_prompt_text.get(1.0, tk.END).strip()
        user_prompt = self.user_prompt_text.get(1.0, tk.END).strip()
        
        # if not system_prompt:
        #     messagebox.showwarning("Warning", "System prompt cannot be empty!")
        #     return
            
        set_current_prompt(system_prompt=system_prompt, user_prompt=user_prompt, save_to_file=True)
        self.current_system_prompt = system_prompt
        self.current_user_prompt = user_prompt
        self._update_status_display()
        self.status_var.set(f"Prompts saved at {self._get_timestamp()}")
        messagebox.showinfo("Success", "Prompts have been saved and are now available globally and persistently!")
        
    def _reset_prompts(self):
        """Reset the prompts to default values."""
        self.system_prompt_text.delete(1.0, tk.END)
        self.system_prompt_text.insert(1.0, DEFAULT_SYSTEM_PROMPT)
        
        self.user_prompt_text.delete(1.0, tk.END)
        self.user_prompt_text.insert(1.0, DEFAULT_USER_PROMPT)
        
        set_current_prompt(system_prompt=DEFAULT_SYSTEM_PROMPT, user_prompt=DEFAULT_USER_PROMPT, save_to_file=True)
        self.current_system_prompt = DEFAULT_SYSTEM_PROMPT
        self.current_user_prompt = DEFAULT_USER_PROMPT
        self._update_status_display()
        self.status_var.set("Prompts reset to defaults")
        
    def _load_from_file(self):
        """Load prompts from a JSON file (like sample_params.json)."""
        from tkinter import filedialog
        file_path = filedialog.askopenfilename(
            title="Select a JSON file",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                # Read system_prompt and user_prompt
                system_prompt = data.get('system_prompt', '')
                user_prompt = data.get('user_prompt', '')
                    
                if system_prompt or user_prompt:
                    self.system_prompt_text.delete(1.0, tk.END)
                    self.system_prompt_text.insert(1.0, system_prompt)
                    
                    self.user_prompt_text.delete(1.0, tk.END)
                    self.user_prompt_text.insert(1.0, user_prompt)
                    
                    self.status_var.set(f"Loaded prompts from {os.path.basename(file_path)}")
                else:
                    messagebox.showwarning("Warning", "No 'system_prompt' or 'user_prompt' field found in JSON file.")
        except json.JSONDecodeError:
            messagebox.showerror("Error", "Invalid JSON file!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {str(e)}")
            
    def _save_to_file(self):
        """Save the current prompts to a JSON file."""
        from tkinter import filedialog
        file_path = filedialog.asksaveasfilename(
            title="Save prompts to JSON file",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
            
        system_prompt = self.system_prompt_text.get(1.0, tk.END).strip()
        user_prompt = self.user_prompt_text.get(1.0, tk.END).strip()
        
        try:
            # Try to load existing file if it exists
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    data = json.load(f)
            else:
                data = {}
                
            data['system_prompt'] = system_prompt
            data['user_prompt'] = user_prompt
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=4)
                
            self.status_var.set(f"Saved to {os.path.basename(file_path)}")
            messagebox.showinfo("Success", f"Prompts saved to {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file: {str(e)}")
            
    def _update_status_display(self):
        """Update the current prompts display."""
        system_prompt, user_prompt = get_current_prompt()
        
        system_display = system_prompt[:80] + "..." if len(system_prompt) > 80 else system_prompt
        self.current_system_display.config(text=f"System: {system_display}")
        
        if user_prompt:
            user_display = user_prompt[:80] + "..." if len(user_prompt) > 80 else user_prompt
            self.current_user_display.config(text=f"User: {user_display}")
        else:
            self.current_user_display.config(text="User: (empty)")
        
    def _get_timestamp(self):
        """Get current timestamp as string."""
        from datetime import datetime
        return datetime.now().strftime("%H:%M:%S")
        
    def _open_run_one_example(self):
        """Open the Run One Example GUI in a new window."""
        try:
            from src.run_one_example import RunOneExampleGUI
            
            # Create new window for Run One Example
            run_one_window = tk.Toplevel(self.root)
            run_one_window.title("Run One Example")
            run_one_window.geometry("1200x900")
            
            # Create Run One Example GUI in the new window
            run_one_gui = RunOneExampleGUI(run_one_window)
            
            self.status_var.set("Opened Run One Example window")
            
        except ImportError as e:
            messagebox.showerror("Error", f"Could not import Run One Example module: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open Run One Example: {e}")
        
    def run(self):
        """Run the GUI application."""
        if self._root_owner:
            self.root.mainloop()
        else:
            # If root is provided externally, just update the display periodically
            self._update_status_display()
            self.root.after(1000, self._periodic_update)
            
    def _periodic_update(self):
        """Periodically update the display when running in external root."""
        self._update_status_display()
        self.root.after(1000, self._periodic_update)


def create_gui(root=None):
    """
    Create and return a PromptGUI instance.
    
    Args:
        root: Optional tkinter root window
        
    Returns:
        PromptGUI: The GUI instance
    """
    return PromptGUI(root)


def run_gui():
    """Run the GUI application in a standalone mode."""
    app = PromptGUI()
    app.run()


if __name__ == "__main__":
    # Run the GUI when executed directly
    run_gui()
    

