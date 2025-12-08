"""
Utility functions for prompt and rubric management.
This replaces the GUI-based prompt and rubric functions with simple file-based storage.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

from src.default_prompt import DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT


def _get_prompt_file_path():
    """Get the path to the persistent prompt storage file."""
    project_root = Path(__file__).parent.parent
    return project_root / ".prompt_config.json"


def _get_rubric_file_path():
    """Get the path to the persistent rubric storage file."""
    project_root = Path(__file__).parent.parent
    return project_root / ".rubric_config.json"


def get_current_prompt(global_config_path: Optional[str] = None) -> Tuple[str, str]:
    """
    Get the current prompt from global config or persistent storage.
    Returns a tuple of (system_prompt, user_prompt).
    Falls back to defaults if file doesn't exist or has errors.
    
    Args:
        global_config_path: Optional path to global config file to check for prompts
    
    Returns:
        tuple: (system_prompt: str, user_prompt: str)
    """
    # First, try to read from global config if provided
    if global_config_path:
        try:
            with open(global_config_path, 'r', encoding='utf-8') as f:
                global_config = json.load(f)
                
            # Check if prompts are defined in global config
            if 'system_prompt' in global_config and 'user_prompt' in global_config:
                return global_config['system_prompt'], global_config['user_prompt']
            elif 'system_prompt' in global_config:
                # Only system prompt in global config, get user prompt from other sources
                system_prompt = global_config['system_prompt']
                _, user_prompt = _get_prompt_from_file_or_default()
                return system_prompt, user_prompt
            elif 'user_prompt' in global_config:
                # Only user prompt in global config, get system prompt from other sources
                user_prompt = global_config['user_prompt']
                system_prompt, _ = _get_prompt_from_file_or_default()
                return system_prompt, user_prompt
        except (json.JSONDecodeError, IOError, FileNotFoundError, Exception) as e:
            print(f"Warning: Could not read global config file for prompts: {e}")
    
    # Fallback to persistent storage or defaults
    return _get_prompt_from_file_or_default()


def _get_prompt_from_file_or_default() -> Tuple[str, str]:
    """
    Helper function to get prompts from persistent storage file or defaults.
    
    Returns:
        tuple: (system_prompt: str, user_prompt: str)
    """
    prompt_file = _get_prompt_file_path()
    
    # Try to read from file first
    if prompt_file.exists():
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                system_prompt = data.get('system_prompt', DEFAULT_SYSTEM_PROMPT)
                user_prompt = data.get('user_prompt', DEFAULT_USER_PROMPT)
                return system_prompt, user_prompt
        except (json.JSONDecodeError, IOError, Exception) as e:
            print(f"Warning: Could not read prompt file: {e}")
    
    # Fallback to defaults
    return DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT


def get_system_prompt(global_config_path: Optional[str] = None) -> str:
    """
    Get the current system prompt.
    
    Args:
        global_config_path: Optional path to global config file to check for prompts
    
    Returns:
        str: The system prompt string
    """
    system_prompt, _ = get_current_prompt(global_config_path)
    return system_prompt


def get_user_prompt(global_config_path: Optional[str] = None) -> str:
    """
    Get the current user prompt.
    
    Args:
        global_config_path: Optional path to global config file to check for prompts
    
    Returns:
        str: The user prompt string
    """
    _, user_prompt = get_current_prompt(global_config_path)
    return user_prompt


def set_current_prompt(system_prompt: str, user_prompt: str):
    """
    Set the current prompt and save to persistent storage.
    
    Args:
        system_prompt: The system prompt string
        user_prompt: The user prompt string
    """
    prompt_file = _get_prompt_file_path()
    
    data = {
        'system_prompt': system_prompt,
        'user_prompt': user_prompt
    }
    
    try:
        with open(prompt_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except (IOError, Exception) as e:
        print(f"Warning: Could not save prompt file: {e}")


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
    return rubrics.get(problem_idx, None)


def set_rubric_for_problem(problem_idx: str, rubric: Union[List[Dict[str, Any]], str]):
    """
    Set the rubric for a specific problem_idx and save to persistent storage.
    
    Args:
        problem_idx: The problem index to set rubric for
        rubric: The rubric (list of items or string) for the problem
    """
    rubrics = get_current_rubrics()
    rubrics[problem_idx] = rubric
    
    rubric_file = _get_rubric_file_path()
    
    try:
        with open(rubric_file, 'w', encoding='utf-8') as f:
            json.dump(rubrics, f, indent=2, ensure_ascii=False)
    except (IOError, Exception) as e:
        print(f"Warning: Could not save rubric file: {e}")


