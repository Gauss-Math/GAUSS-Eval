import sys
from pathlib import Path
import json
# Add project root to path to import from src
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.prompt_utils import get_current_prompt, get_system_prompt, get_user_prompt, get_rubric_for_problem


def parse_prompt(data_item: dict) -> dict:
    """
    Parse the prompts from the data item.
    Uses the persistent prompts from the GUI storage file.
    Checks for edited rubrics by problem_idx first, falls back to original rubric.
    
    Args:
        data_item: Dictionary containing data item information (may contain total_points or other formatting variables)
    
    Returns:
        dict: Dictionary with 'system_prompt' and 'user_prompt' keys
    """
    # Get the current prompts from persistent storage (reads from .prompt_config.json)
    system_prompt, user_prompt = get_current_prompt()
    total_points = data_item.get('max_points_judge_1', 7)
    
    # Check for edited rubric by problem_idx first
    problem_idx = data_item.get("problem_idx", "")
    edited_rubric = get_rubric_for_problem(str(problem_idx)) if problem_idx else None
    
    if edited_rubric is not None:
        # Use edited rubric - handle both string and structured formats
        if isinstance(edited_rubric, str):
            # Plain text rubric - use directly
            rubric = "\n" + edited_rubric
        else:
            # Structured rubric - format as before
            rubric_list = []
            for item in edited_rubric:
                title = item.get("title", "")
                max_points = item.get("max_points", 0)
                content = item.get("grading_scheme_desc", "")
                
                formatted_string = f"Rubric title: {title}, Max points: {max_points}, Rubric content: {content}"
                rubric_list.append(formatted_string)
            
            # join into a string with order 1., 2., ...
            rubric = "\n" + "\n".join([f"{i+1}. {item}" for i, item in enumerate(rubric_list)])
    else:
        # Fall back to original rubric from data item
        grading_detail_raw = data_item.get("grading_details_judge_1", "")
        rubric_list = []
        for item in grading_detail_raw:
            title = item.get("title", "")
            max_points = item.get("max_points", 0)
            content = item.get("grading_scheme_desc", "")
            
            formatted_string = f"Rubric title: {title}, Max points: {max_points}, Rubric content: {content}"
            rubric_list.append(formatted_string)
        
        # join into a string with order 1., 2., ...
        rubric = "\n" + "\n".join([f"{i+1}. {item}" for i, item in enumerate(rubric_list)])
    problem = data_item.get("problem", "")
    # stuent's answer
    answer = data_item.get("answer", "")
    
    return {
        'system_prompt': system_prompt,
        'user_prompt': user_prompt.format(total_points=total_points, rubric=rubric, problem=problem, answer=answer)
    }


def parse_system_prompt(data_item: dict) -> str:
    """
    Parse only the system prompt from the data item.
    
    Args:
        data_item: Dictionary containing data item information
    
    Returns:
        str: The system prompt string
    """
    return get_system_prompt()


def parse_user_prompt(data_item: dict) -> str:
    """
    Parse only the user prompt from the data item.
    
    Args:
        data_item: Dictionary containing data item information
    
    Returns:
        str: The user prompt string
    """
    return get_user_prompt()


def get_dataset_imo2025(data_path: str = "data/IMO2025/data.json") -> list:
    """
    Get the dataset from the data path.
    """
    with open(data_path, 'r') as f:
        return json.load(f)