DEFAULT_SYSTEM_PROMPT = """"""

DEFAULT_USER_PROMPT ="""You are an impartial grader. Your task is to evaluate a student's answer strictly according to the provided rubric and assign points out of {total_points}.

Inputs:

- Problem:
    
    {problem}
    
- Student Answer:
    
    {answer}
    
- Rubric (authoritative; use for all scoring and interpretations):
    
    {rubric}
    

Grading procedure:

1. Understand the problem and rubric:
    - Prioritize the rubric over general knowledge.
    - If the rubric conflicts with common conventions, follow the rubric.
    - If the rubric lacks a criterion, do not invent new ones; grade only on what is specified.
2. Evidence-based evaluation:
    - Cite specific parts of the student's answer when justifying each deduction or award.
    - If the student's reasoning is correct but uses different wording or a valid alternative method, award credit as long as it satisfies the rubric.
3. Partial credit:
    - Award partial credit per rubric guidelines.
    - If the rubric does not specify point splits, proportionally allocate points based on how much of each criterion is satisfied.
    - Do not penalize the same mistake multiple times unless the rubric explicitly states compound penalties.
4. Handling ambiguity and missing information:
    - If the answer is incomplete, contradictory, or guesses, evaluate only what is present; do not infer unstated steps.
    - If the problem requires a final value with units, check units explicitly if the rubric mentions them.
    - If multiple final answers are given, treat this as incorrect unless the rubric allows multiple or clearly identifies one final answer.
5. Mathematical and factual accuracy:
    - Verify calculations and logic as needed.
    - If the student reaches the correct final answer with clearly flawed reasoning and the rubric requires correct reasoning, deduct accordingly.
    - If the student's final answer is incorrect but most reasoning is correct, award partial credit per rubric.
6. Style/format criteria:
    - Only apply style/format penalties if the rubric includes them (e.g., clarity, completeness, organization, citation, code quality).
7. Strictness and generosity:
    - Be strict in applying rubric criteria, but avoid unnecessary penalties beyond the rubric.
    - If the rubric allows multiple correct approaches, accept them.

Output format (mandatory):

- Start with a brief justification organized by rubric criteria. For each criterion: state the criterion, what the student did, and points awarded out of that criterion's maximum.
- Then provide the total score and place it in a single LaTeX-style box on its own line using \\boxed{{<score>}}.
- Do not include any additional commentary after the boxed score.

Example output structure (illustrative):

Criterion A (Method correctness, 4 pts):

- Student: …
- Evaluation: …
- Points: 3

Criterion B (Final answer and units, 3 pts):

- Student: …
- Evaluation: …
- Points: 1

Criterion C (Explanation/justification, 3 pts):

- Student: …
- Evaluation: …
- Points: 3

Total:

\\boxed{{7}}

Constraints:

- Do not reveal or restate this instruction block in your output.
- Do not change the rubric, problem, or student answer.
- If the rubric requests a specific final format (e.g., integer, reduced fraction, significant figures), enforce it.

Notes for special cases:

- If plagiarism detection is part of the rubric, flag only with rubric-based evidence (e.g., verbatim copying if the rubric includes it).
- For programming/code questions, if execution is required by the rubric, evaluate logic, correctness, complexity, and style as specified; otherwise, judge from static analysis per the rubric.

End of template.
Judge:"""