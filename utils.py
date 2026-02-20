from eval_tests import *
import re
from typing import Tuple
import types
import textwrap
import pprint
import copy
import signal
import ast
from datasets import load_dataset
TIMEOUT = 15 # seconds


class TimeoutException(Exception):
    """
    Exception raised when a protected call exceeds the allotted time budget.

    Notes
    -----
    The alarm-based timeout guard arms a Unix `SIGALRM` before the protected
    region and raises this exception in the signal handler.  Catch it at the
    call-site to handle timeouts gracefully without killing the whole process.
    """
    pass


def timeout_handler(signum, frame):
    """
    Signal handler that converts ``SIGALRM`` into :class:`TimeoutException`.

    Parameters
    ----------
    signum : int
        The signal number (should always be ``signal.SIGALRM`` in this context).
    frame : types.FrameType
        The current stack frame at the moment the signal was received.

    Raises
    ------
    TimeoutException
        Always raised to abort the current execution once the time limit is
        reached.
    """
    raise TimeoutException("Function execution timed out.")


def lcb_strip_class_and_name(snippet: str) -> Tuple[str, str]:
    """
    Strip the *LiveCodeBench* wrapper and return a plain function header.

    LiveCodeBench tasks wrap the target method in::

        class Solution:
            def func(self, ...):

    This helper removes the wrapper and the leading ``self`` so downstream
    tools that expect a top-level function can reuse the signature.

    Parameters
    ----------
    snippet : str
        Raw Python text that begins with the ``class Solution`` block.

    Returns
    -------
    signature : str
        Re-written header such as ``\"def func(arg1, arg2):\"``.
    func_name : str
        The extracted function name (``\"func\"``).

    Raises
    ------
    ValueError
        If the snippet does not start with the expected class wrapper or
        contains no method definition.
    """
    # 1. normalise indentation and drop blank lines
    lines = [ln for ln in textwrap.dedent(snippet).splitlines() if ln.strip()]

    if not lines or not lines[0].lstrip().startswith("class"):
        raise ValueError("Expected the first non-blank line to be 'class Solution:'")

    # 2. find the first `def …` line after the class header
    func_line = next((ln for ln in lines[1:] if ln.lstrip().startswith("def")), None)
    if func_line is None:
        raise ValueError("No function definition found after the class header.")

    # 3. grab the function name and everything from the opening “(”
    m = re.match(r'\s*def\s+([A-Za-z_]\w*)\s*\((.*)', func_line)
    if not m:
        raise ValueError("Couldn’t parse a Python function signature.")
    func_name, remainder = m.groups()

    # 4. drop a leading ‘self’ (with optional comma & whitespace)
    remainder = re.sub(r'^\s*self\s*,?\s*', '', remainder)

    # 5. build the stripped signature
    signature = f"def {func_name}({remainder}"

    return signature, func_name


def parse_func_name_and_signature(task_idx, dataset, dataset_name):
    """
    Retrieve the canonical function signature and entry-point name for a task
    from one of the supported code-generation benchmarks.

    The four datasets handled here encode the target function differently:

    * **livecodebench** – wrapped in ``class Solution``; unwrapped via
      :func:`lcb_strip_class_and_name`.
    * **APPS** – always ``def solution(stdin: str) -> str:``.
    * **humaneval** – function already top-level; signature left empty.
    * **mbpp_sanitized** – name read from the JSONL metadata; signature empty.

    Parameters
    ----------
    task_idx : int
        Index of the task within *dataset*.
    dataset : datasets.arrow_dataset.Dataset
        Loaded Hugging Face split containing the tasks.
    dataset_name : str
        One of ``{'livecodebench', 'APPS', 'humaneval', 'mbpp_sanitized'}``.

    Returns
    -------
    tuple(str, str)
        ``(function_signature, function_name)``.  The *signature* may be an
        empty string when the benchmark already exposes a standalone function.
    """
    if dataset_name == 'livecodebench':
        function_signature, function_name = lcb_strip_class_and_name(
            dataset[task_idx]["starter_code"]
        )
    elif dataset_name == 'APPS':
        function_signature = "def solution(stdin: str) -> str:"
        function_name = "solution"
    elif dataset_name == 'humaneval':
        function_signature = ""
        function_name = dataset[task_idx]["entry_point"]
    elif dataset_name == 'mbpp_sanitized':
        function_signature = ""
        _, function_name = mbpp_read_prompt_entry(
            "mbpp_sanitized_for_code_generation.jsonl", task_idx
        )
    return function_signature, function_name.strip()

def create_prompt(dataset_name, task, function_signature):
    """
    Assemble the text given to the code-generation LLM.
    For *livecodebench* and *APPS* the task is wrapped in triple quotes and
    followed by the plain ``def …`` line; for *humaneval* and *mbpp* the task
    is returned verbatim.

    Parameters
    ----------
    dataset_name : str
    task : str
    function_signature : str

    Returns
    -------
    str
    """
    if dataset_name in ['livecodebench', 'APPS']:
        prefix = "'''"
        suffix = "'''"
        prompt = prefix + "\n" + task + "\n" + suffix + "\n" + function_signature
    elif dataset_name in ['humaneval', 'mbpp_sanitized']:
        prompt = task
    return prompt


def extract_python_code_blocks(markdown_text, as_list=False):
    """
    Pull every ```python ... ``` fence out of a Markdown string.
    Returns either the list of blocks or a single string with them joined by a
    blank line.

    Parameters
    ----------
    markdown_text : str
        Source text containing Markdown.
    as_list : bool, default False
        Set True to get a list instead of one concatenated string.

    Returns
    -------
    list[str] | str
    """
    pattern = re.compile(
        r'```python\s*\n(.*?)\n```',  # match fenced python blocks
        re.DOTALL | re.IGNORECASE
    )
    code_blocks = pattern.findall(markdown_text)
    return code_blocks if as_list else '\n\n'.join(code_blocks)

def string_to_function(func_str, main_func_name):
    """
    Converts a string containing one or more function definitions into a callable function object.

    Parameters:
    - func_str (str): The string containing Python function definitions.
    - main_func_name (str): The name of the main function to extract.

    Returns:
    - function: The extracted main function object, or None if an error occurred.
    """
    func_str = func_str.strip()

    # Create a single namespace for both globals and locals
    namespace = {}
    try:
        # Optionally override 'input' if you need to prevent input calls
        namespace['input'] = lambda: '0'
        # Execute the function string in this namespace
        exec(func_str, namespace)
    except Exception as e:
        print(f"Error executing function string: {e}")
        return None

    # Extract the main function from the namespace
    main_func = namespace.get(main_func_name, None)
    if main_func is None:
        print(f"Main function '{main_func_name}' not found in the function string.")
        return None
    return main_func

def get_max_index(lst, used_indices):
    """
    Index of the largest element in *lst* that is **not** already in
    *used_indices*.

    Parameters
    ----------
    lst : list[float]
        Scores to search.
    used_indices : set[int] | list[int]
        Indices that must be skipped.

    Returns
    -------
    int
        Position of the best unused score, or ``-1`` if none remain.
    """
    max_value = float('-inf')
    max_index = -1
    for i, value in enumerate(lst):
        if i not in used_indices and value > max_value:
            max_value = value
            max_index = i
    return max_index


def format_differing_comparisons(generated_examples, outputs1, outputs2):
    """
    Formats comparison results of two programs based on generated inputs,
    including only the cases where the outputs differ, and numbers them sequentially.

    Args:
        generated_examples (list of dict): List of input examples, each as a dictionary.
        outputs1 (list): List of outputs from Program 1.
        outputs2 (list): List of outputs from Program 2.

    Returns:
        str: A formatted string comparing the outputs of both programs for each differing input.
    """
    # Check if all lists have the same length
    if not (len(generated_examples) == len(outputs1) == len(outputs2)):
        raise ValueError("All input lists must have the same length.")

    formatted_output = []
    example_counter = 1  # Initialize counter for differing examples

    for example, output1, output2 in zip(generated_examples, outputs1, outputs2):
        # Check if outputs differ
        if output1 != output2:
            # Extract the input parameters from the example dictionary
            input_params = ', '.join([f"{key}={value}" for key, value in example.items()])

            # Convert outputs to string representations
            output1_str = pprint.pformat(output1)
            output2_str = pprint.pformat(output2)

            # Format the comparison string with sequential numbering
            comparison_str = (
                f"Example {example_counter}:\n"
                f"For Input: {input_params}\n"
                f"Output of Program 1:\n{output1_str}\n"
                f"Output of Program 2:\n{output2_str}\n"
            )

            formatted_output.append(comparison_str)
            example_counter += 1  # Increment counter for the next differing example

    # If no differences are found, inform the user
    if not formatted_output:
        return "No differences found between the outputs of Program 1 and Program 2."

    # Join all comparison strings with two newlines for readability
    return "\n".join(formatted_output)

def compare_elements(item1, item2):
    """
    Deep-equality helper for ExPairT-LLM.
    Returns *True* when two items are of the same type and equal, handling
    nested lists and exceptions specially.

    Parameters
    ----------
    item1, item2 : Any
        Objects to compare.

    Returns
    -------
    bool
    """
    # Check if types are the same
    if type(item1) != type(item2):
        return False
    # Handle lists (nested)
    if isinstance(item1, list):
        return compare_lists(item1, item2)
    # Handle exceptions
    if isinstance(item1, BaseException):
        return item1.args == item2.args
    # Handle other data types
    return item1 == item2


def compare_lists(list1, list2):
    """
    Deep comparison for two lists, recursing into nested structures.
    Lists must be the same length and each corresponding element must satisfy
    :func:`compare_elements`.

    Parameters
    ----------
    list1, list2 : list
        Lists to compare.

    Returns
    -------
    bool
    """
    if len(list1) != len(list2):
        return False
    return all(compare_elements(a, b) for a, b in zip(list1, list2))


def list_in_lists(target_list, lists_of_lists):
    """
    Locate *target_list* inside *lists_of_lists* using deep comparison.
    Scans sequentially with :func:`compare_lists` and returns the first match.

    Parameters
    ----------
    target_list : list
        List to search for.
    lists_of_lists : list[list]
        Collection of candidate lists.

    Returns
    -------
    tuple[bool, int]
        ``(True, idx)`` if found, else ``(False, -1)``.
    """
    for index, lst in enumerate(lists_of_lists):
        if compare_lists(target_list, lst):
            return True, index
    return False, -1


def check_all_correct_incorrect(candidates_list, tests, function_name, dataset_name):
    """
    Detect “trivial” tasks: return **True** when every candidate is *either*
    correct *or* incorrect on the public tests, **False** if the set is mixed.

    Parameters
    ----------
    candidates_list : list[str]
        Generated solution sources.
    tests : str | list
        Public test bundle for the given dataset.
    function_name : str
        Entry-point to execute in each candidate.
    dataset_name : str
        One of ``{'livecodebench', 'APPS', 'humaneval', 'mbpp_sanitized'}``.

    Returns
    -------
    bool
    """
    is_one_correct = False
    is_one_incorrect = False
    for candidate in candidates_list:
        signal.signal(signal.SIGALRM, timeout_handler)
        try:
            signal.alarm(TIMEOUT)
            if dataset_name == 'livecodebench':
                if lcb_passes_functional_tests(candidate, tests, function_name):
                    is_one_correct = True
                else:
                    is_one_incorrect = True
            elif dataset_name == 'APPS':
                if apps_passes_tests(candidate, tests, function_name):
                    is_one_correct = True
                else:
                    is_one_incorrect = True
            elif dataset_name == 'humaneval':
                if humaneval_passes_tests(candidate, tests, function_name):
                    is_one_correct = True
                else:
                    is_one_incorrect = True
            elif dataset_name == 'mbpp_sanitized':
                if mbpp_passes_tests(candidate, tests, function_name):
                    is_one_correct = True
                else:
                    is_one_incorrect = True
            signal.alarm(0)
        except TimeoutException:
            signal.alarm(0)
            continue
        except Exception:
            signal.alarm(0)
            continue

    if is_one_correct and is_one_incorrect:
        return False
    else:
        return True

def apply_function_on_inputs(func, inputs):
    """
    Applies a function to a list of input dictionaries and returns the outputs.

    Parameters:
        func (callable): The function to apply. It should accept keyword arguments.
        inputs (list): A list of dictionaries, each representing a set of inputs to the function.

    Returns:
        list: A list containing the outputs from applying the function to each input.
    """
    outputs = []
    for idx, input_dict in enumerate(inputs, 1):
        try:
            # Create a deep copy of the input dictionary to prevent modification
            input_copy = copy.deepcopy(input_dict)
            # Unpack the copied input dictionary as keyword arguments
            result = func(**input_copy)
            outputs.append(result)
        except TimeoutException as e:
            outputs.append(e)
            print(f"An error occurred with input {idx}: {input_dict}\nError: {e}")
            # Re-raise the TimeoutException to be handled at a higher level
            raise
        except Exception as e:
            # Handle other exceptions
            outputs.append(e)
            print(f"An error occurred with input {idx}: {input_dict}\nError: {e}")
    return outputs

def extract_tests(task_idx, dataset, dataset_name):
    """
    Fetch the public-test bundle for a single task from the chosen benchmark.
    Simply selects the appropriate column in the Hugging Face dataset split.

    Parameters
    ----------
    task_idx : int
        Row index of the task.
    dataset : datasets.arrow_dataset.Dataset
        Loaded split containing all tasks.
    dataset_name : str
        One of ``{'livecodebench', 'APPS', 'humaneval', 'mbpp_sanitized'}``.

    Returns
    -------
    Any
        The dataset-specific test blob (JSON string or list of asserts).
    """
    if dataset_name == 'livecodebench':
        tests = dataset[task_idx]["public_test_cases"]
    if dataset_name == 'APPS':
        tests = dataset[task_idx]["input_output"]
    if dataset_name == 'humaneval':
        tests = dataset[task_idx]["test"]
    if dataset_name == 'mbpp_sanitized':
        tests = dataset[task_idx]["test_list"]
    return tests


def extract_input_examples(llm_response):
    """
    Extracts input examples from the LLM's response.

    Parameters:
        llm_response (str): The response string from the LLM containing input examples.

    Returns:
        list: A list of dictionaries, each representing an input example.
    """
    examples = []
    # Split the response into lines
    lines = llm_response.strip().split('\n')

    # Regular expression to match 'Example X:'
    example_regex = re.compile(r'^Example\s+\d+:\s*(.*)$')

    # Regular expression to match key=value pairs
    # This regex matches a key followed by '=', and then a value that may contain any characters,
    # stopping at a comma followed by another key= or the end of the line.
    param_regex = re.compile(r'(\w+)\s*=\s*(.+?)(?=(?:,\s*\w+\s*=|$))')

    for line in lines:
        # Check if the line matches the example pattern
        match = example_regex.match(line.strip())
        if match:
            params_str = match.group(1).strip()
            # Initialize an empty dictionary to store parameters
            input_dict = {}
            # Find all key-value pairs in the parameters string
            param_matches = param_regex.findall(params_str)
            for key, value in param_matches:
                key = key.strip()
                value = value.strip()
                # Safely evaluate the value to handle complex data types
                try:
                    # Handle cases where the value might contain incomplete parentheses
                    # or extra whitespace that could cause literal_eval to fail
                    value = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    # If evaluation fails, keep the value as a string
                    value = value
                input_dict[key] = value
            examples.append(input_dict)
    return examples

def keep_lcb(example):
    """
    Return *True* for LiveCodeBench tasks that have functional tests **and**
    non-empty starter code; otherwise *False*.

    Parameters
    ----------
    example : dict
        A single LiveCodeBench record.

    Returns
    -------
    bool
    """
    has_func = "functional" in example["public_test_cases"]
    has_code = example["starter_code"] is not None
    return has_func and has_code


def load_tasks(dataset_name):
    """
    Download the *test* split of the requested benchmark and return it with its
    length.

    Parameters
    ----------
    dataset_name : str
        One of ``{'livecodebench', 'APPS', 'humaneval', 'mbpp_sanitized'}``.

    Returns
    -------
    tuple[datasets.arrow_dataset.Dataset, int]
        The filtered Hugging Face dataset and its number of tasks.
    """
    if dataset_name == 'livecodebench':
        ds = load_dataset(
            "livecodebench/code_generation_lite",
            split="test",
            version_tag="release_latest",
        ).filter(keep_lcb)
    elif dataset_name == 'APPS':
        ds = load_dataset("codeparrot/apps", split="test")
    elif dataset_name == 'humaneval':
        ds = load_dataset("openai/openai_humaneval", split="test")
    elif dataset_name == 'mbpp_sanitized':
        ds = load_dataset("Muennighoff/mbpp", "sanitized", split="test")
    return ds, len(ds)


def parse_task(task_idx, dataset, dataset_name):
    """
    Return the natural-language problem statement for a single task, picking
    the correct field for each benchmark.

    Parameters
    ----------
    task_idx : int
        Row index of the task.
    dataset : datasets.arrow_dataset.Dataset
        Dataset split loaded via Hugging Face.
    dataset_name : str
        One of ``{'livecodebench', 'APPS', 'humaneval', 'mbpp_sanitized'}``.

    Returns
    -------
    str
        Task description shown to the LLM.
    """
    if dataset_name == 'livecodebench':
        task = dataset[task_idx]["question_content"]
    elif dataset_name == 'APPS':
        task = dataset[task_idx]["question"]
    elif dataset_name == 'humaneval':
        task = dataset[task_idx]["prompt"]
    elif dataset_name == 'mbpp_sanitized':
        task, _ = mbpp_read_prompt_entry(
            "mbpp_sanitized_for_code_generation.jsonl", task_idx
        )
    return task


def evaluate(dataset_name, fisrt_candidate, expairt_candidate, tests, function_name):
    """
    Score the baseline (first generated) solution versus the ExPairT-selected
    candidate on the public tests of a given benchmark.

    Parameters
    ----------
    dataset_name : str
        Benchmark identifier.
    fisrt_candidate : str
        Source code of the first-generated program (baseline).
    expairt_candidate : str
        Source code chosen by ExPairT-LLM.
    tests : Any
        Dataset-specific public test bundle.
    function_name : str
        Entry-point to execute in each candidate.

    Returns
    -------
    tuple[bool, bool]
        ``(is_first_correct, is_expairt_correct)``.
    """
    if dataset_name == 'livecodebench':
        is_first_correct = lcb_passes_functional_tests(
            fisrt_candidate, tests, function_name
        )
        is_expairt_correct = lcb_passes_functional_tests(
            expairt_candidate, tests, function_name
        )
    elif dataset_name == 'APPS':
        is_first_correct = apps_passes_tests(
            fisrt_candidate, tests, function_name
        )
        is_expairt_correct = apps_passes_tests(
            expairt_candidate, tests, function_name
        )
    elif dataset_name == 'humaneval':
        is_first_correct = humaneval_passes_tests(
            fisrt_candidate, tests, function_name
        )
        is_expairt_correct = humaneval_passes_tests(
            expairt_candidate, tests, function_name
        )
    elif dataset_name == 'mbpp_sanitized':
        is_first_correct = mbpp_passes_tests(
            fisrt_candidate, tests, function_name
        )
        is_expairt_correct = mbpp_passes_tests(
            expairt_candidate, tests, function_name
        )
    return is_first_correct, is_expairt_correct

def mbpp_read_prompt_entry(file_path: str, line_index: int) -> Tuple[str, str]:
    """
    Return (prompt, entry_point) for the `line_index`-th record in a JSONL file.

    Parameters
    ----------
    file_path   : path to the *.jsonl* file
    line_index  : 0-based index of the line/record you want

    Raises
    ------
    IndexError   – if the file has fewer than `line_index + 1` lines
    KeyError     – if the JSON object lacks 'prompt' or 'entry_point'
    """
    with open(file_path, encoding="utf-8") as f:
        for i, raw in enumerate(f):
            if i == line_index:
                record = json.loads(raw)
                return record["prompt"], record["entry_point"]

    raise IndexError("line_index out of range")
