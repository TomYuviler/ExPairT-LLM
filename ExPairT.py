from utils import *
import numpy as np
import signal
import os
import argparse
from tqdm import tqdm
from prompts import template_inputs, template_membership, template_equivalence
from openai import OpenAI
import anthropic
import google.generativeai as genai



os.environ['OPENAI_API_KEY'] = "YOUR_OPENAI_API_KEY"
os.environ['DEEPSEEK_API_KEY'] = "YOUR_DEEPSEEK_API_KEY"
os.environ['ANTHROPIC_API_KEY'] = "YOUR_ANTHROPIC_API_KEY"
os.environ['GEMINI_API_KEY'] = "YOUR_GEMINI_API_KEY"
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
TIMEOUT = 15 # seconds


def llm_equivalence_queries(llm_model, prompt, program_1, program_2):
    """
    Query any chat-style LLM to judge whether *program_1* and *program_2*
    are functionally equivalent for the task described by *prompt*.

    The helper fills in the ``template_equivalence`` prompt and returns the
    model’s raw response (e.g. ``"Program 1"``, ``"Program 2"``, or
    ``"NO_DIFF"``).

    Parameters
    ----------
    llm_model : str
        Identifier for the model to call (OpenAI, vLLM, etc.).
    prompt : str
        Natural-language task description.
    program_1, program_2 : str
        Candidate solution sources.

    Returns
    -------
    str
        The LLM’s reply.
    """
    equivalence_prompt = template_equivalence.format(
        docstring=prompt, program_1=program_1, program_2=program_2
    )
    if llm_model in ["gpt-4o", "gpt-4o-mini", "o1-mini", "o3-mini", "o4-mini", "deepseek-reasoner"]:
        client = OpenAI()
        if llm_model == "deepseek-reasoner":
            client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
        completion = client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": equivalence_prompt}],
        )
        return completion.choices[0].message.content
    elif llm_model in ["claude-3-7-sonnet-20250219"]:
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        message = client.messages.create(
            model=llm_model,
            max_tokens=10000,
            messages=[
                {"role": "user", "content": equivalence_prompt}
            ]
        )
        return message.content[0].text
    elif llm_model in ["gemini-2.5-flash-preview-04-17"]:
        model = genai.GenerativeModel(
            model_name=llm_model)
        message = model.generate_content(
            equivalence_prompt,
            generation_config=genai.GenerationConfig(
            )
        )
        return message.text


def llm_membership_queries(llm_model, prompt, diff_examples):
    """
    Ask an LLM which program wins on the provided *diff_examples* for the task
    *prompt* (“membership” query in ExPairT-LLM).
    If the composed prompt exceeds ~20 k tokens, fall back to ``"Program 2"``.

    Parameters
    ----------
    llm_model : str
        Chat-style model name (OpenAI or local).
    prompt : str
        Task description / docstring.
    diff_examples : str
        Formatted IO examples showing differing outputs.

    Returns
    -------
    str
        Model reply such as ``"Program 1"`` or ``"Program 2"``.
    """
    membership_prompt = template_membership.format(docstring=prompt,
                                                   examples=diff_examples)
    if len(membership_prompt) > 20000:
        return "Program 2"
    if llm_model in ["gpt-4o", "gpt-4o-mini", "o1-mini", "o3-mini", "o4-mini", "deepseek-reasoner"]:
        client = OpenAI()
        if llm_model == "deepseek-reasoner":
            client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
        completion = client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": membership_prompt}],
        )
        return completion.choices[0].message.content

    elif llm_model in ["claude-3-7-sonnet-20250219"]:
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        message = client.messages.create(
            model=llm_model,
            max_tokens=10000,
            messages=[
                {"role": "user", "content": membership_prompt}
            ]
        )
        return message.content[0].text

    elif llm_model in ["gemini-2.5-flash-preview-04-17"]:
        model = genai.GenerativeModel(
            model_name=llm_model)
        message = model.generate_content(
            membership_prompt,
            generation_config=genai.GenerationConfig(
            )
        )
        return message.text


def select_cluster(llm_model, prompt, generated_examples, all_candidate_outputs):
    """
    Run a Copeland-style round-robin to pick the best behavioural cluster.
    Each unseen pair of candidate output vectors is fed to the LLM “membership”
    oracle; wins and losses accumulate until a Condorcet winner emerges or the
    maximum sweep score breaks the tie.

    Parameters
    ----------
    llm_model : str
        Identifier of the chat model used for membership queries.
    prompt : str
        Task description shown to the oracle.
    generated_examples : list[dict]
        Inputs that produced the *all_candidate_outputs*.
    all_candidate_outputs : list[list[Any]]
        Parallel list of outputs—one list per candidate program.

    Returns
    -------
    int
        Index of the selected cluster representative in
        *all_candidate_outputs*.
    """
    already_judged_pairs = []
    num_wins_per_program = [0] * len(all_candidate_outputs)
    num_loses_per_program = [0] * len(all_candidate_outputs)
    used_indices = set()
    num_iterations = 0
    for _ in range(len(all_candidate_outputs)):
        num_iterations += 1
        i = get_max_index(num_wins_per_program, used_indices)
        if num_wins_per_program[i] + (len(all_candidate_outputs) - len(used_indices) - 1) <= max(num_wins_per_program):
            best_idx_program = np.argmax(num_wins_per_program)
            return best_idx_program
        used_indices.add(i)
        for j in range(len(all_candidate_outputs)):
            if {i, j} in already_judged_pairs or i == j:
                continue
            else:
                already_judged_pairs.append({i, j})
            signal.signal(signal.SIGALRM, timeout_handler)
            try:
                signal.alarm(TIMEOUT)
                signal.alarm(0)
            except TimeoutException:
                print("Timeout occurred while calling function, continuing with next iteration...")
                signal.alarm(0)
            except Exception as e:
                print(f"Error occurred while calling function: {e}")
                signal.alarm(0)
            finally:
                signal.alarm(0)

            outputs_current = all_candidate_outputs[i]
            outputs_new = all_candidate_outputs[j]
            diff_examples = format_differing_comparisons(generated_examples, outputs_new, outputs_current)
            judge_reply = llm_membership_queries(llm_model, prompt, diff_examples)
            judge_reply = ''.join(judge_reply.split())
            if judge_reply == "Program1":
                num_wins_per_program[j] += 1
                num_loses_per_program[i] += 1
            elif judge_reply == "Program2":
                num_wins_per_program[i] += 1
                num_loses_per_program[j] += 1
        if num_wins_per_program[i] == len(all_candidate_outputs) - 1:
            return i
    # find argmax in num_wins_per_program
    best_idx_program = np.argmax(num_wins_per_program)
    return best_idx_program


def generate_inputs(llm_model, prompt):
    """
    Query an LLM for new test inputs.
    Fills `template_inputs` with *prompt* and returns the model’s raw reply
    (typically a numbered list of example dictionaries).

    Parameters
    ----------
    llm_model : str
        Chat model identifier.
    prompt : str
        Task description / docstring.

    Returns
    -------
    str
        LLM-generated input examples.
    """
    prompt_for_generating_inputs = template_inputs.format(docstring=prompt)
    if llm_model in ["gpt-4o", "gpt-4o-mini", "o1-mini", "o3-mini", "o4-mini", "deepseek-reasoner"]:
        client = OpenAI()
        if llm_model == "deepseek-reasoner":
            client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
        completion = client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt_for_generating_inputs}],
        )
        return completion.choices[0].message.content
    elif llm_model in ["claude-3-7-sonnet-20250219"]:
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        message = client.messages.create(
            model=llm_model,
            max_tokens=10000,
            messages=[
                {"role": "user", "content": prompt_for_generating_inputs}
            ]
        )
        return message.content[0].text

    elif llm_model in ["gemini-2.5-flash-preview-04-17"]:
        model = genai.GenerativeModel(
            model_name=llm_model)
        message = model.generate_content(
            prompt_for_generating_inputs,
            generation_config=genai.GenerationConfig(
            )
        )
        return message.text
def generate_candidates(llm_model, num_candidates, prompt):
    """
    Produce *num_candidates* solution snippets from the chosen chat model.
    Sends the same *prompt* repeatedly, collects fenced Python blocks from each
    reply, and returns them as a list of strings.

    Parameters
    ----------
    llm_model : str
        Model identifier (OpenAI or local).
    num_candidates : int
        How many completions to request.
    prompt : str
        The code-generation prompt.

    Returns
    -------
    list[str]
        Extracted code blocks (one per candidate).
    """
    candidates_list = []
    for candidate_idx in range(num_candidates):
        if llm_model in ["gpt-4o", "gpt-4o-mini", "deepseek-reasoner"]:
            client = OpenAI()
            if llm_model == "deepseek-reasoner":
                client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
            completion = client.chat.completions.create(
                model=llm_model,
                temperature=0.8,
                top_p=0.95,
                messages=[{"role": "user", "content": prompt}],
            )
            generated_code = completion.choices[0].message.content
            candidates_list.append(extract_python_code_blocks(generated_code))
        if llm_model in ["o1-mini", "o3-mini", "o4-mini"]:
            client = OpenAI()
            completion = client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
            )
            generated_code = completion.choices[0].message.content
            candidates_list.append(extract_python_code_blocks(generated_code))
        elif llm_model in ["claude-3-7-sonnet-20250219"]:
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            message = client.messages.create(
                model=llm_model,
                max_tokens=10000,
                temperature=0.8,
                top_p=0.95,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            generated_code = message.content[0].text
            candidates_list.append(extract_python_code_blocks(generated_code))

        elif llm_model in ["gemini-2.5-flash-preview-04-17"]:
            model = genai.GenerativeModel(
                model_name=llm_model)
            message = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.8,
                    top_p=0.95
                )
            )
            generated_code = message.text
            candidates_list.append(extract_python_code_blocks(generated_code))
    return candidates_list



def search_diff_input(llm_model, prompt, programs_list_str, programs_list):
    """
    Ask the LLM for inputs that separate two candidate clusters.
    Iterates over cluster members, requests an “equivalence” explanation, and
    returns the first input set whose outputs diverge; otherwise ``None``.

    Parameters
    ----------
    llm_model : str
        Chat model for equivalence queries.
    prompt : str
        Task description.
    programs_list_str : list[str]
        Source codes of the current cluster.
    programs_list : list[callable]
        Compiled callables matching *programs_list_str*.

    Returns
    -------
    list[dict] | None
        Discriminating inputs, or ``None`` if clusters appear equivalent.
    """
    cluster_size = len(programs_list_str)
    assert cluster_size > 1
    for equivalence_idx in range(1, cluster_size):
        diff_response = llm_equivalence_queries(
            llm_model, prompt,
            programs_list_str[0], programs_list_str[equivalence_idx]
        )
        if diff_response.strip() == "NO_DIFF":
            continue
        possible_diff_inputs = extract_input_examples("Example 1: " + diff_response)
        signal.signal(signal.SIGALRM, timeout_handler)
        try:
            signal.alarm(TIMEOUT)
            output_program_1 = apply_function_on_inputs(programs_list[0], possible_diff_inputs)
            output_program_2 = apply_function_on_inputs(programs_list[equivalence_idx], possible_diff_inputs)
            signal.alarm(0)
        except TimeoutException:
            print("Timeout occurred while calling function, continuing with next iteration...")
            signal.alarm(0)
            continue
        except Exception as e:
            print(f"Error occurred while calling function: {e}")
            signal.alarm(0)
            continue
        finally:
            signal.alarm(0)
        if not compare_lists(output_program_1, output_program_2):  # found diff input
            return possible_diff_inputs
    return None


def split_to_clusters(programs_list_str, programs_list, diff_inputs):
    """
    Group candidate programs by identical outputs on *diff_inputs*.
    Each unique output vector forms a new cluster; metadata for every cluster
    is returned so the caller can continue the tournament.

    Parameters
    ----------
    programs_list_str : list[str]
        Source code of the current candidates.
    programs_list : list[callable]
        Corresponding compiled callables.
    diff_inputs : list[dict]
        Inputs used to evaluate behavioural equivalence.

    Returns
    -------
    all_candidate_programs : list[callable]
        One representative per cluster.
    all_candidate_outputs : list[list]
        The output vectors of those representatives.
    clusters_dict_programs_str : dict[int, list[str]]
        Cluster index → list of program sources.
    clusters_dict_programs : dict[int, list[callable]]
        Cluster index → list of callables.
    """
    all_candidate_programs = []
    all_candidate_outputs = []
    size_of_clusters_list = []
    clusters_dict_programs_str = {}
    clusters_dict_programs = {}
    for idx in range(len(programs_list)):
        program = programs_list[idx]
        program_str = programs_list_str[idx]
        signal.signal(signal.SIGALRM, timeout_handler)
        try:
            signal.alarm(TIMEOUT)
            outputs_new = apply_function_on_inputs(program, diff_inputs)
            signal.alarm(0)
            is_already_exist, exist_idx = list_in_lists(outputs_new, all_candidate_outputs)
            if is_already_exist:
                size_of_clusters_list[exist_idx] += 1
                clusters_dict_programs_str[exist_idx].append(program_str)
                clusters_dict_programs[exist_idx].append(program)
                continue
            else:
                all_candidate_programs.append(program)
                all_candidate_outputs.append(outputs_new)
                size_of_clusters_list.append(1)
                num_items_so_far = len(clusters_dict_programs_str)
                clusters_dict_programs_str[num_items_so_far] = [program_str]
                clusters_dict_programs[num_items_so_far] = [program]
        except TimeoutException:
            print("Timeout occurred while calling function, continuing with next iteration...")
            continue
        except Exception as e:
            print(f"Error occurred while calling function: {e}")
            continue
        finally:
            signal.alarm(0)  # Ensure the alarm is canceled in all cases
    return all_candidate_programs, all_candidate_outputs, clusters_dict_programs_str, clusters_dict_programs


def ExPairT_LLM(llm_model, prompt, candidates_list, generated_examples, function_name):
    """
    End-to-end ExPairT loop: cluster candidates by behaviour, ask the LLM oracle
    to pick a winning cluster, refine with counter-examples, and repeat until a
    single program (or stable cluster) remains.

    Parameters
    ----------
    llm_model : str
        Chat model used for membership/equivalence queries.
    prompt : str
        Task description shown to the oracle.
    candidates_list : list[str]
        Generated solution sources.
    generated_examples : list[dict]
        Seed inputs used for the first behavioural hash.
    function_name : str
        Entry-point to call inside each candidate.

    Returns
    -------
    tuple[callable | None, str]
        The selected program object (or ``None`` if none ran) and its source.
    """
    all_candidate_programs = []
    all_candidate_outputs = []
    size_of_equivalence_class_list = []
    clusters_dict_programs = {}
    clusters_dict_programs_str = {}

    for candidate in candidates_list:
        signal.signal(signal.SIGALRM, timeout_handler)
        try:
            signal.alarm(TIMEOUT)
            candidate_function = string_to_function(candidate, function_name)
            signal.alarm(0)
            if candidate_function is None:
                continue
        except TimeoutException:
            signal.alarm(0)
            continue
        except Exception:
            signal.alarm(0)
            continue

        signal.signal(signal.SIGALRM, timeout_handler)
        try:
            signal.alarm(TIMEOUT)
            outputs_new = apply_function_on_inputs(candidate_function, generated_examples)
            signal.alarm(0)
            is_already_exist, exist_idx = list_in_lists(outputs_new, all_candidate_outputs)
            if is_already_exist:
                size_of_equivalence_class_list[exist_idx] += 1
                clusters_dict_programs_str[exist_idx].append(candidate)
                clusters_dict_programs[exist_idx].append(candidate_function)
                continue
            else:
                all_candidate_programs.append(candidate_function)
                all_candidate_outputs.append(outputs_new)
                size_of_equivalence_class_list.append(1)
                num_items_so_far = len(clusters_dict_programs_str)
                clusters_dict_programs_str[num_items_so_far] = [candidate]
                clusters_dict_programs[num_items_so_far] = [candidate_function]
        except TimeoutException:
            print("Timeout occurred while calling function, continuing with next iteration...")
            signal.alarm(0)
            continue
        except Exception as e:
            print(f"Error occurred while calling function: {e}")
            signal.alarm(0)
            continue
        finally:
            signal.alarm(0)  # Ensure the alarm is canceled in all cases

    if len(all_candidate_outputs) == 0:
        return None, "None"
    diff_input = generated_examples
    for iter_idx in range(100):
        if len(all_candidate_outputs) == 1:
            selected_cluster = 0
        else:
            selected_cluster = select_cluster(llm_model, prompt, diff_input, all_candidate_outputs)
        programs_list_str = clusters_dict_programs_str[selected_cluster]
        programs_list = clusters_dict_programs[selected_cluster]
        if len(programs_list) == 1:
            return programs_list[0], programs_list_str[0]
        else:
            diff_input = search_diff_input(llm_model, prompt, programs_list_str, programs_list)
        if diff_input is None:
            return programs_list[0], programs_list_str[0]
        else:
            (all_candidate_programs,
             all_candidate_outputs,
             clusters_dict_programs_str,
             clusters_dict_programs) = split_to_clusters(programs_list_str,
                                                         programs_list,
                                                         diff_input)

# --------------------------------------------------------------------------- #
#  ExPairT-LLM CLI                                                           #
#  Run this file directly to benchmark the selector on the chosen dataset.   #
#  Flags allow you to pick the dataset, number of candidates, and the three  #
#  LLM roles (generation / input search / oracle).                           #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="mbpp_sanitized",
                        help="Benchmark to evaluate.")
    parser.add_argument("--num_candidate", type=int, default=8,
                        help="Number of programs to generate per task.")
    parser.add_argument("--llm_for_generation", default="o1-mini",
                        help="Model used to generate code.")
    parser.add_argument("--llm_for_inputs", default="o1-mini",
                        help="Model used to suggest inputs.")
    parser.add_argument("--llm_for_oracle", default="o1-mini",
                        help="Model acting as the membership/equivalence oracle.")
    args = parser.parse_args()

    ds, num_tasks = load_tasks(args.dataset_name)
    num_tasks_for_eval = 0
    num_first_correct = 0
    num_ExPairT_LLM_correct = 0

    for task_idx in tqdm(range(num_tasks), desc="Processing tasks"):
        # if task_idx < 7:
        #     continue
        task = parse_task(task_idx, ds, args.dataset_name)
        signature, func_name = parse_func_name_and_signature(
            task_idx, ds, args.dataset_name
        )
        prompt = create_prompt(args.dataset_name, task, signature)

        candidates = generate_candidates(
            args.llm_for_generation, args.num_candidate, prompt
        )

        tests = extract_tests(task_idx, ds, args.dataset_name)
        if check_all_correct_incorrect(candidates, tests, func_name, args.dataset_name):
            continue  # skip trivial all-right / all-wrong tasks

        num_tasks_for_eval += 1
        gen_examples = extract_input_examples(generate_inputs(args.llm_for_inputs, prompt))

        _, chosen_src = ExPairT_LLM(
            args.llm_for_oracle, prompt, candidates, gen_examples, func_name
        )

        is_first, is_best = evaluate(
            args.dataset_name, candidates[0], chosen_src, tests, func_name
        )
        if is_first:
            num_first_correct += 1
        if is_best:
            num_ExPairT_LLM_correct += 1

        print(
            f"Tasks: {num_tasks_for_eval}, "
            f"baseline pass@1: {round(100 * num_first_correct / num_tasks_for_eval, 2)}, "
            f"ExPairT-LLM pass@1: {round(100 * num_ExPairT_LLM_correct / num_tasks_for_eval, 2)}"
        )
