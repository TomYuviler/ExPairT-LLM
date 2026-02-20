template_inputs = """Generate 5 input examples for a Python function based on its docstring. The function may have a variable number of input parameters. The inputs should be valid according to the function's description, covering the entire valid input space as defined.

**Please ensure that the inputs comprehensively cover the valid input space defined by the function's description.**

Please provide **only** the inputs in the following format and **do not include any additional text**:

Example 1: parameter_name1=input_value1, parameter_name2=input_value2, ...
Example 2: parameter_name1=input_value1, parameter_name2=input_value2, ...
Example 3: parameter_name1=input_value1, parameter_name2=input_value2, ...
Example 4: parameter_name1=input_value1, parameter_name2=input_value2, ...
Example 5: parameter_name1=input_value1, parameter_name2=input_value2, ...

Here is the docstring:

{docstring}

Please provide the 5 input examples following the format above without any additional explanations or text."""


template_membership = """You are provided with the partial docstring of a Python function and multiple input-output examples from two different programs (Program 1 and Program 2). Your task is to analyze these examples and determine which program's outputs better align with the function's intended behavior as described in the docstring.

**Please provide only the classification as a single term: "Program 1" or "Program 2". Do not include any additional text or explanations.**

Here is the information provided:

- **Docstring:** 

{docstring}

- **Input-Output Examples:**

{examples}
"""

template_equivalence = """You are given the following partial docstring of a Python function:

{docstring}

And two Python programs implementing this function:

**Program 1:**

{program_1}

**Program 2:**

{program_2}

**Task:** Find an input example such that when this input is provided to both programs, they produce different outputs. 

**Important:** The input should be **valid according to the function's description**.

**Output Format:** If you find such an input, please return it in the format:

parameter_name1=input_value1, parameter_name2=input_value2, ...

If you cannot find such an input, please return the term "NO_DIFF".

**Important:** Provide **only** the input in the specified format or the term "NO_DIFF", without any additional explanations or output.

"""