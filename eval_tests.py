import math
from typing import Any, List
import types
import json
import ast
import importlib.util
import tempfile
import os
import sys


def humaneval_passes_tests(code_str: str,
                           tests_source: str,   # the Python text that defines check(...)
                           func_name: str,
                           time_limit: float = 2.0) -> bool:
    """
    Load `code_str` into a throw-away module, fetch `func_name`, and run the
    HumanEval-style tests contained in `tests_source`.

    Returns
    -------
    True  – every assertion in the test script passed
    False – any assertion failed or a runtime error occurred
    """

    # ------------------------------------------------------------------
    # 1.  Persist the generated solution as a temp module
    # ------------------------------------------------------------------
    with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as f:
        f.write(code_str.encode())
        mod_path = f.name

    try:
        spec = importlib.util.spec_from_file_location("_candidate", mod_path)
        mod  = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod          # leave it in sys.modules so
        spec.loader.exec_module(mod)          # global look-ups keep working

        # Grab the entry-point the evaluator cares about
        fn = getattr(mod, func_name, None)
        if fn is None or not callable(fn):
            print(f"❌  function {func_name} not found")
            return False

        # ------------------------------------------------------------------
        # 2.  Prepare an execution sandbox for the HumanEval tests
        #     • start with *all* names from the candidate module
        #     • additionally bind the conventional name `candidate`
        # ------------------------------------------------------------------
        test_env = dict(mod.__dict__)   # helpers, constants, etc.
        test_env['candidate'] = fn      # what the tests expect to call

        try:
            exec(tests_source, test_env)          # defines check(...)
        except Exception as e:
            print("❌  test script failed to exec:", e)
            return False

        check_fn = test_env.get("check")
        if not isinstance(check_fn, types.FunctionType):
            print("❌  test script did not define a `check` function")
            return False

        # ------------------------------------------------------------------
        # 3.  Run the assertions inside check(candidate)
        # ------------------------------------------------------------------
        try:
            check_fn(fn)
        except AssertionError as e:
            print("❌  assertion failed:", e)
            return False
        except Exception as e:
            print("❌  runtime error while running tests:", e)
            return False

        return True                      # 🎉  everything passed!

    finally:
        os.remove(mod_path)


def _smart_cast(token: str) -> Any:
    """
    Try to turn `token` into int/float/bool using ast.literal_eval.
    If that yields *anything except* int/float/bool, keep the original string.
    """
    try:
        lit = ast.literal_eval(token)
        if isinstance(lit, (int, float, bool)):
            return lit
    except Exception:
        pass
    return token           # fall back to the raw string

def apps_passes_tests(code_str: str,
                      tests_json: str,
                      func_name: str,
                      time_limit: float = 2.0) -> bool:
    """
    Execute the APPS public tests on the candidate solution.
    The JSON blob supplies parallel ``inputs`` and ``outputs``; the helper
    imports *code_str*, runs *func_name* on each case, and returns **True** iff
    every result matches the expected string.

    Parameters
    ----------
    code_str : str
        Full Python source of the candidate solution.
    tests_json : str
        JSON with two arrays: ``inputs`` and ``outputs``.
    func_name : str
        Name of the entry-point function to test.
    time_limit : float, optional
        Reserved for future timeout handling (unused).

    Returns
    -------
    bool
        ``True`` if all tests pass, else ``False`` (with a diagnostic print).
    """
    # 0) Load the test blob ---------------------------------------------------
    try:
        blob = json.loads(tests_json)
    except json.JSONDecodeError as e:
        print("❌  APPS JSON is malformed:", e)
        return False

    if not (isinstance(blob, dict) and
            isinstance(blob.get("inputs"), list) and
            isinstance(blob.get("outputs"), list)):
        print("❌  expected a dict with \"inputs\" and \"outputs\" arrays")
        return False

    if len(blob["inputs"]) != len(blob["outputs"]):
        print("❌  inputs / outputs length mismatch")
        return False

    tests = list(zip(blob["inputs"], blob["outputs"]))

    # 1) Write the candidate code to a temp module ----------------------------
    with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as f:
        f.write(code_str.encode())
        mod_path = f.name

    try:
        spec = importlib.util.spec_from_file_location("_candidate", mod_path)
        mod  = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)

        fn = getattr(mod, func_name, None)
        if fn is None or not callable(fn):
            print(f"❌  function {func_name} not found")
            return False

        # 2) Run each public test --------------------------------------------
        for raw_in, raw_out in tests:

            # ---- Build the positional-argument list ----
            if isinstance(raw_in, str):
                # Keep original line breaks; then split into individual lines
                lines: List[str] = raw_in.rstrip("\n").splitlines()
            else:
                # Already a list-like object (rare but legal in APPS)
                lines = raw_in

            args: List[Any] = [_smart_cast(line) if isinstance(line, str) else line
                               for line in lines]

            # ---- Execute candidate solution ----
            try:
                res = fn(*args)
            except Exception as e:
                print("❌  runtime error:", e)
                return False

            # ---- Compare to expected output (string) ----
            expect = raw_out.rstrip("\n") if isinstance(raw_out, str) else str(raw_out)
            if str(res).rstrip("\n") != expect:
                print(f"❌  wrong answer: got {res!r}, expected {expect!r}")
                return False

        return True

    finally:
        os.remove(mod_path)

def mbpp_passes_tests(code_str: str,
                      tests_source: List[str],   # list of assert-statements
                      func_name: str,
                      time_limit: float = 2.0) -> bool:
    """
    Load `code_str`, expose *all* its names, and execute every assertion
    contained in `tests_source` (the MBPP-sanitized public tests).

    Parameters
    ----------
    code_str     : full Python solution (may include helpers)
    tests_source : list[str] of individual 'assert …' lines
    func_name    : entry-point the benchmark cares about
    time_limit   : placeholder for future timeout logic (unused)

    Returns
    -------
    True   – every assertion succeeded
    False  – any assertion failed or a runtime error occurred
    """
    # ------------------------------------------------------------------
    # 0.  Basic sanity checks on the test blob
    # ------------------------------------------------------------------
    if not isinstance(tests_source, list) or not all(isinstance(x, str) for x in tests_source):
        print("❌  MBPP tests must be a list of strings")
        return False

    # ------------------------------------------------------------------
    # 1.  Persist the candidate as a temporary module
    # ------------------------------------------------------------------
    with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as f:
        f.write(code_str.encode())
        mod_path = f.name

    try:
        spec = importlib.util.spec_from_file_location("_candidate", mod_path)
        mod  = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)

        # Ensure the advertised entry-point exists
        fn = getattr(mod, func_name, None)
        if fn is None or not callable(fn):
            print(f"❌  function {func_name} not found")
            return False

        # ------------------------------------------------------------------
        # 2.  Build a sandbox for the asserts
        #    • start with *all* names from the candidate module
        #    • just in case, also bind `candidate` to the entry-point
        # ------------------------------------------------------------------
        env = dict(mod.__dict__)
        env["candidate"] = fn
        env["math"] = math              # added import math

        # ------------------------------------------------------------------
        # 3.  Execute each assertion string
        # ------------------------------------------------------------------
        for line in tests_source:
            try:
                exec(line, env)        # each line is something like
                                        # 'assert check_smaller((1,2,3), …)'
            except AssertionError as e:
                print("❌  assertion failed:", e)
                return False
            except Exception as e:
                print("❌  runtime error while running tests:", e)
                return False

        return True                    # 🎉  all tests passed!

    finally:
        os.remove(mod_path)

def lcb_passes_functional_tests(code_str: str,
                            tests_json: str,          # ← always a *string*
                            func_name: str,
                            time_limit: float = 2.0) -> bool:
    """
    Run `code_str` in a temporary module, import `func_name`, and check it
    against LiveCodeBench functional tests.

    Parameters
    ----------
    code_str   : generated Python source code
    tests_json : JSON string  '[{"input": "...", "output": "...", ...}, …]'
    func_name  : entry‑point to call inside the generated code
    time_limit : (unused here, kept for future extensions)

    Returns
    -------
    True iff every test passes and no runtime error occurs.
    """
    # 0.  Parse test cases (the dataset always gives them as one JSON string).
    try:
        tests = json.loads(tests_json)
    except json.JSONDecodeError as e:
        print("❌  test‑case JSON is malformed:", e)
        return False

    # 1.  Write the candidate module to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as f:
        f.write(code_str.encode())
        mod_path = f.name

    try:
        # 2.  Import it as a throw‑away module
        spec = importlib.util.spec_from_file_location("_candidate", mod_path)
        mod  = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)

        # 3.  Fetch the target function
        fn = getattr(mod, func_name, None)
        if fn is None or not callable(fn):
            print(f"❌  function {func_name} not found")
            return False

        # 4.  Run each public test
        for t in tests:
            raw_in = t["input"]
            expect = t["output"]

            # Each argument comes as a line in the JSON string
            parts = raw_in.splitlines()
            args  = [ast.literal_eval(p) for p in parts]

            try:
                res = fn(*args)
            except Exception as e:
                print("❌  runtime error:", e)
                return False

            if str(res) != expect:
                print(f"❌  wrong answer: got {res!r}, expected {expect!r}")
                return False

        return True

    finally:
        os.remove(mod_path)

