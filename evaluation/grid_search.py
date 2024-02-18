import contextlib
import io
import itertools
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from time import time

from utils import write_results


def evaluate_model(
    base_textbook, other_textbooks, fn, param_comb_dict, param_label_dict, name
):
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            scores = fn(
                base_textbook=base_textbook,
                other_textbooks=other_textbooks,
                **param_comb_dict,
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            scores = {"error": str(e), "traceback": traceback.format_exc()}
    evaluation = {"name": name} | param_label_dict | scores
    print(evaluation)
    return evaluation


def evaluate_models(base_textbook, other_textbooks, param_grid):
    futures = []
    with ProcessPoolExecutor() as executor:
        for name, param_sub_grid in param_grid.items():
            fn = param_sub_grid.pop("fn")
            param_combinations = itertools.product(
                *[
                    v if not isinstance(v, dict) else v.values()
                    for v in param_sub_grid.values()
                ]
            )
            param_labels = itertools.product(*param_sub_grid.values())

            for comb, label in zip(param_combinations, param_labels):
                param_comb_dict = dict(zip(param_sub_grid.keys(), comb))
                param_label_dict = dict(zip(param_sub_grid.keys(), label))
                futures.append(
                    executor.submit(
                        evaluate_model,
                        base_textbook,
                        other_textbooks,
                        fn,
                        param_comb_dict,
                        param_label_dict,
                        name,
                    )
                )

        for future in as_completed(futures):
            yield future.result()


def tune_parameters(base_textbook, other_textbooks, param_grid, results_file):
    start = time()
    results = evaluate_models(base_textbook, other_textbooks, param_grid)
    write_results(results, results_file)
    end = time()
    print(f"Completed in {end-start:.2f} seconds.")
