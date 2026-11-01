"""Helpers for paired factual and do-intervention scenarios."""
from copy import deepcopy


def paired_scenarios(initial_state: dict, factual: dict, intervention: dict):
    """Return two scenarios sharing all inputs except explicitly changed keys."""
    return (deepcopy(initial_state), deepcopy(factual)), (deepcopy(initial_state), {**factual, **intervention})
