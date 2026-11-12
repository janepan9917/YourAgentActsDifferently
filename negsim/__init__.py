"""Simulation package for the negotiation experiments (run with ``python -m negsim.main``)."""
from importlib import import_module

# Games used in the paper: Buy-Sell, Ultimatum, CaSiNo, Deal or No Deal, and the
# three stripped PrincipalBench settings (Service Contract, Car Sale, Bike Sale).
for _module_name in (
    "na_harness_fixed_games",
    "allocation_games",
    "principal_bench_stripped_game",
):
    import_module(f"{__name__}.{_module_name}")
