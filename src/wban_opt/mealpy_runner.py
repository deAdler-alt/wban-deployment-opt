import numpy as np
from mealpy.evolutionary_based import GA
from mealpy.swarm_based import PSO
from mealpy import FloatVar  # <--- Kluczowy import dla Mealpy 3.x

from .objective import ObjectiveContext, objective_from_x

def solve_ga(ctx: ObjectiveContext, epochs: int, pop_size: int, seed: int = None):
    D = ctx.get_D()
    
    # W Mealpy 3.x musimy zdefiniować bounds jako obiekt FloatVar
    bounds = FloatVar(lb=[0.0] * D, ub=[1.0] * D, name="wban_search_space")
    
    # Definicja problemu dla Mealpy 3.x
    # Zmieniamy też klucz 'fit_func' na 'obj_func', co jest standardem w nowej wersji
    problem_dict = {
        "obj_func": lambda x: objective_from_x(x, ctx),
        "bounds": bounds,
        "minmax": "min",
        "log_to": None, # Wyłącz logowanie do pliku/konsoli
    }

    # Model GA
    model = GA.BaseGA(epoch=epochs, pop_size=pop_size)
    
    # Rozwiązanie
    best_agent = model.solve(problem_dict, seed=seed)
    
    return best_agent.solution, best_agent.target.fitness

def solve_pso(ctx: ObjectiveContext, epochs: int, pop_size: int, seed: int = None):
    D = ctx.get_D()
    
    bounds = FloatVar(lb=[0.0] * D, ub=[1.0] * D, name="wban_search_space")
    
    problem_dict = {
        "obj_func": lambda x: objective_from_x(x, ctx),
        "bounds": bounds,
        "minmax": "min",
        "log_to": None,
    }

    # Model PSO
    model = PSO.OriginalPSO(epoch=epochs, pop_size=pop_size)
    best_agent = model.solve(problem_dict, seed=seed)
    
    return best_agent.solution, best_agent.target.fitness