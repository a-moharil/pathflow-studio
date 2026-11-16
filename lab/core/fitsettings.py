"""Shared representation fitting limits and reporting."""
import numpy as np
MAX_STEPS = 5000
# Version of the Euclidean picture-fitting procedure; part of every geometry request key.
FIT_VERSION = 'metric-fit-5'
DIAGNOSTIC_DIMS = (8, 16, 32)

def dimensions(value=False):
    if isinstance(value, bool): return [8] if value else []
    if not isinstance(value, (list, tuple)) or any(type(d) is not int or d not in DIAGNOSTIC_DIMS for d in value):
        raise ValueError('Diagnostic dimensions must be selected from 8, 16, 32')
    return sorted(set(value))

def iterations(value):
    if isinstance(value, bool) or int(value) != float(value) or not 25 <= int(value) <= MAX_STEPS:
        raise ValueError('Maximum fitting iterations must be an integer in 25..5000')
    return int(value)

class FitHistory:
    def __init__(self, objective, initial):
        self.objective = objective
        self.rows = [dict(iteration=0, objective=float(initial))]
        self.iteration = 0
    def __call__(self, flat):
        self.iteration += 1
        self.rows.append(dict(iteration=self.iteration, objective=float(self.objective(flat)[0])))
    def record(self, value):
        """Record an objective the solver already evaluated (no extra evaluation)."""
        self.iteration += 1
        self.rows.append(dict(iteration=self.iteration, objective=float(value)))
    def trailing_improvement(self, window):
        """Relative objective decrease over the last ``window`` iterations, or None if too early."""
        if len(self.rows) <= window: return None
        old, new = self.rows[-1-window]['objective'], self.rows[-1]['objective']
        return (old-new)/max(abs(new), 1e-30)
    def finish(self, solution):
        final = dict(iteration=int(solution.nit), objective=float(solution.fun))
        if self.rows[-1]['iteration'] == final['iteration']: self.rows[-1] = final
        else: self.rows.append(final)
        indices = np.unique(np.linspace(0, len(self.rows)-1, min(501,len(self.rows)), dtype=int))
        return [self.rows[i] for i in indices]

def stop_reason(solution, steps, plateau=False):
    if plateau: return 'plateau'
    if solution.success: return 'converged'
    if solution.nit >= steps: return 'iteration_budget'
    if 'EVALUATIONS' in str(solution.message).upper(): return 'evaluation_budget'
    return 'solver_failure'
