from __future__ import annotations

import copy
from typing import Callable, List, Sequence, Tuple

import torch

from .model import LittleLM

ModelFactory = Callable[[], LittleLM]


def clone_model(model: LittleLM) -> LittleLM:
    clone = LittleLM(embed_dim=model.embed_dim, num_heads=model.num_heads)
    clone.load_state_dict(copy.deepcopy(model.state_dict()))
    clone.eval()
    return clone


def init_population(pop_size: int, model_factory: ModelFactory) -> List[LittleLM]:
    return [model_factory() for _ in range(pop_size)]


def evaluate_population(models: Sequence[LittleLM], evaluator: Callable[[LittleLM, int], dict]):
    fitnesses = []
    metrics = []
    for idx, model in enumerate(models):
        result = evaluator(model, idx)
        fitnesses.append(result["fitness"])
        metrics.append(result)
    return fitnesses, metrics


def select_elite(
    models: Sequence[LittleLM], fitnesses: Sequence[float], elite_frac: float
) -> Tuple[List[LittleLM], List[int]]:
    elite_count = max(1, int(len(models) * elite_frac))
    ranked = sorted(enumerate(fitnesses), key=lambda pair: pair[1], reverse=True)
    elite_indices = [idx for idx, _ in ranked[:elite_count]]
    elites = [models[idx] for idx in elite_indices]
    return elites, elite_indices


def crossover(parent_a: LittleLM, parent_b: LittleLM) -> LittleLM:
    child = LittleLM(embed_dim=parent_a.embed_dim, num_heads=parent_a.num_heads)
    child.eval()
    with torch.no_grad():
        for child_param, param_a, param_b in zip(
            child.parameters(), parent_a.parameters(), parent_b.parameters()
        ):
            if child_param.data.shape != param_a.data.shape:
                child_param.copy_(param_a)
                continue
            mask = torch.rand_like(child_param, dtype=torch.float32) < 0.5
            child_param.copy_(torch.where(mask, param_a, param_b))
    return child


def mutate(model: LittleLM, mutation_prob: float, mutation_std: float) -> None:
    if mutation_prob <= 0 or mutation_std <= 0:
        return
    with torch.no_grad():
        for param in model.parameters():
            if not param.requires_grad:
                continue
            rand_tensor = torch.rand_like(param, dtype=torch.float32)
            mask = rand_tensor < mutation_prob
            if not mask.any():
                continue
            noise = torch.randn_like(param) * mutation_std
            param.add_(noise * mask.to(param.dtype))
