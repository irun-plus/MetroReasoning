"""对中断待令、折返、恢复放行和车站限流执行自适应大邻域搜索。"""

from dataclasses import asdict, dataclass
from math import exp
from random import Random
from time import perf_counter
from typing import Callable

from .configure import DecisionConfig
from .ALNSEnv import (EvalResult, Snapshot, StationControl, TrainAction,
                      TrainPlan, safe_initial_plan)
from .Constant import MIN_TRAIN_HEADWAY_SECONDS


@dataclass(frozen=True)
class Candidate:
    """保存一个候选方案、预测评价和产生它的算子。"""
    plan: TrainPlan
    eval_result: EvalResult
    score: float
    destroyed_operator: str
    repaired_operator: str

    def to_dict(self):
        """把候选方案及其评价转成结果记录字典。"""
        return {"plan": self.plan.to_dict(), "feasible": self.eval_result.feasible,
                "violations": self.eval_result.violations,
                "metrics": self.eval_result.metrics, "audit": self.eval_result.audit or {},
                "score": self.score,
                "destroyed_operator": self.destroyed_operator,
                "repaired_operator": self.repaired_operator}


@dataclass(frozen=True)
class SearchResult:
    """保存最优可行解、比较候选、耗时和算子统计。"""
    best_plan: TrainPlan | None
    best_eval: EvalResult | None
    candidates: tuple[Candidate, ...]
    iterations: int
    elapsed_sec: float
    scales: dict[str, float]
    operator_weights: dict[str, dict[str, float]]
    evaluated: int
    feasible_evaluated: int
    turnback_options: int = 0
    turnback_feasible: int = 0
    rejection_reasons: dict[str, int] | None = None
    turnback_trials: tuple[dict, ...] = ()

    def to_dict(self):
        """把本轮搜索过程和候选结果转成可写入 JSON 的字典。"""
        return {"best_plan": self.best_plan.to_dict() if self.best_plan else None,
                "best_eval": {"feasible": self.best_eval.feasible,
                              "violations": self.best_eval.violations,
                              "metrics": self.best_eval.metrics,
                              "audit": self.best_eval.audit or {}} if self.best_eval else None,
                "candidates": [candidate.to_dict() for candidate in self.candidates],
                "iterations": self.iterations, "elapsed_sec": self.elapsed_sec,
                "scales": self.scales, "operator_weights": self.operator_weights,
                "evaluated": self.evaluated, "feasible_evaluated": self.feasible_evaluated,
                "turnback_options": self.turnback_options,
                "turnback_feasible": self.turnback_feasible,
                "rejection_reasons": self.rejection_reasons or {},
                "turnback_trials": list(self.turnback_trials)}


def _fingerprint(plan: TrainPlan):
    """按列车动作和车站控制生成方案去重键。"""
    return (tuple(sorted(plan.train_actions)), tuple(sorted(plan.station_controls)))


class ALNSSolver:
    """通过破坏、修复、接受和算子权重更新搜索逐车方案。"""
    def __init__(self, adaptive: bool = True):
        """设定是否启用算子权重自适应；关闭时作为固定 LNS 对照。"""
        self.adaptive = adaptive

    def search(self, snapshot: Snapshot, settings: DecisionConfig,
               evaluate: Callable[[TrainPlan], EvalResult]) -> SearchResult:
        """按当前阶段搜索逐车方案，返回最好方案和可行比较候选。"""
        started = perf_counter()
        rng = Random(settings.seed + snapshot.time)
        initial = safe_initial_plan(snapshot)
        first = evaluate(initial)
        if not first.feasible:
            return SearchResult(None, None, (), 0, perf_counter() - started, {},
                                {}, 1, 0)
        scales = settings.metric_scales
        initial_score = first.score(settings.weights, scales)
        current, current_score = initial, initial_score
        best, best_eval, best_score = initial, first, initial_score
        pool = {_fingerprint(initial): Candidate(initial, first, initial_score, "initial", "initial")}
        destroy_names = (("remove_train", "remove_control", "remove_pair") if snapshot.blocked
                         else ("remove_release", "remove_control", "remove_pair"))
        repair_names = (("retime_train", "set_admission", "short_turn") if snapshot.blocked
                        else ("release_train", "set_admission"))
        destroy_weights = {name: 1.0 for name in destroy_names}
        repair_weights = {name: 1.0 for name in repair_names}
        destroy_scores = {name: 0.0 for name in destroy_names}
        repair_scores = {name: 0.0 for name in repair_names}
        destroy_uses = {name: 0 for name in destroy_names}
        repair_uses = {name: 0 for name in repair_names}
        iterations = 0
        feasible_evaluated = 1
        evaluated = 1
        temperature = max(0.01, initial_score * 0.05)
        turnback_options = []
        for route in settings.turnback_routes:
            station, direction = route.station, route.approach_direction
            if direction != snapshot.break_direction:
                continue
            if (direction == 0 and not 0 < station <= snapshot.break_from) or (direction == 1 and not snapshot.break_from <= station < 22):
                continue
            next_station = station + (1 if direction == 0 else -1)
            for train_id in range(len(snapshot.schedule)):
                arrival = int(snapshot.schedule[train_id, station, 0])
                if (int(snapshot.schedule[train_id, station, 2]) == direction
                        and snapshot.time < arrival <= snapshot.time + settings.decision_interval
                        and snapshot.schedule[train_id, next_station, 0] > arrival):
                    turnback_options.append((train_id, station))

        def choose(weights):
            """按当前算子权重随机挑选一个破坏或修复算子。"""
            names = tuple(weights)
            return rng.choices(names, weights=tuple(weights.values()), k=1)[0]

        def normalize(actions, controls):
            """按当前阶段修复故障边界的列车顺序和发车间隔。"""
            station, next_station = snapshot.break_from, snapshot.break_to
            direction = snapshot.break_direction
            if not snapshot.blocked:
                proposed = {a.train_id: a for a in actions}
                held = dict(snapshot.held_departures)
                rows = sorted((int(snapshot.schedule[i, station, 0]), i)
                              for i in range(len(snapshot.schedule))
                              if int(snapshot.schedule[i, station, 2]) == direction
                              and snapshot.schedule[i, station, 1] > snapshot.time
                              and snapshot.schedule[i, next_station, 0] > snapshot.schedule[i, station, 1])
                previous = max((int(snapshot.schedule[i, station, 1])
                                for i in range(len(snapshot.schedule))
                                if int(snapshot.schedule[i, station, 2]) == direction
                                and 0 < snapshot.schedule[i, station, 1] <= snapshot.time), default=-10**9)
                normalized = []
                for arrival, train_id in rows:
                    departure = int(snapshot.schedule[train_id, station, 1])
                    action = proposed.get(train_id)
                    target = (action.target_departure if action and action.action == "release"
                              else departure + action.delay_sec if action and action.action == "delay"
                              else departure)
                    target = max(target, snapshot.time + 1, arrival,
                                 previous + MIN_TRAIN_HEADWAY_SECONDS,
                                 held.get(train_id, departure if action is None else 0))
                    if target < departure:
                        normalized.append(TrainAction(train_id, station, "release", 0, target))
                    elif target > departure:
                        normalized.append(TrainAction(train_id, station, "delay", target - departure))
                    previous = target
                return TrainPlan(tuple(normalized), tuple(sorted(controls)))
            rows = sorted((int(snapshot.schedule[i, station, 1]), i)
                          for i in range(len(snapshot.schedule))
                          if int(snapshot.schedule[i, station, 2]) == direction
                          and snapshot.schedule[i, station, 1] >= snapshot.time
                          and snapshot.schedule[i, next_station, 0] > snapshot.schedule[i, station, 1])
            previous = max((int(snapshot.schedule[i, station, 1])
                            for i in range(len(snapshot.schedule))
                            if int(snapshot.schedule[i, station, 2]) == direction
                            and 0 < snapshot.schedule[i, station, 1] < snapshot.time), default=-10**9)
            turnbacks = [action for action in actions if action.action == "turnback"]
            turned_ids = {action.train_id for action in turnbacks}
            proposed = {a.train_id: a.delay_sec for a in actions if a.action != "turnback"}
            normalized = []
            for departure, train_id in rows:
                if train_id in turned_ids:
                    continue
                required = snapshot.predicted_end if snapshot.blocked and departure < snapshot.predicted_end else departure
                target = max(departure + proposed.get(train_id, 0), required,
                             previous + MIN_TRAIN_HEADWAY_SECONDS)
                if target > departure:
                    normalized.append(TrainAction(train_id, station, "hold", target - departure))
                previous = target
            return TrainPlan(tuple(normalized + turnbacks), tuple(sorted(controls)))

        turnback_feasible = 0
        rejection_reasons = {}
        turnback_trials = []
        for train_id, station in turnback_options:
            actions = [action for action in initial.train_actions if action.train_id != train_id]
            actions.append(TrainAction(train_id, station, "turnback", 0))
            candidate = normalize(actions, list(initial.station_controls))
            key = _fingerprint(candidate)
            if key in pool:
                continue
            result = evaluate(candidate)
            evaluated += 1
            score = result.score(settings.weights, scales) if result.feasible else None
            turnback_trials.append({"train_id": train_id, "station": station,
                                    "feasible": result.feasible,
                                    "violations": result.violations,
                                    "metrics": result.metrics, "score": score})
            if result.feasible:
                feasible_evaluated += 1
                turnback_feasible += 1
                pool[key] = Candidate(candidate, result, score, "initial", "short_turn")
                if score < best_score - 1e-12:
                    best, best_eval, best_score = candidate, result, score
            else:
                reason = result.violations[0] if result.violations else "未知原因"
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

        current, current_score = best, best_score
        while iterations < settings.alns_iterations and perf_counter() - started < settings.alns_time_limit:
            iterations += 1
            destroyed = choose(destroy_weights)
            repaired = choose(repair_weights)
            destroy_uses[destroyed] += 1
            repair_uses[repaired] += 1
            actions = list(current.train_actions)
            controls = list(current.station_controls)
            if destroyed == "remove_train" and actions:
                actions.pop(rng.randrange(len(actions)))
            elif destroyed == "remove_release" and actions:
                releases = [a for a in actions if a.action == "release"]
                if releases:
                    actions.remove(rng.choice(releases))
            elif destroyed == "remove_pair" and actions:
                index = rng.randrange(len(actions))
                del actions[index:index + 2]
            elif destroyed == "remove_control" and controls:
                controls.pop(rng.randrange(len(controls)))

            if repaired == "retime_train":
                available = list(initial.train_actions)
                if available:
                    target = rng.choice(available)
                    existing = next((a for a in actions if a.train_id == target.train_id), target)
                    if existing.action != "turnback":
                        actions = [a for a in actions if a.train_id != target.train_id]
                        shift = rng.choice((-120, -60, 0, 60, 120, 180))
                        actions.append(TrainAction(target.train_id, target.station, "hold",
                                                   max(0, existing.delay_sec + shift)))
            elif repaired == "short_turn" and turnback_options:
                train_id, station = rng.choice(turnback_options)
                actions = [action for action in actions if action.train_id != train_id]
                actions.append(TrainAction(train_id, station, "turnback", 0))
            elif repaired == "release_train" and snapshot.held_departures:
                train_id, planned = rng.choice(snapshot.held_departures)
                station = snapshot.break_from
                current_departure = int(snapshot.schedule[train_id, station, 1])
                if current_departure > snapshot.time:
                    existing = next((a for a in actions if a.train_id == train_id), None)
                    base = (existing.target_departure if existing and existing.action == "release"
                            else current_departure)
                    target = max(snapshot.time + 1, planned, base + rng.choice((-120, -60, 0, 60, 120)))
                    actions = [a for a in actions if a.train_id != train_id]
                    if target < current_departure:
                        actions.append(TrainAction(train_id, station, "release", 0, target))
            elif repaired == "set_admission":
                station = rng.choice((snapshot.break_from, snapshot.break_to,
                                      max(0, snapshot.break_from - 1),
                                      min(22, snapshot.break_to + 1)))
                direction = snapshot.break_direction
                controls = [c for c in controls if (c.station, c.direction) != (station, direction)]
                controls.append(StationControl(station, direction,
                                               rng.choice((0, 100, 300, 600, 1200))))
            candidate = normalize(actions, controls)
            key = _fingerprint(candidate)
            if key not in pool:
                result = evaluate(candidate)
                evaluated += 1
                if result.feasible:
                    feasible_evaluated += 1
                    score = result.score(settings.weights, scales)
                    pool[key] = Candidate(candidate, result, score, destroyed, repaired)
                    reward = 0.25
                    if score < best_score - 1e-12:
                        best, best_eval, best_score = candidate, result, score
                        reward = 5.0
                    if score <= current_score or rng.random() < exp(-(score - current_score) / temperature):
                        current, current_score = candidate, score
                        reward = max(reward, 1.0)
                    destroy_scores[destroyed] += reward
                    repair_scores[repaired] += reward
                else:
                    reason = result.violations[0] if result.violations else "未知原因"
                    rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
            temperature = max(0.001, temperature * 0.98)
            if self.adaptive and iterations % 10 == 0:
                for weights, scores, uses in ((destroy_weights, destroy_scores, destroy_uses),
                                               (repair_weights, repair_scores, repair_uses)):
                    for name in weights:
                        if uses[name]:
                            weights[name] = max(0.1, 0.8 * weights[name] + 0.2 * scores[name] / uses[name])
                        scores[name] = 0.0
                        uses[name] = 0
        ordered = sorted(pool.values(), key=lambda c: (c.score, _fingerprint(c.plan)))
        selected = ordered[:3]
        turnback = next((item for item in ordered
                         if any(action.action == "turnback" for action in item.plan.train_actions)), None)
        if turnback and not any(turnback is item for item in selected):
            selected = ordered[:2] + [turnback]
        return SearchResult(best, best_eval, tuple(selected), iterations,
                            perf_counter() - started, scales,
                            {"destroy": destroy_weights, "repair": repair_weights},
                            evaluated, feasible_evaluated, len(turnback_options),
                            turnback_feasible, rejection_reasons, tuple(turnback_trials))
