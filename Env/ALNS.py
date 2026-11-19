"""针对 V1 ActionPlan 的自适应大邻域搜索；不直接改变物理环境。"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import exp
from random import Random
from time import perf_counter

from .Env import ActionPlan, Snapshot
from .Evaluator import Evaluation, Evaluator
from .PassengerOperator import EntryControlAction
from .TrainOperator import DelayAction, HoldingAction, ReleaseAction, TurnbackAction
from .configure import DecisionConfig


def candidate_actions(snapshot: Snapshot, config: DecisionConfig) -> tuple:
    """只枚举当前可见、与封锁区间相关的列车动作。"""
    facilities = {(f.station, f.approach_direction) for f in config.turnback_facilities}
    actions = []
    for row in snapshot.trains:
        status = row["status"]
        station = row["current_station"]
        tid = row["train_id"]
        if snapshot.phase == "recovery":
            if status == "HELD" and station is not None:
                actions.append(ReleaseAction(tid, station))
            continue
        if not row["crosses_blocked_edge"]:
            continue
        if status in ("DWELLING", "READY_TO_DEPART") and station is not None:
            for seconds in (60, 180, 300):
                actions.append(DelayAction(tid, station, seconds))
            if status == "READY_TO_DEPART":
                actions.append(HoldingAction(tid, station))
            if row["next_station"] is not None and (station, row["direction"]) in facilities:
                actions.append(TurnbackAction(tid, station))
        if (status == "RUNNING" and row["next_event_kind"] == "ARRIVAL"
                and row["next_event_time"] is not None
                and row["next_event_time"] <= snapshot.time + config.decision_interval
                and row["has_stop_after_next"]
                and (row["next_station"], row["direction"]) in facilities):
            actions.append(TurnbackAction(tid, row["next_station"]))
    return tuple(actions)


def candidate_controls(snapshot: Snapshot) -> tuple[EntryControlAction, ...]:
    """在故障边界及前一站试探一个窗口的进站配额。"""
    if snapshot.phase != "interruption":
        return ()
    direction = snapshot.disruption.direction
    upstream = snapshot.disruption.from_station - (1 if direction == 0 else -1)
    stations = (snapshot.disruption.from_station, upstream)
    return tuple(EntryControlAction(station, direction, quota)
                 for station in stations if 0 <= station < snapshot.platform.shape[1]
                 for quota in (0, 300))


def _with_action(plan: ActionPlan, action) -> ActionPlan:
    """在完整方案中替换同车或同站向的动作，避免重复下令。"""
    if isinstance(action, EntryControlAction):
        controls = [a for a in plan.passenger_actions
                    if (a.station, a.direction) != (action.station, action.direction)]
        controls.append(action)
        controls.sort(key=lambda a: (a.station, a.direction))
        return ActionPlan(plan.train_actions, tuple(controls))
    trains = [a for a in plan.train_actions if a.train_id != action.train_id]
    trains.append(action)
    trains.sort(key=lambda a: a.train_id)
    return ActionPlan(tuple(trains), plan.passenger_actions)


@dataclass(frozen=True)
class SearchResult:
    best_plan: ActionPlan
    best_evaluation: Evaluation
    baseline_evaluation: Evaluation
    candidates: tuple[tuple[ActionPlan, Evaluation], ...]
    best_turnback: tuple[ActionPlan, Evaluation] | None
    candidates_generated: int
    candidates_feasible: int
    infeasible_by_reason: dict[str, int]
    operator_weights: dict[str, float]
    iterations_completed: int
    search_seconds: float


class ALNS:
    """固定迭代预算下搜索联合动作；adaptive=False 即固定算子权重 LNS。"""

    def __init__(self, *, iterations: int = 30, seed: int = 42, adaptive: bool = True,
                 max_seconds: float | None = None):
        """固定搜索次数、随机种子以及是否更新算子抽样权重。"""
        if iterations < 0 or max_seconds is not None and max_seconds <= 0:
            raise ValueError("搜索预算必须为非负迭代数和正的时间上限")
        self.iterations = iterations
        self.seed = seed
        self.adaptive = adaptive
        self.max_seconds = max_seconds

    def solve(self, snapshot: Snapshot, config: DecisionConfig, evaluator: Evaluator) -> SearchResult:
        """返回预测评分最低的可行方案及候选证据。"""
        start = perf_counter()
        rng = Random(self.seed)
        train_pool = candidate_actions(snapshot, config)
        control_pool = candidate_controls(snapshot)
        cache: dict[ActionPlan, Evaluation] = {}
        infeasible = Counter()
        generated = 0

        def evaluate(plan: ActionPlan) -> Evaluation:
            """缓存已试算方案，并统计可行与不可行的原因。"""
            nonlocal generated
            generated += 1
            if plan not in cache:
                result = evaluator.evaluate(plan)
                cache[plan] = result
                if not result.feasible:
                    infeasible[result.reason or "未知原因"] += 1
            return cache[plan]

        empty = ActionPlan()
        baseline = evaluate(empty)
        if not baseline.feasible:
            raise RuntimeError(f"无动作基础方案不可行：{baseline.reason}")
        best_plan = current_plan = empty
        best_eval = current_eval = baseline
        # 逐一试算单车折返，确保“没有折返”可以由评分解释，而非没生成候选。
        for action in train_pool:
            if not isinstance(action, TurnbackAction):
                continue
            plan = _with_action(empty, action)
            result = evaluate(plan)
            if result.feasible and result.score < best_eval.score:
                best_plan = current_plan = plan
                best_eval = current_eval = result

        weights = {name: 1.0 for name in ("remove_train", "remove_control",
                                          "insert_train", "insert_control")}
        rewards = Counter()
        calls = Counter()
        completed = 0
        temperature = max(1.0, baseline.score * 0.02)
        for index in range(self.iterations):
            if self.max_seconds is not None and perf_counter() - start >= self.max_seconds:
                break
            destroy = rng.choices(("remove_train", "remove_control"),
                                  weights=(weights["remove_train"], weights["remove_control"]))[0]
            available_repairs = [name for name, pool in (("insert_train", train_pool),
                                                          ("insert_control", control_pool)) if pool]
            if not available_repairs:
                break
            repair = rng.choices(available_repairs, weights=[weights[x] for x in available_repairs])[0]
            train_actions = list(current_plan.train_actions)
            controls = list(current_plan.passenger_actions)
            if destroy == "remove_train" and train_actions:
                train_actions.pop(rng.randrange(len(train_actions)))
            elif destroy == "remove_control" and controls:
                controls.pop(rng.randrange(len(controls)))
            partial = ActionPlan(tuple(train_actions), tuple(controls))
            pool = train_pool if repair == "insert_train" else control_pool
            candidate = _with_action(partial, rng.choice(pool))
            if candidate != current_plan:
                result = evaluate(candidate)
                if result.feasible:
                    if result.score < best_eval.score:
                        best_plan, best_eval = candidate, result
                        rewards[destroy] += 5
                        rewards[repair] += 5
                    delta = result.score - current_eval.score
                    if delta <= 0 or rng.random() < exp(-delta / temperature):
                        current_plan, current_eval = candidate, result
                        rewards[destroy] += 1
                        rewards[repair] += 1
            calls[destroy] += 1
            calls[repair] += 1
            completed += 1
            temperature *= 0.98
            if self.adaptive and (index + 1) % 10 == 0:
                for name in weights:
                    if calls[name]:
                        weights[name] = 0.8 * weights[name] + 0.2 * max(0.1, rewards[name] / calls[name])
                rewards.clear()
                calls.clear()

        feasible = sorted(((p, r) for p, r in cache.items() if r.feasible),
                          key=lambda pair: (pair[1].score, len(pair[0].train_actions)
                                            + len(pair[0].passenger_actions)))
        turns = [(p, r) for p, r in feasible
                 if any(isinstance(a, TurnbackAction) for a in p.train_actions)]
        return SearchResult(best_plan, best_eval, baseline, tuple(feasible[:3]),
                            turns[0] if turns else None, generated, len(feasible),
                            dict(infeasible), weights, completed, perf_counter() - start)
