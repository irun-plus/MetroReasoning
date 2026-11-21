"""V1 仿真驱动 ALNS：可切换标准基线与拓扑感知/车客协同增强算法。

不修改V1物理状态转移，不添加任何新物理动作。仅通过Snapshot观察与
DecisionConfig公开设施信息组织搜索；硬约束由Evaluator/Env负责判断。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from math import exp, isfinite, log
from random import Random
from statistics import median
from time import perf_counter

from . import Constant as C
from .Env import ActionPlan, Snapshot
from .Evaluator import Evaluation, Evaluator, RobustEvaluator
from .PassengerOperator import EntryControlAction
from .TrainOperator import DelayAction, HoldingAction, ReleaseAction, TurnbackAction
from .configure import DecisionConfig


def candidate_actions(snapshot: Snapshot, config: DecisionConfig) -> tuple:
    """从快照枚举与封锁相关、状态上具备动作资格的列车指令。"""
    facilities = {(f.station, f.approach_direction) for f in config.turnback_facilities}
    actions = []
    for row in snapshot.trains:
        status, station, tid = row["status"], row["current_station"], row["train_id"]
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


def candidate_controls(snapshot: Snapshot, turnbacks: tuple = ()) -> tuple[EntryControlAction, ...]:
    """封锁边界和计划清客车站；仅扩大合法进站配额的检索位置，不添加动作类型。"""
    if snapshot.phase != "interruption":
        return ()
    direction = snapshot.disruption.direction
    upstream = snapshot.disruption.from_station - C.DIRECTION_STEP[direction]
    stations = {snapshot.disruption.from_station, upstream}
    trains_by_id = {row["train_id"]: row for row in snapshot.trains}
    stations.update(a.station for a in turnbacks if isinstance(a, TurnbackAction)
                    and trains_by_id.get(a.train_id, {}).get("direction") == direction)
    return tuple(EntryControlAction(station, direction, quota)
                 for station in sorted(stations) if 0 <= station < snapshot.platform.shape[1]
                 for quota in (0, 300))


def _with_action(plan: ActionPlan, action) -> ActionPlan:
    """同车/同站向的动作替换，不生成同一对象的冲突指令。"""
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
class TopologyContext:
    """只从决策可观测事实计算的局部拓扑/客流特征。"""
    blocked_from: int
    direction: int
    predicted_remaining_seconds: int
    affected_train_ids: frozenset[int]
    facilities: tuple
    platform_saturation: dict[int, float]
    preferred_turnback_stations: tuple[int, ...]

    @classmethod
    def from_snapshot(cls, snapshot: Snapshot, config: DecisionConfig) -> TopologyContext:
        d = snapshot.disruption.direction
        affected = frozenset(row["train_id"] for row in snapshot.trains
                             if row["crosses_blocked_edge"])
        facilities = tuple(f for f in config.turnback_facilities if f.approach_direction == d)
        ordered = sorted(facilities, key=lambda f: (abs(f.station - snapshot.disruption.from_station),
                                                     f.turnaround_seconds, f.station))
        saturation = {}
        for station in range(C.STATION_NUM):
            used = (int(snapshot.platform[d, station].sum())
                    if station in C.SEPARATE_DIRECTION_CAPACITY_STATIONS
                    else int(snapshot.platform[:, station].sum()))
            saturation[station] = used / max(1, C.PLATFORM_CAPACITY_PEOPLE[station])
        return cls(snapshot.disruption.from_station, d,
                   max(0, snapshot.disruption.predicted_end - snapshot.time),
                   affected, facilities, saturation,
                   tuple(f.station for f in ordered))

    def priority(self, action) -> float:
        """候选的启发式排序分；最终优劣仍由五目标预测评价决定。"""
        station = getattr(action, "station", self.blocked_from)
        dist = abs(station - self.blocked_from)
        crowd = self.platform_saturation.get(station, 0.0)
        if isinstance(action, TurnbackAction):
            f = next((f for f in self.facilities if f.station == station), None)
            if f is None:
                return -100.0
            eta = self.predicted_remaining_seconds / max(1.0, f.turnaround_seconds)
            return 4.0 + min(3.0, eta / 2.0) - 0.2 * dist - 2.0 * crowd
        if isinstance(action, HoldingAction):
            return 2.5 - 0.12 * dist
        if isinstance(action, DelayAction):
            return 1.4 - 0.12 * dist - action.delay_seconds / 1000.0
        if isinstance(action, ReleaseAction):
            return 3.0
        if isinstance(action, EntryControlAction):
            return 1.0 + 2.0 * crowd - 0.12 * dist
        return 0.0


def _coupled_plan(snapshot: Snapshot, plan: ActionPlan, turnback: TurnbackAction,
                  safety_margin_ratio: float = 0.03) -> ActionPlan:
    """折返+清客人数+站台裕度→配额；只是启发式建议，Env仍作最终硬约束判定。"""
    train_row = next((row for row in snapshot.trains if row["train_id"] == turnback.train_id), None)
    if train_row is None:
        return plan
    station = turnback.station
    direction = train_row["direction"]
    on_train = snapshot.onboard[turnback.train_id]
    # 目的地正好是折返站的乘客正常下车，不应重复算为强制清客人数。
    cleared = max(0, int(on_train.sum()) - int(on_train[station]))
    existing = (int(snapshot.platform[direction, station].sum())
                if station in C.SEPARATE_DIRECTION_CAPACITY_STATIONS
                else int(snapshot.platform[:, station].sum()))
    forecast_inflow = sum(float(od[direction, station].sum())
                          for minute, od in zip(snapshot.forecast_times, snapshot.forecast_od)
                          if snapshot.time < minute <= snapshot.time + 300)
    cap = C.PLATFORM_CAPACITY_PEOPLE[station]
    target = min(cap, int(C.PLATFORM_WARNING_PEOPLE[station] *
                          (1 - safety_margin_ratio)))
    quota = max(0, int(target - existing - cleared - forecast_inflow))
    quota = min(300, quota)  # 不增加原版允许的单窗口配额上限
    return _with_action(_with_action(plan, turnback),
                        EntryControlAction(station, direction, quota))


def _remove_related(plan: ActionPlan, rng: Random, snapshot: Snapshot) -> ActionPlan:
    """按因果邻近关系移除列车动作及其同站限流，避免随机地拆散一半协同方案。"""
    if plan.train_actions:
        action = rng.choice(plan.train_actions)
        train_row = next((r for r in snapshot.trains if r["train_id"] == action.train_id), None)
        direction = train_row["direction"] if train_row else snapshot.disruption.direction
        return ActionPlan(tuple(a for a in plan.train_actions if a.train_id != action.train_id),
                          tuple(c for c in plan.passenger_actions
                                if (c.station, c.direction) != (action.station, direction)))
    if plan.passenger_actions:
        remove = rng.choice(plan.passenger_actions)
        return ActionPlan(plan.train_actions,
                          tuple(x for x in plan.passenger_actions if x != remove))
    return plan


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
    search_profile: str = "standard"
    unique_candidates_evaluated: int = 0
    operator_calls: dict[str, int] = field(default_factory=dict)
    candidate_audit: tuple[tuple[ActionPlan, Evaluation, str], ...] = ()
    topology_summary: dict = field(default_factory=dict)
    evaluation_budget: int | None = None
    initial_temperature: float = 1.0
    seed_candidates_evaluated: int = 0


class ALNS:
    """ALNS支持standard基准和enhanced拓扑/协同搜索；仿真评价器可以是单场景或鲁棒版。"""

    def __init__(self, *, iterations: int = 30, seed: int = 42, adaptive: bool = True,
                 max_seconds: float | None = None, profile: str = "enhanced",
                 max_evaluations: int | None = None):
        if iterations < 0 or max_seconds is not None and max_seconds <= 0:
            raise ValueError("搜索预算必须为非负迭代数和正的时间上限")
        if profile not in ("standard", "enhanced"):
            raise ValueError("搜索策略只支持standard和enhanced")
        if max_evaluations is not None and max_evaluations < 1:
            raise ValueError("候选评价次数上限必须是正整数")
        self.iterations = iterations
        self.seed = seed
        self.adaptive = adaptive
        self.max_seconds = max_seconds
        self.profile = profile
        self.max_evaluations = max_evaluations

    @staticmethod
    def _temperature_from_deltas(deltas) -> float:
        """使样本中位劣解在初始时有约一半概率被接受。"""
        positive = [float(x) for x in deltas if isfinite(x) and x > 0]
        return median(positive) / log(2) if positive else 1.0

    def solve(self, snapshot: Snapshot, config: DecisionConfig,
              evaluator: Evaluator | RobustEvaluator) -> SearchResult:
        start = perf_counter()
        rng = Random(self.seed)
        context = TopologyContext.from_snapshot(snapshot, config)
        train_pool = candidate_actions(snapshot, config)
        # 仅增强模式把折返关联站加入候选限流地点；标准模式保持旧候选池。
        control_pool = candidate_controls(snapshot,
                                          train_pool if self.profile == "enhanced" else ())
        turnbacks = tuple(a for a in train_pool if isinstance(a, TurnbackAction))
        cache: dict[ActionPlan, Evaluation] = {}
        sources: dict[ActionPlan, str] = {}
        infeasible = Counter()
        generated = 0

        def budget_full() -> bool:
            return self.max_evaluations is not None and len(cache) >= self.max_evaluations

        def time_full() -> bool:
            return self.max_seconds is not None and perf_counter() - start >= self.max_seconds

        def evaluate(plan: ActionPlan, source: str = "neighborhood") -> Evaluation:
            nonlocal generated
            generated += 1
            if plan not in cache:
                if budget_full():
                    raise RuntimeError("候选评价次数预算已用尽")
                result = evaluator.evaluate(plan)
                cache[plan] = result
                sources[plan] = source
                if not result.feasible:
                    infeasible[result.reason or "未知原因"] += 1
            return cache[plan]

        empty = ActionPlan()
        baseline = evaluate(empty, "baseline")
        if not baseline.feasible:
            raise RuntimeError(f"无动作基础方案不可行：{baseline.reason}")
        best_plan = current_plan = empty
        best_eval = current_eval = baseline
        uphill_deltas = []
        # 有次数上限时，预试算最多使用剩余预算的三分之一。
        seed_limit = (None if self.max_evaluations is None
                      else max(0, (self.max_evaluations - 1) // 3))
        fold_limit = (None if seed_limit is None else
                      max(1, seed_limit - 2) if self.profile == "enhanced" and seed_limit else seed_limit)
        seed_actions = turnbacks if turnbacks else (*train_pool, *control_pool)
        for action in seed_actions[:fold_limit]:
            if budget_full() or time_full():
                break
            plan = _with_action(empty, action)
            result = evaluate(plan, "single_turnback" if isinstance(action, TurnbackAction)
                              else "initial_action")
            if result.feasible and result.score > baseline.score:
                uphill_deltas.append(result.score - baseline.score)
            if result.feasible and result.score < best_eval.score:
                best_plan = current_plan = plan
                best_eval = current_eval = result
        if self.profile == "enhanced":
            for action in sorted(turnbacks, key=context.priority, reverse=True)[:3]:
                if (budget_full() or time_full()
                        or seed_limit is not None and len(cache) - 1 >= seed_limit):
                    break
                plan = _coupled_plan(snapshot, empty, action)
                result = evaluate(plan, "initial_coupled_turnback")
                if result.feasible and result.score < best_eval.score:
                    best_plan = current_plan = plan
                    best_eval = current_eval = result
                if result.feasible and result.score > baseline.score:
                    uphill_deltas.append(result.score - baseline.score)
        seed_candidates_evaluated = len(cache) - 1

        destroy_names = ("remove_train", "remove_control") if self.profile == "standard" else (
            "remove_train", "remove_control", "related_destroy")
        repair_names = ("insert_train", "insert_control") if self.profile == "standard" else (
            "insert_train", "insert_control", "topology_insert", "coupled_turnback", "greedy_repair")
        weights = {name: 1.0 for name in (*destroy_names, *repair_names)}
        if self.profile == "enhanced" and turnbacks and snapshot.phase == "interruption":
            # 仅基于公开预测时长/拓扑设置先验抽样偏好；迭代收益仍自适应更新。
            weights["topology_insert"] = 1.6
            weights["coupled_turnback"] = 1.8
            weights["greedy_repair"] = 1.4
        rewards = Counter()
        calls = Counter()
        total_calls = Counter()
        completed = 0
        temperature = self._temperature_from_deltas(uphill_deltas)
        initial_temperature = temperature

        for index in range(self.iterations):
            if budget_full() or time_full():
                break
            destroy = rng.choices(destroy_names, [weights[x] for x in destroy_names])[0]
            available_repairs = [r for r in repair_names
                                 if (r not in ("insert_train", "topology_insert", "coupled_turnback") or
                                     (turnbacks if r == "coupled_turnback" else train_pool))
                                 and (r != "insert_control" or control_pool)
                                 and (r != "greedy_repair" or train_pool or control_pool)]
            if not available_repairs:
                break
            repair = rng.choices(available_repairs, [weights[x] for x in available_repairs])[0]
            train_actions = list(current_plan.train_actions)
            controls = list(current_plan.passenger_actions)
            if destroy == "related_destroy":
                partial = _remove_related(current_plan, rng, snapshot)
            else:
                if destroy == "remove_train" and train_actions:
                    train_actions.pop(rng.randrange(len(train_actions)))
                elif destroy == "remove_control" and controls:
                    controls.pop(rng.randrange(len(controls)))
                partial = ActionPlan(tuple(train_actions), tuple(controls))

            if repair == "insert_train":
                candidate = _with_action(partial, rng.choice(train_pool))
            elif repair == "insert_control":
                candidate = _with_action(partial, rng.choice(control_pool))
            elif repair == "topology_insert":
                ranked = sorted(train_pool, key=context.priority, reverse=True)
                # 从前5个优先候选中抽样，避免始终只选固定一趟列车。
                candidate = _with_action(partial, rng.choice(ranked[:5]))
            elif repair == "coupled_turnback":
                ranked = sorted(turnbacks, key=context.priority, reverse=True)
                candidate = _coupled_plan(snapshot, partial, rng.choice(ranked[:5]))
            elif repair == "greedy_repair":
                all_pool = sorted((*train_pool, *control_pool),
                                  key=context.priority, reverse=True)
                # 评价预算受限，每次只局部试算3个启发式候选。
                trial_actions = rng.sample(all_pool[:min(8, len(all_pool))],
                                           min(3, len(all_pool)))
                options = [(_with_action(partial, a), a) for a in trial_actions]
                tested = []
                for p, _ in options:
                    if p not in cache and (budget_full() or time_full()):
                        break
                    tested.append((p, evaluate(p, "greedy_probe")))
                if not tested:
                    break
                feasible = [(p, r) for p, r in tested if r.feasible]
                candidate = min(feasible, key=lambda p: p[1].score)[0] if feasible else tested[0][0]
            else:
                raise RuntimeError("未知修复算子")

            if candidate != current_plan:
                result = evaluate(candidate, repair)
                if result.feasible:
                    if result.score < best_eval.score:
                        best_plan, best_eval = candidate, result
                        rewards[destroy] += 5
                        rewards[repair] += 5
                    delta = result.score - current_eval.score
                    if delta <= 0 or rng.random() < exp(-delta / max(1e-12, temperature)):
                        current_plan, current_eval = candidate, result
                        rewards[destroy] += 1
                        rewards[repair] += 1
            calls[destroy] += 1
            calls[repair] += 1
            total_calls[destroy] += 1
            total_calls[repair] += 1
            completed += 1
            temperature *= 0.98
            if self.adaptive and (index + 1) % 10 == 0:
                for name in weights:
                    if calls[name]:
                        weights[name] = 0.8 * weights[name] + 0.2 * max(
                            0.1, rewards[name] / calls[name])
                rewards.clear()
                calls.clear()

        if self.profile == "enhanced" and (best_plan.train_actions or best_plan.passenger_actions):
            # 仅去掉"不增加五目标鲁棒成本"的冗余动作，绝不把更差方案称为最优解。
            for action in (*best_plan.train_actions, *best_plan.passenger_actions):
                if budget_full() or time_full():
                    break
                if isinstance(action, EntryControlAction):
                    simpler = ActionPlan(best_plan.train_actions,
                                         tuple(x for x in best_plan.passenger_actions if x != action))
                else:
                    simpler = ActionPlan(tuple(x for x in best_plan.train_actions if x != action),
                                         best_plan.passenger_actions)
                result = evaluate(simpler, "post_pruning")
                if result.feasible and result.score <= best_eval.score + 1e-9:
                    best_plan, best_eval = simpler, result

        feasible = sorted(((p, r) for p, r in cache.items() if r.feasible),
                          key=lambda pair: (pair[1].score,
                                            len(pair[0].train_actions) + len(pair[0].passenger_actions)))
        turns = [(p, r) for p, r in feasible
                 if any(isinstance(a, TurnbackAction) for a in p.train_actions)]
        audit = tuple((p, r, sources.get(p, "neighborhood")) for p, r in cache.items())
        topo_summary = {
            "blocked_from": context.blocked_from,
            "direction": context.direction,
            "predicted_remaining_seconds": context.predicted_remaining_seconds,
            "affected_train_count": len(context.affected_train_ids),
            "preferred_turnback_stations": list(context.preferred_turnback_stations),
            "platform_saturation": {str(k): round(v, 4)
                                    for k, v in context.platform_saturation.items()
                                    if k in context.preferred_turnback_stations},
        }
        return SearchResult(best_plan, best_eval, baseline, tuple(feasible[:3]),
                            turns[0] if turns else None, generated, len(feasible),
                            dict(infeasible), weights, completed, perf_counter() - start,
                            search_profile=self.profile, unique_candidates_evaluated=len(cache),
                            operator_calls=dict(total_calls), candidate_audit=audit,
                            topology_summary=topo_summary,
                            evaluation_budget=self.max_evaluations,
                            initial_temperature=initial_temperature,
                            seed_candidates_evaluated=seed_candidates_evaluated)
