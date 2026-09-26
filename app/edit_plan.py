"""编辑计划优化：在参考时间窗内为每个小区间挑选覆盖片段。

标定换算后的片段先裁剪到参考时间窗，再用全部片段端点与窗口端点把窗口
划分为半开小区间 [p_i, p_{i+1})。每个小区间必须选择一个完整覆盖它的片段，
全局最小化：

    Σ 区间时长 × 所选片段单位毫秒成本
    + 相邻区间所选片段 id 变化次数 × switch_cost

同价时取逐区间 id 序列字典序最小者。任一小区间没有完整覆盖它的片段即
构成空档，抛出 GapError 并附带合并后的空档范围。

最短时长约束：给定非负 min_run_ms，最终计划中每一段连续使用同一片段 id
的参考时间长度（含时间窗首尾两段）都不得小于该值。约束在片段端点构成的
半开区间上全局求解：动态规划以“一次连续使用（run）”为转移单位，枚举
每个 run 覆盖的连续小区间范围并只接受时长达标的 run，因此不会出现先求
无约束方案再删除短段的情况。覆盖充分却没有任何达标方案时抛出
MinRunInfeasibleError，与空档错误区分。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlanSegment:
    """已换算到参考时间轴的候选片段及其设备映射。"""

    seg_id: str
    device: str
    slope: float
    intercept_ms: float
    start_ms: float  # 参考时间起点（未裁剪）
    end_ms: float  # 参考时间终点（未裁剪）
    cost_per_ms: float


@dataclass(frozen=True)
class PlanItem:
    """计划中连续使用同一片段的一段。"""

    seg_id: str
    device: str
    start_ms: float  # 参考时间起点
    end_ms: float  # 参考时间终点
    device_start_ms: float  # 逆变换后的设备本地起点
    device_end_ms: float  # 逆变换后的设备本地终点


@dataclass(frozen=True)
class EditPlan:
    total_cost: float
    items: list[PlanItem]


class GapError(Exception):
    """窗口内存在没有任何片段完整覆盖的空档。"""

    def __init__(self, gaps: list[tuple[float, float]]) -> None:
        self.gaps = gaps
        joined = ", ".join(f"[{a}, {b})" for a, b in gaps)
        super().__init__(f"窗口内存在无片段覆盖的空档：{joined}")


class MinRunInfeasibleError(Exception):
    """覆盖充分，但不存在满足最短连续时长约束的可行计划。"""

    def __init__(self, min_run_ms: float) -> None:
        self.min_run_ms = min_run_ms
        super().__init__(
            f"覆盖充分，但不存在每段连续使用同一片段 id 的参考时长都不少于 "
            f"{min_run_ms} ms 的可行计划（含时间窗首尾段）"
        )


def _merge_ranges(ranges: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """把首尾相接（或重叠）的范围合并成连续范围，输入按起点升序。"""
    merged: list[list[float]] = []
    for start, end in ranges:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def _make_item(seg: PlanSegment, start_ms: float, end_ms: float) -> PlanItem:
    return PlanItem(
        seg_id=seg.seg_id,
        device=seg.device,
        start_ms=start_ms,
        end_ms=end_ms,
        device_start_ms=(start_ms - seg.intercept_ms) / seg.slope,
        device_end_ms=(end_ms - seg.intercept_ms) / seg.slope,
    )


def _offer(
    slots: list[tuple[float, tuple[str, ...]]],
    run_id: str,
    state: tuple[float, tuple[str, ...]],
) -> None:
    """在边界状态槽中保留按 (成本, id 序列) 最优的至多两个、末 run id 互异的状态。

    扩展新 run 时只需要“末 run id 不等于待扩展 id”的最优前驱，因此两个
    id 互异的最优状态足以应对任意排除查询；被淘汰状态在该角色下必被保留
    状态支配。
    """
    for idx, (_, seq) in enumerate(slots):
        if seq[-1] == run_id:
            if state < slots[idx]:
                slots[idx] = state
                slots.sort()
            return
    if len(slots) < 2:
        slots.append(state)
        slots.sort()
    elif state < slots[-1]:
        slots[-1] = state
        slots.sort()


def _best_excluding(
    slots: list[tuple[float, tuple[str, ...]]], run_id: str
) -> tuple[float, tuple[str, ...]] | None:
    """返回末 run id 不等于 run_id 的最优状态，没有则 None。"""
    for state in slots:  # slots 已按 (成本, id 序列) 升序
        if state[1][-1] != run_id:
            return state
    return None


def build_edit_plan(
    segments: list[PlanSegment],
    window_start_ms: float,
    window_end_ms: float,
    switch_cost: float,
    min_run_ms: float = 0.0,
) -> EditPlan:
    """求解窗口内总成本最低的片段选择。

    存在空档时抛 GapError；覆盖充分但没有满足 min_run_ms 的方案时抛
    MinRunInfeasibleError。
    """
    if not window_start_ms < window_end_ms:
        raise ValueError("时间窗起点必须小于终点")
    if switch_cost < 0:
        raise ValueError("切换成本必须非负")
    if min_run_ms < 0:
        raise ValueError("最短连续时长必须非负")

    # 裁剪到窗口，丢弃与窗口无正长度交集的片段
    clipped: list[tuple[PlanSegment, float, float]] = []
    for seg in segments:
        lo = max(seg.start_ms, window_start_ms)
        hi = min(seg.end_ms, window_end_ms)
        if lo < hi:
            clipped.append((seg, lo, hi))

    # 用窗口端点与全部（裁剪后）片段端点划分半开小区间
    points = {window_start_ms, window_end_ms}
    for _, lo, hi in clipped:
        points.add(lo)
        points.add(hi)
    cuts = sorted(points)
    intervals = list(zip(cuts, cuts[1:]))

    # 每个小区间的候选：完整覆盖它的片段；无候选即为空档
    candidates: list[list[int]] = []
    gap_ranges: list[tuple[float, float]] = []
    for a, b in intervals:
        cand = [j for j, (_, lo, hi) in enumerate(clipped) if lo <= a and hi >= b]
        if not cand:
            gap_ranges.append((a, b))
        candidates.append(cand)
    if gap_ranges:
        raise GapError(_merge_ranges(gap_ranges))

    ids = [seg.seg_id for seg, _, _ in clipped]
    costs = [seg.cost_per_ms for seg, _, _ in clipped]
    n = len(intervals)

    # 每个片段覆盖的首个小区间编号（片段覆盖在区间序列上连续，因此从首个
    # 覆盖区间到 i 都由它覆盖）
    first_cell: list[int | None] = [None] * len(clipped)
    for i, cand in enumerate(candidates):
        for j in cand:
            if first_cell[j] is None:
                first_cell[j] = i

    # 以“一次连续使用同一片段（run）”为单位的动态规划。
    # at[b] 保存覆盖完前 b 个小区间的最优状态槽（至多两个末 run id 互异的
    # 状态），状态 = (最低成本, 同价下字典序最小的逐区间 id 序列)。
    # 枚举在第 i 个小区间收尾的新 run：片段 j 连续覆盖小区间 l..i，该 run
    # 的参考时长 cuts[i+1]-cuts[l] 必须 >= min_run_ms；前驱取边界 l 上末
    # run id 与 j 不同的最优状态，再加一次切换成本。首 run 从空状态出发。
    at: list[list[tuple[float, tuple[str, ...]]]] = [
        [] for _ in range(n + 1)
    ]
    at[0] = [(0.0, ())]

    for i in range(n):
        for j in candidates[i]:
            start_l = first_cell[j]
            assert start_l is not None
            for l in range(start_l, i + 1):
                run_ms = cuts[i + 1] - cuts[l]
                if run_ms < min_run_ms:
                    break  # l 越大 run 越短，之后都不可能达标
                if l == 0:
                    pred_cost, pred_seq = 0.0, ()
                else:
                    pred = _best_excluding(at[l], ids[j])
                    if pred is None:
                        continue
                    pred_cost, pred_seq = pred
                cost = (
                    pred_cost
                    + (switch_cost if l > 0 else 0.0)
                    + run_ms * costs[j]
                )
                seq = pred_seq + (ids[j],) * (i - l + 1)
                _offer(at[i + 1], ids[j], (cost, seq))

    if not at[n]:
        raise MinRunInfeasibleError(min_run_ms)

    total_cost, best_seq = at[n][0]

    # 相邻区间选择同一片段的合并为一个计划项（结果天然按时间升序）
    by_id = {seg.seg_id: seg for seg, _, _ in clipped}
    items: list[PlanItem] = []
    run_id = best_seq[0]
    run_start = intervals[0][0]
    for i in range(1, n):
        if best_seq[i] != run_id:
            items.append(_make_item(by_id[run_id], run_start, intervals[i - 1][1]))
            run_id = best_seq[i]
            run_start = intervals[i][0]
    items.append(_make_item(by_id[run_id], run_start, intervals[-1][1]))

    return EditPlan(total_cost=total_cost, items=items)
