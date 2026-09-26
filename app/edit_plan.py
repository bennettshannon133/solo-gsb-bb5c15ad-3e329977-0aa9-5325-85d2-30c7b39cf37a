"""编辑计划优化：在参考时间窗内为每个小区间挑选覆盖片段。

标定换算后的片段先裁剪到参考时间窗，再用全部片段端点与窗口端点把窗口
划分为半开小区间 [p_i, p_{i+1})。每个小区间必须选择一个完整覆盖它的片段，
全局最小化：

    Σ 区间时长 × 所选片段单位毫秒成本
    + 相邻区间所选片段 id 变化次数 × switch_cost

同价时取逐区间 id 序列字典序最小者。任一小区间没有完整覆盖它的片段即
构成空档，抛出 GapError 并附带合并后的空档范围。

可附加非负 min_run_ms 约束：最终计划（同一片段相邻区间已合并）中每段
连续使用同一片段 id 的参考时长都不得少于该值，时间窗首尾两段同样适用。
求解按“段”在区间边界上直接做动态规划，全局搜索满足约束的最低成本方案，
而不是先求无约束方案再删除短段；覆盖充分但不存在满足最短时长的计划时
抛出 MinRunError。
"""

from __future__ import annotations

import bisect
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


class MinRunError(Exception):
    """覆盖充分，但不存在满足最短连续时长 min_run_ms 的可行计划。"""

    def __init__(self, min_run_ms: float) -> None:
        self.min_run_ms = min_run_ms
        super().__init__(
            "覆盖充分，但不存在满足最短连续时长 "
            f"min_run_ms={min_run_ms} 的可行计划"
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


def _best_predecessor(
    best2: list[tuple[int, tuple[float, tuple[str, ...]]]],
    exclude_id: str,
    ids: list[str],
) -> tuple[float, tuple[str, ...]] | None:
    """边界处“最后一段 id 与 exclude_id 不同”的最优前驱状态。"""
    for j, state in best2:
        if ids[j] != exclude_id:
            return state
    return None


def build_edit_plan(
    segments: list[PlanSegment],
    window_start_ms: float,
    window_end_ms: float,
    switch_cost: float,
    min_run_ms: float = 0.0,
) -> EditPlan:
    """求解窗口内总成本最低且每段连续时长不少于 min_run_ms 的片段选择。

    存在空档时抛 GapError；覆盖充分但最短连续时长约束不可满足时抛
    MinRunError。
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

    # 每个小区间都需存在完整覆盖它的片段，否则构成空档
    gap_ranges: list[tuple[float, float]] = []
    for a, b in intervals:
        if not any(lo <= a and hi >= b for _, lo, hi in clipped):
            gap_ranges.append((a, b))
    if gap_ranges:
        raise GapError(_merge_ranges(gap_ranges))

    n = len(intervals)
    ids = [seg.seg_id for seg, _, _ in clipped]
    costs = [seg.cost_per_ms for seg, _, _ in clipped]
    los = [lo for _, lo, _ in clipped]
    his = [hi for _, _, hi in clipped]
    # 最短时长比较留出浮点余量，避免端点相减的舍入误差误杀恰好达标的段
    tol = 1e-9 * max(1.0, min_run_ms)

    # 动态规划按“段”转移（一段 = 最终计划中连续使用同一片段 id 的最大
    # 区间），从而在求解中直接保证最短连续时长，而非先求无约束解再修补：
    # dp[i][j] = (最低成本, 同价下字典序最小的逐区间 id 序列)，表示已覆盖
    # 前 i 个小区间、最后一段使用 clipped[j] 且该段已满足最短时长。转移时
    # 枚举最后一段的起点边界 s：要求 clipped[j] 完整覆盖 [cuts[s], cuts[i])
    # 且段长达标；前驱取边界 s 处最后一段 id 不同的最优状态（s = 0 为首段，
    # 无切换成本）。相邻段 id 必不同，与合并后的计划一一对应。
    dp: list[list[tuple[float, tuple[str, ...]] | None]] = [
        [None] * len(clipped) for _ in range(n + 1)
    ]
    # best2[i]：边界 i 处按 (成本, 序列) 最优的两个 (j, 状态)。片段 id 唯一，
    # “id 不同”至多排除一个状态，前二名必然够用。
    best2: list[list[tuple[int, tuple[float, tuple[str, ...]]]]] = [
        [] for _ in range(n + 1)
    ]

    for i in range(1, n + 1):
        end_cut = cuts[i]
        for j in range(len(clipped)):
            if his[j] < end_cut:
                continue  # 片段到不了边界 i
            best: tuple[float, tuple[str, ...]] | None = None
            # 段起点边界 s 需满足 cuts[s] >= los[j]；cuts 升序，段长随 s 增大
            # 而变短，一旦短于 min_run_ms 即可停止枚举
            for s in range(bisect.bisect_left(cuts, los[j], 0, i), i):
                run_ms = end_cut - cuts[s]
                if run_ms < min_run_ms - tol:
                    break
                if s == 0:
                    cand = (run_ms * costs[j], (ids[j],) * i)
                else:
                    prev = _best_predecessor(best2[s], ids[j], ids)
                    if prev is None:
                        continue
                    cand = (
                        prev[0] + switch_cost + run_ms * costs[j],
                        prev[1] + (ids[j],) * (i - s),
                    )
                if best is None or cand < best:
                    best = cand
            dp[i][j] = best
        ranked = sorted(
            (state, j) for j, state in enumerate(dp[i]) if state is not None
        )
        best2[i] = [(j, state) for state, j in ranked[:2]]

    final = [state for state in dp[n] if state is not None]
    if not final:
        raise MinRunError(min_run_ms)
    total_cost, best_seq = min(final)

    # 相邻区间选择同一片段的合并为一个计划项（结果天然按时间升序）
    by_id = {seg.seg_id: seg for seg, _, _ in clipped}
    items: list[PlanItem] = []
    run_id = best_seq[0]
    run_start = intervals[0][0]
    for i in range(1, len(intervals)):
        if best_seq[i] != run_id:
            items.append(_make_item(by_id[run_id], run_start, intervals[i - 1][1]))
            run_id = best_seq[i]
            run_start = intervals[i][0]
    items.append(_make_item(by_id[run_id], run_start, intervals[-1][1]))

    return EditPlan(total_cost=total_cost, items=items)
