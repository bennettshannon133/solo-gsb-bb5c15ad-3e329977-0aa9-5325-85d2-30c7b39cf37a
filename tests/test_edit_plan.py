"""POST /v1/edit-plan 的行为测试。"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _payload(**overrides):
    # B 相对 A 整体超前 100ms（slope=1, intercept=-100），同步点精确拟合
    body = {
        "devices": ["A", "B"],
        "reference_device": "A",
        "sync_points": [
            {"device_a": "A", "time_a_ms": 0.0, "device_b": "B", "time_b_ms": 100.0},
            {"device_a": "A", "time_a_ms": 1000.0, "device_b": "B", "time_b_ms": 1100.0},
        ],
        "tolerance_ms": 1e-6,
        "segments": [
            {"id": "a1", "device": "A", "start_ms": 0.0, "end_ms": 1000.0, "cost_per_ms": 1.0},
            {"id": "b1", "device": "B", "start_ms": 600.0, "end_ms": 1600.0, "cost_per_ms": 0.5},
        ],
        "window_start_ms": 0.0,
        "window_end_ms": 1500.0,
        "switch_cost": 10.0,
    }
    body.update(overrides)
    return body


# ---------- 成功路径 ----------

def test_success_picks_cheaper_middle_and_inverse_maps_device_times():
    r = client.post("/v1/edit-plan", json=_payload())
    assert r.status_code == 200, r.text
    data = r.json()
    # b1 换算到统一时间为 [500, 1500]；中间区间用 b1 更便宜：
    # 500*1 + 500*0.5 + 500*0.5 + 1 次切换*10 = 1010
    assert data["total_cost"] == pytest.approx(1010.0)
    assert [item["id"] for item in data["plan"]] == ["a1", "b1"]
    first, second = data["plan"]
    assert first["start_ms"] == pytest.approx(0.0)
    assert first["end_ms"] == pytest.approx(500.0)
    assert first["device_start_ms"] == pytest.approx(0.0)
    assert first["device_end_ms"] == pytest.approx(500.0)
    assert second["start_ms"] == pytest.approx(500.0)
    assert second["end_ms"] == pytest.approx(1500.0)
    # 逆变换回 B 本地时间：unified + 100
    assert second["device_start_ms"] == pytest.approx(600.0)
    assert second["device_end_ms"] == pytest.approx(1600.0)


def test_high_switch_cost_prefers_single_segment():
    # s1/s3 免费但分属两段，s2 全程覆盖但收费；切换费高时全程用 s2
    segments = [
        {"id": "s1", "device": "A", "start_ms": 0.0, "end_ms": 100.0, "cost_per_ms": 0.0},
        {"id": "s2", "device": "A", "start_ms": 0.0, "end_ms": 200.0, "cost_per_ms": 1.0},
        {"id": "s3", "device": "A", "start_ms": 100.0, "end_ms": 200.0, "cost_per_ms": 0.0},
    ]
    r = client.post(
        "/v1/edit-plan",
        json=_payload(
            segments=segments, window_start_ms=0.0, window_end_ms=200.0, switch_cost=1000.0
        ),
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert [item["id"] for item in data["plan"]] == ["s2"]
    assert data["plan"][0]["start_ms"] == pytest.approx(0.0)
    assert data["plan"][0]["end_ms"] == pytest.approx(200.0)
    assert data["total_cost"] == pytest.approx(200.0)

    # 切换费低时改用 s1+s3：0 + 0 + 10 = 10
    r = client.post(
        "/v1/edit-plan",
        json=_payload(
            segments=segments, window_start_ms=0.0, window_end_ms=200.0, switch_cost=10.0
        ),
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert [item["id"] for item in data["plan"]] == ["s1", "s3"]
    assert data["total_cost"] == pytest.approx(10.0)


def test_tie_breaks_to_lexicographically_smallest_id_sequence():
    # 全部片段同价且 switch_cost=0：任意选择同价，取逐区间 id 序列字典序最小
    segments = [
        {"id": "b", "device": "A", "start_ms": 0.0, "end_ms": 200.0, "cost_per_ms": 1.0},
        {"id": "a", "device": "A", "start_ms": 0.0, "end_ms": 200.0, "cost_per_ms": 1.0},
        {"id": "c", "device": "A", "start_ms": 100.0, "end_ms": 300.0, "cost_per_ms": 1.0},
    ]
    r = client.post(
        "/v1/edit-plan",
        json=_payload(
            segments=segments, window_start_ms=0.0, window_end_ms=300.0, switch_cost=0.0
        ),
    )
    assert r.status_code == 200, r.text
    data = r.json()
    # 区间序列为 ("a", "a", "c")，合并后是 a:[0,200) + c:[200,300)
    assert [item["id"] for item in data["plan"]] == ["a", "c"]
    assert data["plan"][0]["end_ms"] == pytest.approx(200.0)
    assert data["plan"][1]["start_ms"] == pytest.approx(200.0)
    assert data["total_cost"] == pytest.approx(300.0)


def test_segments_clipped_to_window_and_inverse_with_slope():
    # B 走时是 A 的两倍快：slope=0.5；片段超出窗口的部分被裁掉
    payload = _payload(
        sync_points=[
            {"device_a": "A", "time_a_ms": 0.0, "device_b": "B", "time_b_ms": 0.0},
            {"device_a": "A", "time_a_ms": 1000.0, "device_b": "B", "time_b_ms": 2000.0},
        ],
        segments=[
            {"id": "wide", "device": "B", "start_ms": -1000.0, "end_ms": 4000.0, "cost_per_ms": 2.0},
        ],
        window_start_ms=0.0,
        window_end_ms=1000.0,
        switch_cost=0.0,
    )
    r = client.post("/v1/edit-plan", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    assert len(data["plan"]) == 1
    item = data["plan"][0]
    assert item["start_ms"] == pytest.approx(0.0)
    assert item["end_ms"] == pytest.approx(1000.0)
    # 逆变换：device = unified / 0.5
    assert item["device_start_ms"] == pytest.approx(0.0)
    assert item["device_end_ms"] == pytest.approx(2000.0)
    assert data["total_cost"] == pytest.approx(2000.0)


# ---------- 空档 422 ----------

def test_gap_returns_422_with_merged_ranges():
    segments = [
        {"id": "s1", "device": "A", "start_ms": 0.0, "end_ms": 200.0, "cost_per_ms": 1.0},
        {"id": "s2", "device": "A", "start_ms": 400.0, "end_ms": 600.0, "cost_per_ms": 1.0},
        {"id": "s3", "device": "A", "start_ms": 800.0, "end_ms": 1000.0, "cost_per_ms": 1.0},
    ]
    r = client.post(
        "/v1/edit-plan",
        json=_payload(segments=segments, window_start_ms=0.0, window_end_ms=1000.0),
    )
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["gaps_ms"] == [
        {"start_ms": 200.0, "end_ms": 400.0},
        {"start_ms": 600.0, "end_ms": 800.0},
    ]


def test_empty_segments_reports_whole_window_as_gap():
    r = client.post("/v1/edit-plan", json=_payload(segments=[]))
    assert r.status_code == 422
    assert r.json()["detail"]["gaps_ms"] == [{"start_ms": 0.0, "end_ms": 1500.0}]


# ---------- 最短连续时长 min_run_ms ----------

def test_min_run_forces_global_replan_not_short_run_deletion():
    # s1 覆盖 [0,100) 免费、s2 覆盖全程 [0,300) 成本 1、s3 覆盖 [200,300)
    # 免费。无约束最优：s1:[0,100) + s2:[100,200) + s3:[200,300)，仅一次
    # 切换成本 10；中间 s2 段只有 100ms。min_run=150 时该方案不达标，不能
    # 简单删除 s2（会留下 [100,200) 空档）；全局重排的可行解为全程 s2，
    # 成本 300。
    segments = [
        {"id": "s1", "device": "A", "start_ms": 0.0, "end_ms": 100.0, "cost_per_ms": 0.0},
        {"id": "s2", "device": "A", "start_ms": 0.0, "end_ms": 300.0, "cost_per_ms": 1.0},
        {"id": "s3", "device": "A", "start_ms": 200.0, "end_ms": 300.0, "cost_per_ms": 0.0},
    ]
    payload = _payload(
        segments=segments,
        window_start_ms=0.0,
        window_end_ms=300.0,
        switch_cost=10.0,
    )
    r0 = client.post("/v1/edit-plan", json=payload)
    assert r0.status_code == 200, r0.text
    assert [item["id"] for item in r0.json()["plan"]] == ["s1", "s2", "s3"]

    r = client.post("/v1/edit-plan", json={**payload, "min_run_ms": 150.0})
    assert r.status_code == 200, r.text
    data = r.json()
    assert [item["id"] for item in data["plan"]] == ["s2"]
    assert data["plan"][0]["start_ms"] == pytest.approx(0.0)
    assert data["plan"][0]["end_ms"] == pytest.approx(300.0)
    assert data["plan"][0]["device_start_ms"] == pytest.approx(0.0)
    assert data["plan"][0]["device_end_ms"] == pytest.approx(300.0)
    assert data["total_cost"] == pytest.approx(300.0)


def test_min_run_boundary_lengths_at_window_head_and_tail_apply():
    # b1 覆盖 [500,1500)，a1 覆盖全程；默认样例窗口 [0,1500)
    # 切点只能在 500（唯一内部端点）：a1 首段 500、b1 末段 1000。
    # min_run=500 恰好达标；min_run=501 则首段必然不足（a1 之外没有片段
    # 能覆盖窗口开头），应报与空档可区分的 min_run_infeasible。
    r = client.post("/v1/edit-plan", json={**_payload(), "min_run_ms": 500.0})
    assert r.status_code == 200, r.text
    assert [item["id"] for item in r.json()["plan"]] == ["a1", "b1"]

    r = client.post("/v1/edit-plan", json={**_payload(), "min_run_ms": 501.0})
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["reason"] == "min_run_infeasible"
    assert detail["min_run_ms"] == pytest.approx(501.0)
    assert "gaps_ms" not in detail


def test_min_run_infeasible_is_distinct_from_gap_error():
    # 覆盖充分（存在无约束可行解），但任何切点都无法满足 151ms：
    # 三个等长 150ms 区间，各自由单个片段独占，只能产生三个 150ms 的段。
    segments = [
        {"id": "s1", "device": "A", "start_ms": 0.0, "end_ms": 150.0, "cost_per_ms": 1.0},
        {"id": "s2", "device": "A", "start_ms": 150.0, "end_ms": 300.0, "cost_per_ms": 1.0},
        {"id": "s3", "device": "A", "start_ms": 300.0, "end_ms": 450.0, "cost_per_ms": 1.0},
    ]
    payload = _payload(
        segments=segments,
        window_start_ms=0.0,
        window_end_ms=450.0,
        switch_cost=0.0,
        min_run_ms=151.0,
    )
    r = client.post("/v1/edit-plan", json=payload)
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["reason"] == "min_run_infeasible"
    assert isinstance(detail["message"], str) and detail["message"]

    # 对照：同窗口有空档时返回的是 gaps_ms 结构
    gap_payload = dict(payload)
    gap_payload["segments"] = segments[:2]
    gap_payload.pop("min_run_ms")
    r_gap = client.post("/v1/edit-plan", json=gap_payload)
    assert r_gap.status_code == 422
    assert "gaps_ms" in r_gap.json()["detail"]
    assert "reason" not in r_gap.json()["detail"]


def test_min_run_exactly_equal_to_cut_length_is_feasible():
    segments = [
        {"id": "s1", "device": "A", "start_ms": 0.0, "end_ms": 150.0, "cost_per_ms": 1.0},
        {"id": "s2", "device": "A", "start_ms": 150.0, "end_ms": 300.0, "cost_per_ms": 1.0},
    ]
    r = client.post(
        "/v1/edit-plan",
        json=_payload(
            segments=segments,
            window_start_ms=0.0,
            window_end_ms=300.0,
            switch_cost=0.0,
            min_run_ms=150.0,
        ),
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert [item["id"] for item in data["plan"]] == ["s1", "s2"]
    assert data["total_cost"] == pytest.approx(300.0)


def test_min_run_lexicographic_tie_break_still_applies():
    # 端点 0/100/200/300/400 划出四个等长 100ms 区间：a 覆盖 [0,200)、
    # b 覆盖 [200,400)，z1/z2 全程覆盖，全部同价、switch_cost=0。
    # min_run=200 时可行方案包括全程 z1、全程 z2、a+b，时长成本恒为 400；
    # 逐区间 id 序列字典序上 ("a","a","b","b") 最小（'a' < 'b' < 'z1'）。
    segments = [
        {"id": "a", "device": "A", "start_ms": 0.0, "end_ms": 200.0, "cost_per_ms": 1.0},
        {"id": "b", "device": "A", "start_ms": 200.0, "end_ms": 400.0, "cost_per_ms": 1.0},
        {"id": "z1", "device": "A", "start_ms": 0.0, "end_ms": 400.0, "cost_per_ms": 1.0},
        {"id": "z2", "device": "A", "start_ms": 0.0, "end_ms": 400.0, "cost_per_ms": 1.0},
    ]
    r = client.post(
        "/v1/edit-plan",
        json=_payload(
            segments=segments,
            window_start_ms=0.0,
            window_end_ms=400.0,
            switch_cost=0.0,
            min_run_ms=200.0,
        ),
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert [item["id"] for item in data["plan"]] == ["a", "b"]
    assert data["plan"][0]["start_ms"] == pytest.approx(0.0)
    assert data["plan"][0]["end_ms"] == pytest.approx(200.0)
    assert data["plan"][1]["start_ms"] == pytest.approx(200.0)
    assert data["plan"][1]["end_ms"] == pytest.approx(400.0)
    assert data["total_cost"] == pytest.approx(400.0)


def test_min_run_longer_than_window_is_infeasible_even_single_segment():
    segments = [
        {"id": "s1", "device": "A", "start_ms": 0.0, "end_ms": 100.0, "cost_per_ms": 1.0},
    ]
    r = client.post(
        "/v1/edit-plan",
        json=_payload(
            segments=segments,
            window_start_ms=0.0,
            window_end_ms=100.0,
            switch_cost=0.0,
            min_run_ms=100.0,
        ),
    )
    assert r.status_code == 200, r.text
    r = client.post(
        "/v1/edit-plan",
        json=_payload(
            segments=segments,
            window_start_ms=0.0,
            window_end_ms=100.0,
            switch_cost=0.0,
            min_run_ms=100.001,
        ),
    )
    assert r.status_code == 422
    assert r.json()["detail"]["reason"] == "min_run_infeasible"


def test_min_run_defaults_to_zero_and_rejects_invalid_values():
    r = client.post("/v1/edit-plan", json=_payload())
    assert r.status_code == 200, r.text
    assert r.json()["total_cost"] == pytest.approx(1010.0)

    assert client.post(
        "/v1/edit-plan", json=_payload(min_run_ms=-0.1)
    ).status_code == 422

    # NaN 无法进入标准 JSON，请求模型层同样拒绝非有限值
    from pydantic import ValidationError

    from app.models import EditPlanRequest

    with pytest.raises(ValidationError):
        EditPlanRequest(
            devices=["A"],
            reference_device="A",
            tolerance_ms=0.0,
            segments=[],
            window_start_ms=0.0,
            window_end_ms=1.0,
            switch_cost=0.0,
            min_run_ms=float("nan"),
        )


# ---------- 标定失败沿用原错误 ----------

def test_calibration_failure_passthrough():
    r = client.post(
        "/v1/edit-plan",
        json=_payload(
            sync_points=[
                {"device_a": "A", "time_a_ms": 0.0, "device_b": "B", "time_b_ms": 0.0},
                {"device_a": "A", "time_a_ms": 1000.0, "device_b": "B", "time_b_ms": 1000.0},
                {"device_a": "A", "time_a_ms": 2000.0, "device_b": "B", "time_b_ms": 2050.0},
            ],
            tolerance_ms=5.0,
        ),
    )
    assert r.status_code == 422
    assert "超过容限" in r.json()["detail"]


def test_rank_deficient_calibration_passthrough():
    r = client.post(
        "/v1/edit-plan",
        json=_payload(devices=["A", "B", "C"]),
    )
    assert r.status_code == 422
    assert "不满列秩" in r.json()["detail"]


# ---------- 请求校验 ----------

def test_rejects_negative_or_nonfinite_costs():
    seg = {"id": "x", "device": "A", "start_ms": 0.0, "end_ms": 10.0, "cost_per_ms": -1.0}
    assert client.post("/v1/edit-plan", json=_payload(segments=[seg])).status_code == 422
    seg["cost_per_ms"] = float("inf")
    assert client.post("/v1/edit-plan", json=_payload(segments=[seg])).status_code == 422
    assert client.post("/v1/edit-plan", json=_payload(switch_cost=-0.5)).status_code == 422
    assert client.post(
        "/v1/edit-plan", json=_payload(switch_cost=float("nan"))
    ).status_code == 422


def test_rejects_unordered_window():
    assert client.post(
        "/v1/edit-plan", json=_payload(window_start_ms=100.0, window_end_ms=100.0)
    ).status_code == 422
    assert client.post(
        "/v1/edit-plan", json=_payload(window_start_ms=200.0, window_end_ms=100.0)
    ).status_code == 422


def test_rejects_duplicate_segment_ids():
    segments = [
        {"id": "dup", "device": "A", "start_ms": 0.0, "end_ms": 1500.0, "cost_per_ms": 1.0},
        {"id": "dup", "device": "A", "start_ms": 0.0, "end_ms": 1500.0, "cost_per_ms": 2.0},
    ]
    r = client.post("/v1/edit-plan", json=_payload(segments=segments))
    assert r.status_code == 422


def test_rejects_unknown_device_and_reversed_segment():
    seg = {"id": "x", "device": "ZZ", "start_ms": 0.0, "end_ms": 10.0, "cost_per_ms": 1.0}
    r = client.post("/v1/edit-plan", json=_payload(segments=[seg]))
    assert r.status_code == 422
    assert "未知设备" in r.json()["detail"]

    seg = {"id": "x", "device": "A", "start_ms": 10.0, "end_ms": 5.0, "cost_per_ms": 1.0}
    r = client.post("/v1/edit-plan", json=_payload(segments=[seg]))
    assert r.status_code == 422
    assert "起止颠倒" in r.json()["detail"]


def test_calibrate_endpoint_still_works():
    r = client.post(
        "/v1/calibrate",
        json={
            "devices": ["A", "B"],
            "reference_device": "A",
            "sync_points": [
                {"device_a": "A", "time_a_ms": 0.0, "device_b": "B", "time_b_ms": 100.0},
                {"device_a": "A", "time_a_ms": 1000.0, "device_b": "B", "time_b_ms": 1100.0},
            ],
            "tolerance_ms": 1e-6,
            "segments": [],
        },
    )
    assert r.status_code == 200, r.text
    assert r.json()["mappings"]["B"] == {"slope": 1.0, "intercept_ms": -100.0}
