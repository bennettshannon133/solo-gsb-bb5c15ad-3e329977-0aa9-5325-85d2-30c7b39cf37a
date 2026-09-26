"""POST /v1/edit-plan 的行为测试。"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _post_raw(payload: dict):
    # NaN/Infinity 不是合法 JSON 字面量，新版 httpx 会在客户端拒绝序列化；
    # 标准库 json 默认保留这些字面量，用于构造非有限值的原始请求体
    return client.post(
        "/v1/edit-plan",
        content=json.dumps(payload),
        headers={"Content-Type": "application/json"},
    )


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
    assert client.post("/v1/edit-plan", json=_payload(switch_cost=-0.5)).status_code == 422
    # 非有限值以原始 JSON 字面量发送，仍应被服务端拒绝
    seg["cost_per_ms"] = float("inf")
    assert _post_raw(_payload(segments=[seg])).status_code == 422
    assert _post_raw(_payload(switch_cost=float("nan"))).status_code == 422


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
    mapping = r.json()["mappings"]["B"]
    assert mapping["slope"] == pytest.approx(1.0)
    assert mapping["intercept_ms"] == pytest.approx(-100.0)


# ---------- min_run_ms：最短连续时长 ----------

def test_min_run_zero_matches_unconstrained_default():
    # 显式传 0 与不传结果一致，短段不受限制
    segments = [
        {"id": "x", "device": "A", "start_ms": 0.0, "end_ms": 100.0, "cost_per_ms": 0.0},
        {"id": "y", "device": "A", "start_ms": 100.0, "end_ms": 200.0, "cost_per_ms": 0.0},
    ]
    base = dict(segments=segments, window_start_ms=0.0, window_end_ms=200.0, switch_cost=1.0)
    default = client.post("/v1/edit-plan", json=_payload(**base))
    explicit = client.post("/v1/edit-plan", json=_payload(**base, min_run_ms=0.0))
    assert default.status_code == 200 and explicit.status_code == 200
    assert explicit.json() == default.json()
    assert [item["id"] for item in default.json()["plan"]] == ["x", "y"]


def test_min_run_forces_global_replan_not_short_run_deletion():
    # 无约束最优为 cheap/mid/cheap（mid 段 100ms）；min_run_ms=150 时 mid
    # 最多连续 100ms，唯一可行解是 cheap 全程，而不是删掉 mid 后的残方案
    segments = [
        {"id": "cheap", "device": "A", "start_ms": 0.0, "end_ms": 300.0, "cost_per_ms": 1.0},
        {"id": "mid", "device": "A", "start_ms": 100.0, "end_ms": 200.0, "cost_per_ms": 0.0},
    ]
    base = dict(segments=segments, window_start_ms=0.0, window_end_ms=300.0, switch_cost=0.0)
    r = client.post("/v1/edit-plan", json=_payload(**base))
    assert r.status_code == 200, r.text
    assert [item["id"] for item in r.json()["plan"]] == ["cheap", "mid", "cheap"]
    assert r.json()["total_cost"] == pytest.approx(200.0)

    r = client.post("/v1/edit-plan", json=_payload(**base, min_run_ms=150.0))
    assert r.status_code == 200, r.text
    data = r.json()
    assert [item["id"] for item in data["plan"]] == ["cheap"]
    assert data["plan"][0]["start_ms"] == pytest.approx(0.0)
    assert data["plan"][0]["end_ms"] == pytest.approx(300.0)
    assert data["total_cost"] == pytest.approx(300.0)


def test_min_run_picks_costlier_feasible_plan_over_cheaper_infeasible():
    # a+b 免费但 b 段只有 40ms；min_run_ms=50 时只能选更贵的 c 全程
    segments = [
        {"id": "a", "device": "A", "start_ms": 0.0, "end_ms": 60.0, "cost_per_ms": 0.0},
        {"id": "b", "device": "A", "start_ms": 60.0, "end_ms": 100.0, "cost_per_ms": 0.0},
        {"id": "c", "device": "A", "start_ms": 0.0, "end_ms": 100.0, "cost_per_ms": 1.0},
    ]
    r = client.post(
        "/v1/edit-plan",
        json=_payload(
            segments=segments,
            window_start_ms=0.0,
            window_end_ms=100.0,
            switch_cost=0.0,
            min_run_ms=50.0,
        ),
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert [item["id"] for item in data["plan"]] == ["c"]
    assert data["total_cost"] == pytest.approx(100.0)


def test_min_run_shifts_switch_point_to_satisfy_constraint():
    segments = [
        {"id": "a", "device": "A", "start_ms": 0.0, "end_ms": 80.0, "cost_per_ms": 0.0},
        {"id": "b", "device": "A", "start_ms": 50.0, "end_ms": 100.0, "cost_per_ms": 0.0},
    ]
    base = dict(segments=segments, window_start_ms=0.0, window_end_ms=100.0, switch_cost=10.0)
    # 无约束：字典序最小为 a 用满 [0,80) 再切 b，b 段只有 20ms
    r = client.post("/v1/edit-plan", json=_payload(**base))
    assert r.status_code == 200, r.text
    assert [(i["id"], i["start_ms"], i["end_ms"]) for i in r.json()["plan"]] == [
        ("a", 0.0, 80.0),
        ("b", 80.0, 100.0),
    ]
    # min_run_ms=50：切换点提前到 50，两段各 50ms
    r = client.post("/v1/edit-plan", json=_payload(**base, min_run_ms=50.0))
    assert r.status_code == 200, r.text
    data = r.json()
    assert [(i["id"], i["start_ms"], i["end_ms"]) for i in data["plan"]] == [
        ("a", 0.0, 50.0),
        ("b", 50.0, 100.0),
    ]
    assert data["total_cost"] == pytest.approx(10.0)


def test_min_run_applies_to_window_head_and_tail():
    # 首段 100ms、尾段 100ms 均短于 150ms：覆盖充分但无可行计划
    segments = [
        {"id": "x", "device": "A", "start_ms": 0.0, "end_ms": 100.0, "cost_per_ms": 0.0},
        {"id": "y", "device": "A", "start_ms": 100.0, "end_ms": 200.0, "cost_per_ms": 0.0},
    ]
    base = dict(segments=segments, window_start_ms=0.0, window_end_ms=200.0, switch_cost=1.0)
    r = client.post("/v1/edit-plan", json=_payload(**base, min_run_ms=150.0))
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "gaps_ms" not in detail  # 与空档错误可区分
    assert detail["min_run_ms"] == pytest.approx(150.0)
    assert "最短连续时长" in detail["message"]

    # 恰好等于段长时可行
    r = client.post("/v1/edit-plan", json=_payload(**base, min_run_ms=100.0))
    assert r.status_code == 200, r.text
    assert [item["id"] for item in r.json()["plan"]] == ["x", "y"]
    assert r.json()["total_cost"] == pytest.approx(1.0)


def test_min_run_infeasible_when_window_shorter_than_min_run():
    segments = [
        {"id": "only", "device": "A", "start_ms": 0.0, "end_ms": 100.0, "cost_per_ms": 1.0},
    ]
    r = client.post(
        "/v1/edit-plan",
        json=_payload(
            segments=segments,
            window_start_ms=0.0,
            window_end_ms=100.0,
            min_run_ms=150.0,
        ),
    )
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "gaps_ms" not in detail
    assert detail["min_run_ms"] == pytest.approx(150.0)


def test_gap_takes_precedence_over_min_run():
    # 既有空档又满足不了最短时长时，仍按空档 422 返回
    segments = [
        {"id": "s1", "device": "A", "start_ms": 0.0, "end_ms": 100.0, "cost_per_ms": 1.0},
    ]
    r = client.post(
        "/v1/edit-plan",
        json=_payload(
            segments=segments,
            window_start_ms=0.0,
            window_end_ms=200.0,
            min_run_ms=500.0,
        ),
    )
    assert r.status_code == 422
    assert r.json()["detail"]["gaps_ms"] == [{"start_ms": 100.0, "end_ms": 200.0}]


def test_min_run_keeps_lexicographic_tie_break():
    segments = [
        {"id": "b", "device": "A", "start_ms": 0.0, "end_ms": 100.0, "cost_per_ms": 1.0},
        {"id": "a", "device": "A", "start_ms": 0.0, "end_ms": 100.0, "cost_per_ms": 1.0},
    ]
    r = client.post(
        "/v1/edit-plan",
        json=_payload(
            segments=segments,
            window_start_ms=0.0,
            window_end_ms=100.0,
            switch_cost=0.0,
            min_run_ms=50.0,
        ),
    )
    assert r.status_code == 200, r.text
    assert [item["id"] for item in r.json()["plan"]] == ["a"]


def test_min_run_response_keeps_device_local_times():
    # b1 在 B 本地 [140,400]，换算到统一时间约为 [40,300]（最小二乘映射端点
    # 可能有 1 ulp 误差，故让 b1 两端超出窗口以免疫边界噪声）；
    # min_run_ms=100 时唯一可行计划为 a1[0,100) + b1[100,200)
    r = client.post(
        "/v1/edit-plan",
        json=_payload(
            segments=[
                {"id": "a1", "device": "A", "start_ms": 0.0, "end_ms": 100.0, "cost_per_ms": 0.0},
                {"id": "b1", "device": "B", "start_ms": 140.0, "end_ms": 400.0, "cost_per_ms": 0.0},
            ],
            window_start_ms=0.0,
            window_end_ms=200.0,
            switch_cost=10.0,
            min_run_ms=100.0,
        ),
    )
    assert r.status_code == 200, r.text
    data = r.json()
    plan = data["plan"]
    assert [(i["id"], i["start_ms"], i["end_ms"]) for i in plan] == [
        ("a1", 0.0, 100.0),
        ("b1", 100.0, 200.0),
    ]
    # b1 逆变换回 B 本地时间：约 unified + 100
    assert plan[1]["device_start_ms"] == pytest.approx(200.0)
    assert plan[1]["device_end_ms"] == pytest.approx(300.0)
    assert data["total_cost"] == pytest.approx(10.0)


def test_rejects_invalid_min_run_ms():
    assert client.post("/v1/edit-plan", json=_payload(min_run_ms=-1.0)).status_code == 422
    assert _post_raw(_payload(min_run_ms=float("nan"))).status_code == 422
    assert _post_raw(_payload(min_run_ms=float("inf"))).status_code == 422
