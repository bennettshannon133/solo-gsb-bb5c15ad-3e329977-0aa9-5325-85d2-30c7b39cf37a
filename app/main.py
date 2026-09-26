"""FastAPI 入口：多录音机时钟联合标定服务。"""

from __future__ import annotations

import math

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .calibration import CalibrationError, SyncPoint, calibrate
from .edit_plan import GapError, MinRunError, PlanSegment, build_edit_plan
from .models import (
    CalibrateRequest,
    CalibrateResponse,
    EditPlanRequest,
    EditPlanResponse,
    EditPlanItemOut,
    MappingOut,
    SegmentOut,
)

app = FastAPI(
    title="多录音机时钟联合标定服务",
    version="1.2.0",
    description=(
        "对离线采访素材的多台录音机做时钟联合标定：参考设备映射固定为原时间，"
        "其余设备采用正斜率仿射映射，联立全部同步点做最小二乘；并可在标定后的"
        "统一时间轴上为指定时间窗求解总成本最低的片段编辑计划，可附加每段"
        "连续使用同一片段的最短参考时长（min_run_ms）约束。"
    ),
)


def _json_safe(value):
    """递归把非有限浮点替换为字符串，保证错误详情在严格 JSON 下可序列化。"""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)  # 'nan'、'inf'、'-inf'
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    # starlette 1.x 的 JSONResponse 拒绝序列化 NaN/Infinity，而默认处理器会把
    # 原始输入（可能含非有限值）回显在错误详情里导致 422 响应本身渲染失败；
    # 这里先净化详情，保持“非法输入返回 422”的接口行为
    return JSONResponse(
        status_code=422,
        content=jsonable_encoder({"detail": _json_safe(exc.errors())}),
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/calibrate", response_model=CalibrateResponse)
def calibrate_endpoint(req: CalibrateRequest) -> CalibrateResponse:
    known = set(req.devices)

    if req.reference_device not in known:
        raise HTTPException(
            status_code=422,
            detail=f"参考设备 '{req.reference_device}' 不在设备列表中",
        )
    for i, seg in enumerate(req.segments):
        if seg.device not in known:
            raise HTTPException(
                status_code=422,
                detail=f"第 {i} 个片段引用了未知设备 '{seg.device}'",
            )
        if seg.start_ms > seg.end_ms:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"第 {i} 个片段起止颠倒：start_ms={seg.start_ms} "
                    f"大于 end_ms={seg.end_ms}"
                ),
            )

    try:
        result = calibrate(
            devices=req.devices,
            reference_device=req.reference_device,
            sync_points=[
                SyncPoint(sp.device_a, sp.time_a_ms, sp.device_b, sp.time_b_ms)
                for sp in req.sync_points
            ],
            tolerance_ms=req.tolerance_ms,
        )
    except CalibrationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    unified = [
        SegmentOut(
            device=seg.device,
            label=seg.label,
            start_ms=result.mappings[seg.device][0] * seg.start_ms
            + result.mappings[seg.device][1],
            end_ms=result.mappings[seg.device][0] * seg.end_ms
            + result.mappings[seg.device][1],
        )
        for seg in req.segments
    ]
    unified.sort(key=lambda s: s.start_ms)

    return CalibrateResponse(
        reference_device=req.reference_device,
        mappings={
            d: MappingOut(slope=slope, intercept_ms=intercept)
            for d, (slope, intercept) in result.mappings.items()
        },
        segments=unified,
        sync_point_residuals_ms=result.residuals_ms,
        max_residual_ms=max(result.residuals_ms, default=0.0),
    )


@app.post("/v1/edit-plan", response_model=EditPlanResponse)
def edit_plan_endpoint(req: EditPlanRequest) -> EditPlanResponse:
    known = set(req.devices)

    if req.reference_device not in known:
        raise HTTPException(
            status_code=422,
            detail=f"参考设备 '{req.reference_device}' 不在设备列表中",
        )
    for i, seg in enumerate(req.segments):
        if seg.device not in known:
            raise HTTPException(
                status_code=422,
                detail=f"第 {i} 个片段引用了未知设备 '{seg.device}'",
            )
        if seg.start_ms > seg.end_ms:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"第 {i} 个片段起止颠倒：start_ms={seg.start_ms} "
                    f"大于 end_ms={seg.end_ms}"
                ),
            )

    try:
        result = calibrate(
            devices=req.devices,
            reference_device=req.reference_device,
            sync_points=[
                SyncPoint(sp.device_a, sp.time_a_ms, sp.device_b, sp.time_b_ms)
                for sp in req.sync_points
            ],
            tolerance_ms=req.tolerance_ms,
        )
    except CalibrationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    plan_segments = []
    for seg in req.segments:
        slope, intercept = result.mappings[seg.device]
        plan_segments.append(
            PlanSegment(
                seg_id=seg.id,
                device=seg.device,
                slope=slope,
                intercept_ms=intercept,
                start_ms=slope * seg.start_ms + intercept,
                end_ms=slope * seg.end_ms + intercept,
                cost_per_ms=seg.cost_per_ms,
            )
        )

    try:
        plan = build_edit_plan(
            segments=plan_segments,
            window_start_ms=req.window_start_ms,
            window_end_ms=req.window_end_ms,
            switch_cost=req.switch_cost,
            min_run_ms=req.min_run_ms,
        )
    except GapError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "message": str(exc),
                "gaps_ms": [
                    {"start_ms": start, "end_ms": end} for start, end in exc.gaps
                ],
            },
        ) from exc
    except MinRunError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "message": str(exc),
                "min_run_ms": exc.min_run_ms,
            },
        ) from exc

    return EditPlanResponse(
        reference_device=req.reference_device,
        window_start_ms=req.window_start_ms,
        window_end_ms=req.window_end_ms,
        total_cost=plan.total_cost,
        plan=[
            EditPlanItemOut(
                id=item.seg_id,
                device=item.device,
                start_ms=item.start_ms,
                end_ms=item.end_ms,
                device_start_ms=item.device_start_ms,
                device_end_ms=item.device_end_ms,
            )
            for item in plan.items
        ],
    )
