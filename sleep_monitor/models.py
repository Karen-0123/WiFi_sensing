# models.py
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime, date


# ══════════════════════════════════════════
# 一、Users 相關模型
# ══════════════════════════════════════════

class UserCreate(BaseModel):
    """新增使用者時，前端傳入的資料格式"""
    username: str
    display_name: Optional[str] = None
    wake_preference: Optional[str] = None  # JSON 字串，例如 '{"window_min": 30}'


class UserResponse(BaseModel):
    """API 回傳使用者資料的格式"""
    id: int
    username: str
    display_name: Optional[str] = None
    created_at: datetime
    wake_preference: Optional[str] = None

    class Config:
        from_attributes = True  # 允許從 sqlite3.Row 物件直接轉換


# ══════════════════════════════════════════
# 二、Sleep Session 相關模型
# ══════════════════════════════════════════

class SessionCreate(BaseModel):
    """建立新睡眠監測 Session 時，前端傳入的資料格式"""
    user_id: int
    started_at: datetime
    date: date  # 入睡當天日期，格式 YYYY-MM-DD


class SessionResponse(BaseModel):
    """API 回傳 Session 基本資訊的格式"""
    id: int
    user_id: int
    date: date
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    status: str                              # recording / analyzing / done
    sleep_score: Optional[float] = None
    avg_respiration_rate: Optional[float] = None
    deep_sleep_minutes: Optional[int] = None
    light_sleep_minutes: Optional[int] = None
    rem_sleep_minutes: Optional[int] = None
    reward_points: int = 0
    created_at: datetime

    class Config:
        from_attributes = True


# ══════════════════════════════════════════
# 三、Respiration Log 相關模型
# ══════════════════════════════════════════

class RespirationLogCreate(BaseModel):
    """新增一筆呼吸數據時，前端傳入的資料格式"""
    session_id: int
    timestamp: datetime
    respiration_rate: float = Field(
        ...,
        ge=0,    # 大於等於 0
        le=60,   # 小於等於 60，過濾明顯異常值
        description="呼吸率（次/分鐘），正常範圍 12–20"
    )
    signal_quality: float = Field(
        ...,
        ge=0.0,  # 最低 0
        le=1.0,  # 最高 1
        description="CSI 訊號品質（0–1）"
    )
    motion_detected: Optional[int] = 0  # 0=無移動, 1=偵測到移動


class RespirationLogResponse(BaseModel):
    """API 回傳單筆呼吸數據的格式"""
    id: int
    session_id: int
    timestamp: datetime
    respiration_rate: float
    signal_quality: float
    inferred_stage: Optional[str] = None   # deep / light / rem / awake
    is_outlier: int                         # 0 或 1
    motion_detected: int                    # 0 或 1

    class Config:
        from_attributes = True


# ══════════════════════════════════════════
# 四、圖表資料模型（前端視覺化專用）
# ══════════════════════════════════════════

class ChartDataPoint(BaseModel):
    """
    時序圖表中的單一資料點。
    對應前端 x 軸（時間）與 y 軸（呼吸率）的格式。
    """
    timestamp: datetime          # x 軸：時間點
    respiration_rate: float      # y 軸：呼吸率
    inferred_stage: Optional[str] = None   # 用於圖表顏色區分睡眠階段
    is_outlier: int = 0          # 前端可用此欄位標記異常點（例如用不同顏色）
    signal_quality: float = 0.0  # 可用於圖表透明度或信心度顯示
    motion_detected: int = 0     # 可用於標記移動事件


class SleepStageBreakdown(BaseModel):
    """睡眠階段分佈（用於圓餅圖或長條圖）"""
    deep_sleep_minutes: int = 0
    light_sleep_minutes: int = 0
    rem_sleep_minutes: int = 0
    awake_minutes: int = 0       # 由 respiration_logs 中 awake 筆數推算


class SessionChartData(BaseModel):
    """
    完整的圖表資料回傳格式。
    包含 summary 摘要 + 時序資料點 + 睡眠階段分佈。
    前端可直接用此物件渲染所有圖表。
    """
    # 基本 Session 資訊
    session_id: int
    user_id: int
    date: date
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    status: str

    # 睡眠評分摘要
    sleep_score: Optional[float] = None
    avg_respiration_rate: Optional[float] = None
    reward_points: int = 0

    # 時序圖表資料（每 3 分鐘一筆）
    chart_data: List[ChartDataPoint] = []

    # 睡眠階段分佈（圓餅圖用）
    stage_breakdown: SleepStageBreakdown = Field(
        default_factory=SleepStageBreakdown
    )

    # 統計資訊
    total_logs: int = 0          # 總筆數
    outlier_count: int = 0       # 異常值筆數


# ══════════════════════════════════════════
# 五、通用回應模型
# ══════════════════════════════════════════

class SuccessResponse(BaseModel):
    """操作成功時的通用回傳格式"""
    success: bool = True
    message: str
    data: Optional[dict] = None  # 可附帶任意額外資訊


class ErrorResponse(BaseModel):
    """操作失敗時的通用回傳格式"""
    success: bool = False
    message: str
    detail: Optional[str] = None  # 可附帶錯誤細節（開發環境用）