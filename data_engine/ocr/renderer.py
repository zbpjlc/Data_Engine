"""渲染器框架：将 OCR 结果渲染为图片，用于 Judge-and-Refine 的视觉对比。

当前实现：can_render() + render_text() 基础版
TODO：LATEX/HTML 渲染后续接入渲染服务
"""

from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path

logger = logging.getLogger(__name__)


class ResultRenderer:
    """将 OCR 结果渲染为图片。"""

    def can_render(self, block_type: str) -> bool:
        """判断该 block 类型是否支持渲染。"""
        return block_type in ("formula", "table")

    def render(self, block_type: str, content: str | dict | None) -> bytes | None:
        """渲染 OCR 结果为 PNG 图片。

        Args:
            block_type: block 类型 (formula/table)
            content: OCR 结果内容（公式字符串或表格 dict/str）

        Returns:
            PNG 图片字节，渲染失败返回 None
        """
        if not content:
            return None

        if block_type == "formula":
            return self._render_latex(content)
        elif block_type == "table":
            return self._render_html_table(content)
        return None

    def _render_latex(self, formula: str) -> bytes | None:
        """LATEX 公式 → PNG。TODO: 接入 LaTeX 渲染服务。"""
        logger.info("[renderer] LATEX 渲染暂未实现，跳过 render-then-verify")
        return None

    def _render_html_table(self, table: str | dict) -> bytes | None:
        """HTML 表格 → PNG。TODO: 接入 HTML 渲染服务。"""
        logger.info("[renderer] HTML 表格渲染暂未实现，跳过 render-then-verify")
        return None
