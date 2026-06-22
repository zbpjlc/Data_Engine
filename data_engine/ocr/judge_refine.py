"""Judge-and-Refine 引擎：对 Hard Case 执行 render-then-verify 纠错。

核心流程：
1. 获取原始 block 图片 + 三模型 OCR 结果
2. 渲染 OCR 结果为图片（如有）
3. 调用 Qwen3-VL-235B 做 Judge-and-Refine
4. 解析纠正结果，判断是否需要专家标注
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import requests

from data_engine.config import get_config
from data_engine.ocr.renderer import ResultRenderer

logger = logging.getLogger(__name__)


@dataclass
class JudgeRefineResult:
    """Judge-and-Refine 单轮结果。"""
    corrected_text: str | None = None
    corrected_table: str | None = None  # JSON string
    corrected_formula: str | None = None
    confidence: float = 0.0
    needs_expert: bool = False
    error_locations: list[dict] = field(default_factory=list)
    judge_output: str = ""
    raw_response: dict = field(default_factory=dict)


@dataclass
class JudgeRefineRound:
    """单轮纠错的完整记录。"""
    round_num: int
    result: JudgeRefineResult
    rendered_image_available: bool = False


class JudgeRefineEngine:
    """Judge-and-Refine 引擎：调用 Qwen3-VL-235B 做 Hard Case 纠错。"""

    def __init__(
        self,
        api_url: str | None = None,
        max_rounds: int | None = None,
        agreement_threshold: float | None = None,
        timeout: int | None = None,
        max_retries: int | None = None,
    ) -> None:
        self._api_url = api_url or get_config(
            "ocr", "judge_refine", "api_url", default="http://localhost:8002"
        )
        self._max_rounds = max_rounds or int(
            get_config("ocr", "judge_refine", "max_rounds", default=3)
        )
        self._threshold = agreement_threshold or float(
            get_config("ocr", "judge_refine", "agreement_threshold", default=0.9)
        )
        self._timeout = timeout or int(
            get_config("ocr", "judge_refine", "timeout", default=120)
        )
        self._max_retries = max_retries or int(
            get_config("ocr", "judge_refine", "max_retries", default=2)
        )
        self._renderer = ResultRenderer()

    def judge_and_refine(
        self,
        block_type: str,
        paddle_text: str | None = None,
        glm_text: str | None = None,
        self_text: str | None = None,
        paddle_table: str | dict | None = None,
        glm_table: str | dict | None = None,
        self_table: str | dict | None = None,
        paddle_formula: str | None = None,
        glm_formula: str | None = None,
        self_formula: str | None = None,
        original_image: bytes | None = None,
    ) -> list[JudgeRefineRound]:
        """对一个 Hard block 执行多轮 Judge-and-Refine。

        Returns:
            每轮的纠错结果列表
        """
        rounds: list[JudgeRefineRound] = []

        # 准备渲染图片
        rendered_image = self._prepare_rendered_image(block_type, paddle_table, glm_table, self_table, paddle_formula, glm_formula, self_formula)

        # 当前结果（初始为 self_ocr 的结果）
        current_text = self_text
        current_table = self_table
        current_formula = self_formula

        for round_num in range(1, self._max_rounds + 1):
            result = self._call_judge_refine(
                block_type=block_type,
                paddle_text=paddle_text,
                glm_text=glm_text,
                current_text=current_text,
                paddle_table=paddle_table,
                glm_table=glm_table,
                current_table=current_table,
                paddle_formula=paddle_formula,
                glm_formula=glm_formula,
                current_formula=current_formula,
                original_image=original_image,
                rendered_image=rendered_image,
                round_num=round_num,
            )

            rounds.append(JudgeRefineRound(
                round_num=round_num,
                result=result,
                rendered_image_available=rendered_image is not None,
            ))

            # 判断终止条件：Judge 认为无错误
            if result.confidence >= self._threshold and not result.needs_expert:
                break

            # 更新当前结果为纠正后的结果
            if result.corrected_text is not None:
                current_text = result.corrected_text
            if result.corrected_table is not None:
                current_table = result.corrected_table
            if result.corrected_formula is not None:
                current_formula = result.corrected_formula

        # 最终判断：如果达到最大轮数仍未通过
        if rounds and rounds[-1].result.confidence < self._threshold:
            rounds[-1].result.needs_expert = True

        return rounds

    def _prepare_rendered_image(
        self,
        block_type: str,
        paddle_table, glm_table, self_table,
        paddle_formula, glm_formula, self_formula,
    ) -> bytes | None:
        """准备渲染图片：优先用 self_ocr 结果，其次 paddle。"""
        if block_type == "formula":
            content = self_formula or paddle_formula
            if content and self._renderer.can_render("formula"):
                return self._renderer.render("formula", content)
        elif block_type == "table":
            content = self_table or paddle_table
            if content and self._renderer.can_render("table"):
                return self._renderer.render("table", content)
        return None

    def _call_judge_refine(
        self,
        block_type: str,
        paddle_text: str | None,
        glm_text: str | None,
        current_text: str | None,
        paddle_table,
        glm_table,
        current_table,
        paddle_formula: str | None,
        glm_formula: str | None,
        current_formula: str | None,
        original_image: bytes | None,
        rendered_image: bytes | None,
        round_num: int,
    ) -> JudgeRefineResult:
        """调用 Qwen3-VL-235B API 执行 Judge-and-Refine。"""
        import base64

        # 构造 prompt
        prompt = self._build_prompt(
            block_type=block_type,
            paddle_text=paddle_text,
            glm_text=glm_text,
            current_text=current_text,
            paddle_table=paddle_table,
            glm_table=glm_table,
            current_table=current_table,
            paddle_formula=paddle_formula,
            glm_formula=glm_formula,
            current_formula=current_formula,
            round_num=round_num,
        )

        # 构造图片列表
        images = []
        if original_image:
            images.append(base64.b64encode(original_image).decode("utf-8"))
        if rendered_image:
            images.append(base64.b64encode(rendered_image).decode("utf-8"))

        # 调用 API
        payload = {
            "model": "qwen3-vl-235b",
            "messages": [
                {"role": "system", "content": "你是文档 OCR 纠错专家。给定原始图片和模型 OCR 结果，判断并纠正错误。"},
                {"role": "user", "content": prompt},
            ],
            "images": images,
            "temperature": 0.1,
            "max_tokens": 2048,
        }

        try:
            response = self._post_with_retry(payload)
            return self._parse_response(response, block_type)
        except Exception as e:
            logger.warning("[JudgeRefine] API 调用失败: %s", e)
            return JudgeRefineResult(
                confidence=0.0,
                needs_expert=True,
                judge_output=f"API 调用失败: {e}",
                raw_response={"error": str(e)},
            )

    def _build_prompt(
        self,
        block_type: str,
        paddle_text, glm_text, current_text,
        paddle_table, glm_table, current_table,
        paddle_formula, glm_formula, current_formula,
        round_num: int,
    ) -> str:
        """构造 Judge-and-Refine prompt。"""
        parts = [f"这是第 {round_num} 轮纠错。\n"]

        if block_type == "table":
            parts.append("## 表格识别结果对比\n")
            parts.append(f"PaddleOCR:\n```\n{self._format_table(paddle_table)}\n```\n")
            parts.append(f"GLM OCR:\n```\n{self._format_table(glm_table)}\n```\n")
            parts.append(f"当前结果:\n```\n{self._format_table(current_table)}\n```\n")
            parts.append("\n请对比原始图片和上述结果，判断当前结果是否正确。如果有错误，请给出纠正后的完整表格结构。")
        elif block_type == "formula":
            parts.append("## 公式识别结果对比\n")
            parts.append(f"PaddleOCR: `{paddle_formula}`\n")
            parts.append(f"GLM OCR: `{glm_formula}`\n")
            parts.append(f"当前结果: `{current_formula}`\n")
            parts.append("\n请对比原始图片和渲染图片，判断当前公式是否正确。如果有错误，请给出纠正后的 LaTeX。")
        else:
            parts.append("## 文本识别结果对比\n")
            parts.append(f"PaddleOCR: {paddle_text}\n")
            parts.append(f"GLM OCR: {glm_text}\n")
            parts.append(f"当前结果: {current_text}\n")
            parts.append("\n请对比原始图片，判断当前文本是否正确。如果有错误，请给出纠正后的文本。")

        parts.append("\n\n请以 JSON 格式返回：\n```json\n{\"corrected\": \"纠正后的内容\", \"confidence\": 0.0-1.0, \"error_locations\": [{\"description\": \"错误描述\"}]}\n```")

        return "\n".join(parts)

    def _format_table(self, table) -> str:
        """格式化表格为可读字符串。"""
        if table is None:
            return "(无)"
        if isinstance(table, str):
            return table
        if isinstance(table, dict):
            return json.dumps(table, ensure_ascii=False, indent=2)
        return str(table)

    def _parse_response(self, response: dict, block_type: str) -> JudgeRefineResult:
        """解析 Qwen3-VL 的响应。"""
        try:
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        except (KeyError, IndexError):
            content = str(response)

        # 尝试从 content 中提取 JSON
        result = JudgeRefineResult(judge_output=content)

        try:
            # 找到 JSON 块
            json_match = None
            if "```json" in content:
                start = content.index("```json") + 7
                end = content.index("```", start)
                json_match = content[start:end].strip()
            elif "```" in content:
                start = content.index("```") + 3
                end = content.index("```", start)
                json_match = content[start:end].strip()
            else:
                # 尝试直接解析
                json_match = content.strip()

            if json_match:
                parsed = json.loads(json_match)
                result.confidence = float(parsed.get("confidence", 0.5))
                result.error_locations = parsed.get("error_locations", [])

                corrected = parsed.get("corrected", "")
                if block_type == "table":
                    result.corrected_table = corrected if corrected else None
                elif block_type == "formula":
                    result.corrected_formula = corrected if corrected else None
                else:
                    result.corrected_text = corrected if corrected else None

                result.needs_expert = result.confidence < self._threshold
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("[JudgeRefine] 解析响应失败: %s", e)
            result.confidence = 0.0
            result.needs_expert = True

        result.raw_response = response
        return result

    def _post_with_retry(self, payload: dict) -> dict:
        """带重试的 API 调用。"""
        last_exc = None
        for attempt in range(self._max_retries):
            try:
                resp = requests.post(
                    f"{self._api_url}/v1/chat/completions",
                    json=payload,
                    timeout=self._timeout,
                )
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                last_exc = exc
                logger.warning("[JudgeRefine] API attempt %d failed: %s", attempt + 1, exc)
        raise RuntimeError(f"[JudgeRefine] API 调用失败（{self._max_retries} 次）: {last_exc}")
