# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Atlas TTS provider — WebSocket streaming via the OpenAI-compatible speech API."""

from __future__ import annotations

import json
import time
from typing import Any

import structlog
import websockets.asyncio.client as ws_client

from coval_bench.config import Settings
from coval_bench.providers.base import TTSProvider, TTSResult
from coval_bench.providers.tts._common import finalize_tts_result

logger: structlog.BoundLogger = structlog.get_logger(__name__)

_VALID_MODELS = ("atlas-tts",)
_WS_URL = "wss://api.tts.runatlas.com/v1/audio/speech/stream"
_FALLBACK_SAMPLE_RATE = 24000
_MAX_WS_SIZE = 16 * 1024 * 1024
_LAST_FRAMES_KEPT = 3


class AtlasTTSProvider(TTSProvider):
    """Atlas TTS provider using the streaming WebSocket API (binary PCM frames)."""

    _VALID_MODELS = frozenset(_VALID_MODELS)

    def __init__(self, settings: Settings, model: str, voice: str) -> None:
        if not self._model_supported(model):
            raise ValueError(
                f"Invalid Atlas TTS model {model!r}. Valid: {sorted(self._VALID_MODELS)}"
            )
        if not voice:
            raise ValueError("Atlas TTS requires a voice")
        self._model = model
        self._voice = voice

        api_key_secret = settings.atlas_api_key
        if api_key_secret is None or not api_key_secret.get_secret_value():
            raise ValueError("atlas_api_key is required in Settings")
        self._api_key = api_key_secret.get_secret_value()

    @property
    def name(self) -> str:
        return f"atlas-{self._model}"

    @property
    def model(self) -> str:
        return self._model

    async def synthesize(self, text: str) -> TTSResult:
        audio_chunks: list[bytes] = []
        last_frames: list[str] = []
        start: float | None = None
        first_chunk_at: float | None = None
        sample_rate = _FALLBACK_SAMPLE_RATE
        done = False

        try:
            async with ws_client.connect(
                _WS_URL,
                additional_headers={"Authorization": f"Bearer {self._api_key}"},
                max_size=_MAX_WS_SIZE,
            ) as ws:
                await ws.send(
                    json.dumps({"type": "start", "voice": self._voice, "response_format": "pcm"})
                )

                async for raw in ws:
                    if isinstance(raw, (bytes, bytearray)):
                        if raw:
                            if first_chunk_at is None:
                                first_chunk_at = time.monotonic()
                            audio_chunks.append(bytes(raw))
                        continue

                    frame: dict[str, Any] = json.loads(raw)
                    frame_type = frame.get("type")

                    if frame_type == "error":
                        raise RuntimeError(f"atlas error: {frame.get('message', frame)}")

                    last_frames.append(raw)
                    del last_frames[:-_LAST_FRAMES_KEPT]

                    if frame_type == "ready":
                        sample_rate = _resolve_sample_rate(frame, sample_rate)
                        # t0 — session open, before text dispatch, so setup stays out of TTFA.
                        start = time.monotonic()
                        await ws.send(json.dumps({"type": "text", "text": text}))
                        await ws.send(json.dumps({"type": "done"}))
                    elif frame_type == "audio.start":
                        sample_rate = _resolve_sample_rate(frame, sample_rate)
                    elif frame_type == "audio.done":
                        # `audio.done` carries a per-sentence error flag; unchecked it
                        # would score a truncated clip against the full prompt.
                        if frame.get("error"):
                            index = frame.get("sentence_index")
                            raise RuntimeError(f"atlas sentence {index} failed to synthesize")
                    elif frame_type == "session.done":
                        done = True
                        break

                if not done:
                    raise RuntimeError("connection closed before the session.done frame")

        except Exception as exc:
            logger.warning("atlas_tts_error", provider="atlas", model=self._model, exc_info=exc)
            return finalize_tts_result(
                provider="atlas",
                model=self._model,
                voice=self._voice,
                pcm=b"",
                sample_rate=sample_rate,
                audio_synthesis_start=start,
                first_audio_chunk_at=first_chunk_at,
                last_frames=last_frames,
                error=str(exc),
            )

        return finalize_tts_result(
            provider="atlas",
            model=self._model,
            voice=self._voice,
            pcm=b"".join(audio_chunks),
            sample_rate=sample_rate,
            audio_synthesis_start=start,
            first_audio_chunk_at=first_chunk_at,
            last_frames=last_frames,
        )


def _resolve_sample_rate(frame: dict[str, Any], current: int) -> int:
    """Sample rate declared by *frame*, or *current* when it declares nothing usable."""
    declared = frame.get("sample_rate")
    if isinstance(declared, int) and not isinstance(declared, bool) and declared > 0:
        return declared
    return current
