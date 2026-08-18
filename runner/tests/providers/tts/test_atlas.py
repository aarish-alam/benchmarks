# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Atlas WebSocket streaming TTS provider."""

from __future__ import annotations

import json
import wave
from unittest.mock import patch

import pytest

from coval_bench.config import Settings
from coval_bench.providers.tts.atlas import AtlasTTSProvider

from .conftest import FakeWebSocket, make_pcm_bytes

_MODEL = "atlas-tts"
_VOICE = "dax"


def _settings() -> Settings:
    return Settings(
        database_url="postgresql://runner:password@localhost:5432/benchmarks",
        dataset_bucket="test-bucket",
        dataset_id="stt-v1",
        runner_sha="test",
        log_level="DEBUG",
        atlas_api_key="test-atlas-key",
    )


def _events(
    pcm_chunks: list[bytes], *, sample_rate: int = 24000, sentences: int = 1
) -> list[str | bytes]:
    """A full session: ready, then one audio.start/PCM/audio.done per sentence."""
    events: list[str | bytes] = [json.dumps({"type": "ready", "sample_rate": sample_rate})]
    for index in range(sentences):
        events.append(
            json.dumps(
                {
                    "type": "audio.start",
                    "sentence_index": index,
                    "sentence_text": "Hello from Atlas.",
                    "format": "pcm",
                    "sample_rate": sample_rate,
                }
            )
        )
        events.extend(pcm_chunks)
        events.append(
            json.dumps(
                {
                    "type": "audio.done",
                    "sentence_index": index,
                    "total_bytes": sum(len(c) for c in pcm_chunks),
                    "error": False,
                }
            )
        )
    events.append(json.dumps({"type": "session.done", "total_sentences": sentences}))
    return events


@pytest.fixture()
def atlas_settings() -> Settings:
    return _settings()


@pytest.mark.asyncio
async def test_atlas_tts_happy_path(atlas_settings: Settings) -> None:
    ws = FakeWebSocket(_events([make_pcm_bytes(240)]))
    provider = AtlasTTSProvider(atlas_settings, model=_MODEL, voice=_VOICE)

    with patch("coval_bench.providers.tts.atlas.ws_client.connect", return_value=ws):
        result = await provider.synthesize("Hello from Atlas")

    assert result.error is None, f"Unexpected error: {result.error}"
    assert result.ttfa_ms is not None
    assert 0 < result.ttfa_ms < 60_000
    assert result.audio_path is not None
    assert result.audio_path.exists()
    assert result.audio_path.read_bytes()[:4] == b"RIFF"
    assert result.provider == "atlas"
    assert result.model == _MODEL
    assert result.voice == _VOICE
    result.audio_path.unlink()


@pytest.mark.asyncio
async def test_atlas_tts_url_auth_and_frames(atlas_settings: Settings) -> None:
    ws = FakeWebSocket(_events([make_pcm_bytes(240)]))
    captured: dict[str, object] = {}

    def connect_side_effect(url: str, **kwargs: object) -> FakeWebSocket:
        captured["url"] = url
        captured["headers"] = kwargs.get("additional_headers")
        return ws

    provider = AtlasTTSProvider(atlas_settings, model=_MODEL, voice=_VOICE)

    with patch(
        "coval_bench.providers.tts.atlas.ws_client.connect",
        side_effect=connect_side_effect,
    ):
        result = await provider.synthesize("Hello world")

    assert result.error is None
    assert captured["url"] == "wss://api.tts.runatlas.com/v1/audio/speech/stream"
    assert captured["headers"] == {"Authorization": "Bearer test-atlas-key"}
    sent = [json.loads(m) for m in ws.sent if isinstance(m, str)]
    # `start` first, then the whole prompt in one `text` frame, then `done`. pcm is
    # required: the container formats only complete at end of synthesis, which would
    # fold the entire synthesis into TTFA.
    assert sent == [
        {"type": "start", "voice": _VOICE, "response_format": "pcm"},
        {"type": "text", "text": "Hello world"},
        {"type": "done"},
    ]
    if result.audio_path is not None:
        result.audio_path.unlink()


@pytest.mark.asyncio
async def test_atlas_tts_text_submitted_only_after_ready(atlas_settings: Settings) -> None:
    """Nothing is sent between `start` and `ready` — the clock starts on `ready`."""
    ws = FakeWebSocket(_events([make_pcm_bytes(240)]))
    provider = AtlasTTSProvider(atlas_settings, model=_MODEL, voice=_VOICE)

    with patch("coval_bench.providers.tts.atlas.ws_client.connect", return_value=ws):
        result = await provider.synthesize("Hello")

    # The fake replays `ready` only after the first recv, so a `text` frame at index 1
    # proves it waited for it rather than pipelining the prompt behind `start`.
    assert [json.loads(m)["type"] for m in ws.sent if isinstance(m, str)] == [
        "start",
        "text",
        "done",
    ]
    if result.audio_path is not None:
        result.audio_path.unlink()


@pytest.mark.asyncio
async def test_atlas_tts_wav_uses_declared_sample_rate(atlas_settings: Settings) -> None:
    """The rate declared on the wire wins over the default: it varies by voice."""
    ws = FakeWebSocket(_events([make_pcm_bytes(240)], sample_rate=44100))
    provider = AtlasTTSProvider(atlas_settings, model=_MODEL, voice="ember")

    with patch("coval_bench.providers.tts.atlas.ws_client.connect", return_value=ws):
        result = await provider.synthesize("Hello")

    assert result.error is None
    assert result.audio_path is not None
    with wave.open(str(result.audio_path)) as wav_file:
        assert wav_file.getframerate() == 44100
    result.audio_path.unlink()


@pytest.mark.asyncio
async def test_atlas_tts_falls_back_to_default_sample_rate(atlas_settings: Settings) -> None:
    """A missing or unusable sample_rate keeps the default instead of failing the run."""
    events: list[str | bytes] = [
        json.dumps({"type": "ready"}),
        json.dumps({"type": "audio.start", "sentence_index": 0, "sample_rate": 0}),
        make_pcm_bytes(240),
        json.dumps({"type": "audio.done", "sentence_index": 0, "error": False}),
        json.dumps({"type": "session.done", "total_sentences": 1}),
    ]
    ws = FakeWebSocket(events)
    provider = AtlasTTSProvider(atlas_settings, model=_MODEL, voice=_VOICE)

    with patch("coval_bench.providers.tts.atlas.ws_client.connect", return_value=ws):
        result = await provider.synthesize("Hello")

    assert result.error is None
    assert result.audio_path is not None
    with wave.open(str(result.audio_path)) as wav_file:
        assert wav_file.getframerate() == 24000
    result.audio_path.unlink()


@pytest.mark.asyncio
async def test_atlas_tts_multi_sentence_audio_is_concatenated(atlas_settings: Settings) -> None:
    chunk = make_pcm_bytes(240)
    ws = FakeWebSocket(_events([chunk], sentences=3))
    provider = AtlasTTSProvider(atlas_settings, model=_MODEL, voice=_VOICE)

    with patch("coval_bench.providers.tts.atlas.ws_client.connect", return_value=ws):
        result = await provider.synthesize("One. Two. Three.")

    assert result.error is None
    assert result.audio_path is not None
    with wave.open(str(result.audio_path)) as wav_file:
        assert wav_file.getnframes() == 3 * (len(chunk) // 2)
    result.audio_path.unlink()


@pytest.mark.asyncio
async def test_atlas_tts_error_frame(atlas_settings: Settings) -> None:
    events: list[str | bytes] = [
        json.dumps({"type": "ready", "sample_rate": 24000}),
        json.dumps({"type": "error", "message": "Unknown voice 'nope'"}),
    ]
    ws = FakeWebSocket(events)
    provider = AtlasTTSProvider(atlas_settings, model=_MODEL, voice=_VOICE)

    with patch("coval_bench.providers.tts.atlas.ws_client.connect", return_value=ws):
        result = await provider.synthesize("Hello")

    assert result.error is not None
    assert "Unknown voice" in result.error
    assert result.audio_path is None
    assert result.ttfa_ms is None


@pytest.mark.asyncio
async def test_atlas_tts_sentence_error_flag_is_an_error(atlas_settings: Settings) -> None:
    """A set per-sentence error flag fails the row instead of scoring a short clip."""
    events: list[str | bytes] = [
        json.dumps({"type": "ready", "sample_rate": 24000}),
        json.dumps({"type": "audio.start", "sentence_index": 0, "sample_rate": 24000}),
        make_pcm_bytes(240),
        json.dumps({"type": "audio.done", "sentence_index": 0, "error": True}),
        json.dumps({"type": "session.done", "total_sentences": 1}),
    ]
    ws = FakeWebSocket(events)
    provider = AtlasTTSProvider(atlas_settings, model=_MODEL, voice=_VOICE)

    with patch("coval_bench.providers.tts.atlas.ws_client.connect", return_value=ws):
        result = await provider.synthesize("Hello")

    assert result.error is not None
    assert "sentence 0" in result.error
    # No WAV survives, and the reported error is what fails the row — the orchestrator's
    # single failure model discards the measurement even though audio did arrive first.
    assert result.audio_path is None


@pytest.mark.asyncio
async def test_atlas_tts_truncated_stream_is_error(atlas_settings: Settings) -> None:
    events: list[str | bytes] = [
        json.dumps({"type": "ready", "sample_rate": 24000}),
        json.dumps({"type": "audio.start", "sentence_index": 0, "sample_rate": 24000}),
        make_pcm_bytes(240),
    ]
    ws = FakeWebSocket(events)
    provider = AtlasTTSProvider(atlas_settings, model=_MODEL, voice=_VOICE)

    with patch("coval_bench.providers.tts.atlas.ws_client.connect", return_value=ws):
        result = await provider.synthesize("Hello")

    assert result.error is not None
    assert "session.done" in result.error
    assert result.audio_path is None


@pytest.mark.asyncio
async def test_atlas_tts_ignores_unknown_frames(atlas_settings: Settings) -> None:
    events: list[str | bytes] = [
        json.dumps({"type": "ready", "sample_rate": 24000}),
        json.dumps({"type": "audio.start", "sentence_index": 0, "sample_rate": 24000}),
        make_pcm_bytes(240),
        json.dumps({"type": "usage", "characters": 5}),
        make_pcm_bytes(240),
        json.dumps({"type": "audio.done", "sentence_index": 0, "error": False}),
        json.dumps({"type": "session.done", "total_sentences": 1}),
    ]
    ws = FakeWebSocket(events)
    provider = AtlasTTSProvider(atlas_settings, model=_MODEL, voice=_VOICE)

    with patch("coval_bench.providers.tts.atlas.ws_client.connect", return_value=ws):
        result = await provider.synthesize("Hello")

    assert result.error is None
    assert result.ttfa_ms is not None
    assert result.audio_path is not None
    result.audio_path.unlink()


@pytest.mark.asyncio
async def test_atlas_tts_ttfa_measured_from_ready_to_first_binary(
    atlas_settings: Settings,
) -> None:
    """TTFA excludes connection setup: the clock starts on `ready`, not on connect."""
    chunks = [make_pcm_bytes(240), make_pcm_bytes(240), make_pcm_bytes(240)]
    ws = FakeWebSocket(_events(chunks))
    provider = AtlasTTSProvider(atlas_settings, model=_MODEL, voice=_VOICE)

    times = iter([0.0, 0.1])

    with (
        patch(
            "coval_bench.providers.tts.atlas.time.monotonic",
            side_effect=lambda: next(times, 10.0),
        ),
        patch("coval_bench.providers.tts.atlas.ws_client.connect", return_value=ws),
    ):
        result = await provider.synthesize("Hello")

    assert result.error is None
    assert result.ttfa_ms == pytest.approx(100.0)
    assert result.audio_path is not None
    result.audio_path.unlink()


@pytest.mark.asyncio
async def test_atlas_tts_silent_failure_retains_last_frames(atlas_settings: Settings) -> None:
    """No audio and no error frame still yields a diagnosable reason."""
    events: list[str | bytes] = [
        json.dumps({"type": "ready", "sample_rate": 24000}),
        json.dumps({"type": "session.done", "total_sentences": 0}),
    ]
    ws = FakeWebSocket(events)
    provider = AtlasTTSProvider(atlas_settings, model=_MODEL, voice=_VOICE)

    with patch("coval_bench.providers.tts.atlas.ws_client.connect", return_value=ws):
        result = await provider.synthesize("Hello")

    assert result.error is not None
    assert "session.done" in result.error
    assert result.audio_path is None


def test_atlas_tts_invalid_model_raises(atlas_settings: Settings) -> None:
    with pytest.raises(ValueError, match="Invalid Atlas TTS model"):
        AtlasTTSProvider(atlas_settings, model="atlas-tts-2", voice=_VOICE)


def test_atlas_tts_missing_voice_raises(atlas_settings: Settings) -> None:
    with pytest.raises(ValueError, match="requires a voice"):
        AtlasTTSProvider(atlas_settings, model=_MODEL, voice="")


def test_atlas_tts_missing_key_raises() -> None:
    settings = Settings(
        database_url="postgresql://runner:password@localhost:5432/benchmarks",
        dataset_bucket="test-bucket",
        dataset_id="stt-v1",
        runner_sha="test",
        log_level="DEBUG",
    )
    with pytest.raises(ValueError, match="atlas_api_key"):
        AtlasTTSProvider(settings, model=_MODEL, voice=_VOICE)


def test_atlas_tts_empty_key_raises() -> None:
    settings = Settings(
        database_url="postgresql://runner:password@localhost:5432/benchmarks",
        dataset_bucket="test-bucket",
        dataset_id="stt-v1",
        runner_sha="test",
        log_level="DEBUG",
        atlas_api_key="",
    )
    with pytest.raises(ValueError, match="atlas_api_key"):
        AtlasTTSProvider(settings, model=_MODEL, voice=_VOICE)


def test_atlas_tts_provider_name(atlas_settings: Settings) -> None:
    provider = AtlasTTSProvider(atlas_settings, model=_MODEL, voice=_VOICE)
    assert provider.name == "atlas-atlas-tts"
    assert provider.model == _MODEL
