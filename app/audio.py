import base64
import json
import subprocess
import tempfile
import urllib.request
import uuid
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from app.config import Settings
from app.models import Sentiment, ToneFlag, TranscriptResult, TranscriptSegment


WHISPER_SIZE_LIMIT = 25 * 1024 * 1024  # 25 MB — OpenAI hard limit
WHISPER_CHUNK_MINUTES = 10  # split files that exceed the limit into 10-min chunks

OPENAI_DIRECT_AUDIO_EXTENSIONS = {
    ".flac",
    ".m4a",
    ".mp3",
    ".mp4",
    ".mpeg",
    ".mpga",
    ".ogg",
    ".wav",
    ".webm",
}

SUPPORTED_AUDIO_EXTENSIONS = OPENAI_DIRECT_AUDIO_EXTENSIONS | {
    ".3g2",
    ".3gp",
    ".aac",
    ".aif",
    ".aiff",
    ".amr",
    ".au",
    ".caf",
    ".m4b",
    ".m4p",
    ".m4r",
    ".mka",
    ".mkv",
    ".mov",
    ".mp2",
    ".mpg",
    ".oga",
    ".ogv",
    ".opus",
    ".wave",
    ".wma",
}


def audio_duration_seconds(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            return round(frames / float(rate), 2) if rate else None
    except wave.Error:
        return None


class TranscriptionService:
    def __init__(self, settings: Settings):
        self.settings = settings

    def transcribe(
        self,
        path: Path,
        on_partial: Callable[[TranscriptResult], None] | None = None,
    ) -> TranscriptResult:
        provider = self.settings.transcribe_provider.lower()
        if provider == "openai_realtime":
            return self._openai_realtime_transcribe(path, on_partial=on_partial)
        if provider == "openai":
            return self._openai_transcribe(path)
        if provider == "azure_speech":
            return self._azure_speech_transcribe(path)
        return self._mock_transcribe(path)

    def _azure_speech_transcribe(self, path: Path) -> TranscriptResult:
        api_key = self.settings.azure_speech_api_key
        if not api_key:
            raise RuntimeError("AZURE_SPEECH_API_KEY is required when TRANSCRIBE_PROVIDER=azure_speech")

        region = self.settings.azure_speech_region
        url = (
            f"https://{region}.stt.speech.microsoft.com"
            "/speechtotext/transcriptions:transcribe"
            "?api-version=2024-11-15"
        )

        diarization_mode = (self.settings.azure_speech_diarization or "diarization").lower()
        use_channels = diarization_mode == "channel"
        use_diarization = diarization_mode == "diarization"

        # Channel mode needs stereo audio — skip mono conversion
        stereo = use_channels
        upload_path, temporary_path = self._prepare_upload_file(path, stereo=stereo)
        try:
            definition = self._build_azure_speech_definition(use_channels, use_diarization)
            audio_bytes = upload_path.read_bytes()
            body, content_type = self._build_multipart(audio_bytes, json.dumps(definition))
            req = urllib.request.Request(
                url,
                data=body,
                headers={
                    "Content-Type": content_type,
                    "Ocp-Apim-Subscription-Key": api_key,
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.settings.openai_request_timeout_seconds) as resp:
                payload = json.loads(resp.read())
        finally:
            if temporary_path and temporary_path.exists():
                temporary_path.unlink()

        return self._parse_azure_speech_response(payload, path, use_channels=use_channels)

    def _build_azure_speech_definition(self, use_channels: bool, use_diarization: bool) -> dict[str, Any]:
        language = self.settings.azure_speech_language or "th-TH"
        definition: dict[str, Any] = {
            "profanityFilterMode": "None",
            "punctuationMode": self.settings.azure_speech_punctuation_mode or "DictatedAndAutomatic",
        }

        # Language / locale settings
        candidates_raw = (self.settings.azure_speech_language_candidates or "").strip()
        if candidates_raw:
            # Auto-detect among candidate locales
            candidates = [c.strip() for c in candidates_raw.split(",") if c.strip()]
            definition["locales"] = candidates
            definition["languageIdentification"] = {"candidateLocales": candidates}
        else:
            definition["locales"] = [language]

        # Speaker separation
        if use_channels:
            # Stereo telephony: process each channel independently (ch0=Agent, ch1=Customer)
            definition["channels"] = [0, 1]
        elif use_diarization:
            # AI-based diarization on mono/mixed audio
            definition["diarizationSettings"] = {"enabled": True, "maxSpeakerCount": 2}

        return definition

    @staticmethod
    def _build_multipart(audio_data: bytes, definition_json: str) -> tuple[bytes, str]:
        boundary = uuid.uuid4().hex

        def part_text(name: str, value: str, content_type: str) -> bytes:
            return (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n'
                f"Content-Type: {content_type}\r\n"
                f"\r\n"
                f"{value}\r\n"
            ).encode("utf-8")

        def part_binary(name: str, data: bytes, filename: str) -> bytes:
            header = (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                f"Content-Type: application/octet-stream\r\n"
                f"\r\n"
            ).encode("utf-8")
            return header + data + b"\r\n"

        body = (
            part_text("definition", definition_json, "application/json")
            + part_binary("audio", audio_data, "audio.mp3")
            + f"--{boundary}--\r\n".encode("utf-8")
        )
        return body, f"multipart/form-data; boundary={boundary}"

    def _parse_azure_speech_response(
        self, payload: dict[str, Any], path: Path, use_channels: bool = False
    ) -> TranscriptResult:
        phrases = payload.get("phrases") or []
        duration_ms = payload.get("durationMilliseconds")
        duration = round(duration_ms / 1000.0, 3) if duration_ms else audio_duration_seconds(path)

        # Detect language from the first phrase when auto-detection was used
        detected_language = None
        if phrases:
            detected_language = phrases[0].get("locale") or self.settings.azure_speech_language

        segments: list[TranscriptSegment] = []
        for item in phrases:
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            start = round((item.get("offsetMilliseconds") or 0) / 1000.0, 3)
            end = round(start + (item.get("durationMilliseconds") or 0) / 1000.0, 3)

            if use_channels:
                # Channel 0 = Agent side, Channel 1 = Customer side (telephony convention)
                channel = item.get("channel")
                speaker = "Agent" if channel == 0 else "Customer" if channel == 1 else "Unknown"
            else:
                speaker_id = item.get("speaker")
                speaker = f"Speaker {speaker_id}" if speaker_id is not None else "Unknown"

            segments.append(TranscriptSegment(start=start, end=end, speaker=speaker, text=text))

        # Sort by start time (multi-channel phrases may arrive channel-by-channel)
        segments.sort(key=lambda s: s.start)

        if not segments:
            combined = " ".join(
                p.get("text", "") for p in (payload.get("combinedPhrases") or [])
            ).strip()
            segments = [TranscriptSegment(
                start=0, end=float(duration or 0), speaker="Unknown", text=combined or ""
            )]

        return TranscriptResult(
            language=detected_language or self.settings.azure_speech_language,
            duration_seconds=duration,
            segments=segments,
            full_text="\n".join(f"{s.speaker}: {s.text}" for s in segments),
        )

    def _mock_transcribe(self, path: Path) -> TranscriptResult:
        duration = audio_duration_seconds(path) or 240
        step = max(duration / 5, 8)
        sample_lines = [
            ("Customer", "I am worried about a credit card charge and need help checking it."),
            ("Agent", "I can help with that. I will verify your account and review the transaction."),
            ("Customer", "The amount looks unfamiliar and I feel frustrated because it happened twice."),
            ("Agent", "I understand. I found two similar authorizations and will explain the next steps."),
            ("Customer", "Thank you. Please block the suspicious charge and send me the confirmation."),
        ]
        segments: list[TranscriptSegment] = []
        for index, (speaker, text) in enumerate(sample_lines):
            start = round(index * step, 2)
            end = round(min((index + 1) * step - 1, duration), 2)
            segments.append(
                TranscriptSegment(
                    start=start,
                    end=end,
                    speaker=speaker,
                    text=text,
                    sentiment=Sentiment.negative if "frustrated" in text else Sentiment.neutral,
                    tone_flags=[ToneFlag.frustrated] if "frustrated" in text else [],
                    keywords=["credit card", "charge"] if index in {0, 2, 4} else ["verification"],
                )
            )
        return TranscriptResult(
            language="en",
            duration_seconds=duration,
            segments=segments,
            full_text="\n".join(f"{item.speaker}: {item.text}" for item in segments),
        )

    def _openai_transcribe(self, path: Path) -> TranscriptResult:
        client = self._build_transcription_client()

        is_diarization_model = "diarize" in self.settings.openai_transcribe_model
        wants_whisper_segments = self.settings.openai_transcribe_model == "whisper-1"
        response_format = "diarized_json" if is_diarization_model else "verbose_json" if wants_whisper_segments else "json"
        extra_body = (
            {"chunking_strategy": self.settings.openai_chunking_strategy}
            if is_diarization_model
            else None
        )
        timestamp_granularities = ["segment"] if wants_whisper_segments else None
        upload_path, temporary_path = self._prepare_upload_file(path)
        try:
            if upload_path.stat().st_size > WHISPER_SIZE_LIMIT:
                if temporary_path:
                    temporary_path.unlink(missing_ok=True)
                return self._openai_transcribe_chunked(
                    path, client, response_format, timestamp_granularities, extra_body
                )
            with upload_path.open("rb") as audio_file:
                response = client.audio.transcriptions.create(
                    model=self.settings.openai_transcribe_model,
                    file=audio_file,
                    response_format=response_format,
                    language=self._transcribe_language(),
                    timestamp_granularities=timestamp_granularities,
                    extra_body=extra_body,
                    timeout=self.settings.openai_request_timeout_seconds,
                )
            payload = response.model_dump() if hasattr(response, "model_dump") else dict(response)
            return self._parse_openai_response(payload, path)
        finally:
            if temporary_path and temporary_path.exists():
                temporary_path.unlink()

    def _build_transcription_client(self):
        """Return an OpenAI-compatible client for audio transcriptions."""
        azure_endpoint = self.settings.azure_openai_transcribe_endpoint
        azure_key = self.settings.azure_openai_api_key
        if azure_endpoint and azure_key:
            from openai import AzureOpenAI
            return AzureOpenAI(
                api_key=azure_key,
                azure_endpoint=azure_endpoint,
                api_version=self.settings.azure_openai_transcribe_api_version,
            )
        if not self.settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY (or Azure transcription credentials) required when TRANSCRIBE_PROVIDER=openai")
        from openai import OpenAI
        return OpenAI(api_key=self.settings.openai_api_key)

    def _openai_transcribe_chunked(
        self,
        path: Path,
        client,
        response_format: str,
        timestamp_granularities,
        extra_body,
    ) -> TranscriptResult:
        chunks = self._split_audio_chunks(path)
        if not chunks:
            raise RuntimeError("Could not split oversized audio file — ffprobe/ffmpeg may be missing")
        all_segments: list[TranscriptSegment] = []
        language: str | None = None
        total_duration: float | None = None
        try:
            for chunk_path, offset in chunks:
                with chunk_path.open("rb") as audio_file:
                    response = client.audio.transcriptions.create(
                        model=self.settings.openai_transcribe_model,
                        file=audio_file,
                        response_format=response_format,
                        language=self._transcribe_language(),
                        timestamp_granularities=timestamp_granularities,
                        extra_body=extra_body,
                        timeout=self.settings.openai_request_timeout_seconds,
                    )
                payload = response.model_dump() if hasattr(response, "model_dump") else dict(response)
                chunk_result = self._parse_openai_response(payload, chunk_path)
                if language is None:
                    language = chunk_result.language
                for seg in chunk_result.segments:
                    all_segments.append(
                        seg.model_copy(update={"start": seg.start + offset, "end": seg.end + offset})
                    )
                chunk_duration = chunk_result.duration_seconds
                if chunk_duration is not None:
                    total_duration = offset + chunk_duration
        finally:
            for chunk_path, _ in chunks:
                chunk_path.unlink(missing_ok=True)
        if not all_segments:
            raise RuntimeError("Chunked transcription returned no segments")
        return TranscriptResult(
            language=language,
            duration_seconds=total_duration,
            segments=all_segments,
            full_text="\n".join(f"{s.speaker}: {s.text}" for s in all_segments),
        )

    def _split_audio_chunks(self, path: Path) -> list[tuple[Path, float]]:
        duration = self._probe_duration_seconds(path)
        if not duration:
            return []
        chunk_secs = WHISPER_CHUNK_MINUTES * 60
        chunks: list[tuple[Path, float]] = []
        offset = 0.0
        while offset < duration:
            output = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name)
            command = [
                "ffmpeg", "-y",
                "-ss", str(offset),
                "-i", str(path),
                "-t", str(chunk_secs),
                "-vn", "-ac", "1", "-ar", "16000",
                "-b:a", self.settings.audio_transcode_bitrate,
                str(output),
            ]
            try:
                subprocess.run(command, check=True, capture_output=True, text=True)
            except FileNotFoundError as exc:
                for p, _ in chunks:
                    p.unlink(missing_ok=True)
                raise RuntimeError("ffmpeg is required to split oversized audio") from exc
            except subprocess.CalledProcessError as exc:
                output.unlink(missing_ok=True)
                for p, _ in chunks:
                    p.unlink(missing_ok=True)
                message = exc.stderr.strip() or exc.stdout.strip() or "ffmpeg audio split failed"
                raise RuntimeError(message) from exc
            chunks.append((output, offset))
            offset += chunk_secs
        return chunks

    def _openai_realtime_transcribe(
        self,
        path: Path,
        on_partial: Callable[[TranscriptResult], None] | None = None,
    ) -> TranscriptResult:
        if not self.settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required when TRANSCRIBE_PROVIDER=openai_realtime")

        from websockets.sync.client import connect

        sample_rate = self.settings.realtime_sample_rate
        chunk_seconds = self.settings.realtime_chunk_seconds
        chunk_bytes = sample_rate * 2 * chunk_seconds
        duration = audio_duration_seconds(path) or self._probe_duration_seconds(path)
        if self._should_use_channel_diarization(path):
            return self._openai_realtime_transcribe_channels(path, duration, on_partial)

        pcm_path = self._convert_to_pcm16(path, sample_rate=sample_rate)
        segments: list[TranscriptSegment] = []

        headers = {
            "Authorization": f"Bearer {self.settings.openai_api_key}",
        }
        uri = "wss://api.openai.com/v1/realtime?intent=transcription"
        try:
            with connect(
                uri,
                additional_headers=headers,
                open_timeout=30,
                ping_interval=20,
                ping_timeout=20,
                max_size=None,
            ) as ws:
                self._send_realtime_session_update(ws)
                self._drain_realtime_setup_events(ws)

                with pcm_path.open("rb") as pcm:
                    index = 0
                    while True:
                        chunk = pcm.read(chunk_bytes)
                        if not chunk:
                            break
                        start = index * chunk_seconds
                        end = start + (len(chunk) / (sample_rate * 2))
                        transcript = self._transcribe_realtime_chunk(ws, chunk)
                        if transcript:
                            segments.append(
                                TranscriptSegment(
                                    start=round(start, 3),
                                    end=round(end, 3),
                                    speaker=self._local_speaker(index),
                                    text=transcript,
                                )
                            )
                            update_every = self.settings.realtime_partial_update_every_segments
                            should_publish_partial = bool(update_every) and len(segments) % update_every == 0
                            if on_partial and should_publish_partial:
                                on_partial(
                                    TranscriptResult(
                                        language=self._transcribe_language() or "auto",
                                        duration_seconds=duration,
                                        segments=segments.copy(),
                                        full_text="\n".join(
                                            f"{item.speaker}: {item.text}" for item in segments
                                        ),
                                    )
                                )
                        index += 1
        finally:
            pcm_path.unlink(missing_ok=True)

        if not segments:
            raise RuntimeError("Realtime transcription returned no transcript segments")

        return TranscriptResult(
            language=self._transcribe_language() or "auto",
            duration_seconds=duration,
            segments=segments,
            full_text="\n".join(f"{item.speaker}: {item.text}" for item in segments),
        )

    def _send_realtime_session_update(self, ws) -> None:
        session = {
            "type": "transcription",
            "audio": {
                "input": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": self.settings.realtime_sample_rate,
                    },
                    "transcription": {
                        "model": self.settings.openai_realtime_transcribe_model,
                        "delay": self.settings.openai_realtime_transcription_delay,
                    },
                    "turn_detection": None,
                }
            },
        }
        language = self._transcribe_language()
        if language:
            session["audio"]["input"]["transcription"]["language"] = language

        ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": session,
                }
            )
        )

    def _drain_realtime_setup_events(self, ws) -> None:
        for _ in range(5):
            try:
                event = json.loads(ws.recv(timeout=2))
            except TimeoutError:
                return
            event_type = event.get("type")
            if event_type == "error":
                raise RuntimeError(event.get("error", {}).get("message") or str(event))
            if event_type in {"transcription_session.updated", "session.updated"}:
                return

    def _transcribe_realtime_chunk(self, ws, chunk: bytes) -> str:
        ws.send(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(chunk).decode("ascii"),
                }
            )
        )
        ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        deltas: list[str] = []
        while True:
            try:
                event = json.loads(ws.recv(timeout=self.settings.realtime_chunk_timeout_seconds))
            except TimeoutError as exc:
                raise RuntimeError("Timed out waiting for realtime transcription") from exc

            event_type = event.get("type")
            if event_type == "error":
                raise RuntimeError(event.get("error", {}).get("message") or str(event))
            if event_type in {
                "conversation.item.input_audio_transcription.delta",
                "input_audio_transcription.delta",
                "session.input_transcript.delta",
            }:
                delta = event.get("delta") or event.get("text") or ""
                if delta:
                    deltas.append(str(delta))
            if event_type in {
                "conversation.item.input_audio_transcription.completed",
                "input_audio_transcription.completed",
                "session.input_transcript.completed",
            }:
                transcript = event.get("transcript") or event.get("text") or "".join(deltas)
                return str(transcript).strip()
            if event_type == "conversation.item.input_audio_transcription.failed":
                error = event.get("error") or {}
                raise RuntimeError(error.get("message") or str(event))

    def _openai_realtime_transcribe_channels(
        self,
        path: Path,
        duration: float | None,
        on_partial: Callable[[TranscriptResult], None] | None = None,
    ) -> TranscriptResult:
        sample_rate = self.settings.realtime_sample_rate
        normalized_path = self._normalize_to_wav(path, sample_rate=sample_rate)
        channel_count = min(self._source_channel_count(normalized_path) or 0, 2)
        pcm_paths: list[tuple[int, Path]] = []
        try:
            for channel in range(channel_count):
                pcm_paths.append((channel, self._convert_to_pcm16(normalized_path, sample_rate=sample_rate, channel=channel)))

            chunk_bytes = sample_rate * 2 * self.settings.realtime_chunk_seconds

            def _transcribe_channel(args: tuple[int, Path]) -> tuple[int, list[TranscriptSegment]]:
                channel, pcm_path = args
                return channel, self._openai_realtime_transcribe_pcm(
                    pcm_path,
                    duration=duration,
                    chunk_bytes=chunk_bytes,
                    speaker=self._channel_speaker(channel),
                )

            with ThreadPoolExecutor(max_workers=channel_count) as pool:
                results = dict(pool.map(_transcribe_channel, pcm_paths))

            segments: list[TranscriptSegment] = []
            for channel in sorted(results):
                segments.extend(results[channel])
            segments.sort(key=lambda item: (item.start, item.end, item.speaker))

            if on_partial and segments:
                on_partial(self._transcript_result(segments, duration))
        finally:
            for _, pcm_path in pcm_paths:
                pcm_path.unlink(missing_ok=True)
            normalized_path.unlink(missing_ok=True)

        if not segments:
            raise RuntimeError("Realtime transcription returned no transcript segments")
        return self._transcript_result(segments, duration)

    def _openai_realtime_transcribe_pcm(
        self,
        pcm_path: Path,
        duration: float | None,
        chunk_bytes: int,
        speaker: str | None = None,
    ) -> list[TranscriptSegment]:
        from websockets.sync.client import connect

        sample_rate = self.settings.realtime_sample_rate
        chunk_seconds = self.settings.realtime_chunk_seconds
        segments: list[TranscriptSegment] = []
        headers = {"Authorization": f"Bearer {self.settings.openai_api_key}"}
        uri = "wss://api.openai.com/v1/realtime?intent=transcription"
        with connect(
            uri,
            additional_headers=headers,
            open_timeout=30,
            ping_interval=20,
            ping_timeout=20,
            max_size=None,
        ) as ws:
            self._send_realtime_session_update(ws)
            self._drain_realtime_setup_events(ws)
            with pcm_path.open("rb") as pcm:
                index = 0
                while True:
                    chunk = pcm.read(chunk_bytes)
                    if not chunk:
                        break
                    start = index * chunk_seconds
                    end = start + (len(chunk) / (sample_rate * 2))
                    transcript = self._transcribe_realtime_chunk(ws, chunk)
                    if transcript:
                        segments.append(
                            TranscriptSegment(
                                start=round(start, 3),
                                end=round(end, 3),
                                speaker=speaker or self._local_speaker(index),
                                text=transcript,
                            )
                        )
                    index += 1
        return segments

    def _transcript_result(self, segments: list[TranscriptSegment], duration: float | None) -> TranscriptResult:
        return TranscriptResult(
            language=self._transcribe_language() or "auto",
            duration_seconds=duration,
            segments=segments.copy(),
            full_text="\n".join(f"{item.speaker}: {item.text}" for item in segments),
        )

    def _transcribe_language(self) -> str | None:
        language = (self.settings.openai_transcribe_language or "").strip().lower()
        if language in {"", "auto", "multilingual", "detect", "none"}:
            return None
        return language

    def _should_use_channel_diarization(self, path: Path) -> bool:
        strategy = self.settings.local_diarization_strategy.lower()
        if strategy not in {"channel", "channels"}:
            return False
        return (self._source_channel_count(path) or 0) >= 2

    def _source_channel_count(self, path: Path) -> int | None:
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "a:0",
                    "-show_entries",
                    "stream=channels",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            return int(result.stdout.strip())
        except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
            return None

    def _channel_speaker(self, channel: int) -> str:
        return f"Channel {channel + 1}"

    def _normalize_to_wav(self, path: Path, sample_rate: int) -> Path:
        output = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name)
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(path),
            "-vn",
            "-ar",
            str(sample_rate),
            "-acodec",
            "pcm_s16le",
            str(output),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise RuntimeError("ffmpeg is required to normalize audio for diarization") from exc
        except subprocess.CalledProcessError as exc:
            output.unlink(missing_ok=True)
            message = exc.stderr.strip() or exc.stdout.strip() or "ffmpeg WAV normalization failed"
            raise RuntimeError(message) from exc
        return output

    def _convert_to_pcm16(self, path: Path, sample_rate: int, channel: int | None = None) -> Path:
        output = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".pcm").name)
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(path),
            "-vn",
            "-ar",
            str(sample_rate),
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            str(output),
        ]
        if channel is None:
            command[5:5] = ["-ac", "1"]
        else:
            command[5:5] = ["-af", f"pan=mono|c0=c{channel}"]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise RuntimeError("ffmpeg is required for realtime transcription input conversion") from exc
        except subprocess.CalledProcessError as exc:
            output.unlink(missing_ok=True)
            message = exc.stderr.strip() or exc.stdout.strip() or "ffmpeg PCM conversion failed"
            raise RuntimeError(message) from exc
        return output

    def _probe_duration_seconds(self, path: Path) -> float | None:
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            return round(float(result.stdout.strip()), 3)
        except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
            return None

    def _prepare_upload_file(self, path: Path, stereo: bool = False) -> tuple[Path, Path | None]:
        suffix = path.suffix.lower()
        should_transcode = suffix not in OPENAI_DIRECT_AUDIO_EXTENSIONS
        if suffix == ".wave":
            should_transcode = False

        # Force-transcode native formats that are already over the upload limit
        if not should_transcode and path.stat().st_size > WHISPER_SIZE_LIMIT:
            should_transcode = True

        if not should_transcode:
            return path, None

        output = self._ffmpeg_to_mp3(path, self.settings.audio_transcode_bitrate, stereo=stereo)

        # If still over limit, re-transcode at a low speech-quality bitrate (16k handles ~3h files)
        if output.stat().st_size > WHISPER_SIZE_LIMIT:
            compressed = self._ffmpeg_to_mp3(path, "16k", stereo=stereo, existing=output)
            output = compressed

        return output, output

    def _ffmpeg_to_mp3(
        self, path: Path, bitrate: str, stereo: bool = False, existing: Path | None = None
    ) -> Path:
        if existing:
            existing.unlink(missing_ok=True)
        output = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name)
        channels = "2" if stereo else "1"
        command = [
            "ffmpeg", "-y", "-i", str(path),
            "-vn", "-ac", channels, "-ar", "16000",
            "-b:a", bitrate, str(output),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except FileNotFoundError as exc:
            output.unlink(missing_ok=True)
            raise RuntimeError("ffmpeg is required to process this audio format") from exc
        except subprocess.CalledProcessError as exc:
            output.unlink(missing_ok=True)
            message = exc.stderr.strip() or exc.stdout.strip() or "ffmpeg conversion failed"
            raise RuntimeError(message) from exc
        return output

    def _parse_openai_response(self, payload: dict[str, Any], path: Path) -> TranscriptResult:
        raw_segments = payload.get("segments") or payload.get("speaker_segments") or []
        duration = payload.get("duration") or audio_duration_seconds(path)
        segments: list[TranscriptSegment] = []
        for index, item in enumerate(raw_segments):
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            speaker = item.get("speaker") or item.get("speaker_label") or self._local_speaker(index)
            segments.append(
                TranscriptSegment(
                    start=float(item.get("start") or item.get("start_time") or 0),
                    end=float(item.get("end") or item.get("end_time") or 0),
                    speaker=str(speaker),
                    text=text,
                )
            )

        if not segments:
            text = str(payload.get("text") or "").strip()
            segments = [TranscriptSegment(start=0, end=float(duration or 0), speaker="Unknown", text=text)]
        elif self.settings.local_diarization_strategy.lower() != "none":
            segments = self._apply_local_diarization(segments)

        return TranscriptResult(
            language=payload.get("language"),
            duration_seconds=float(duration) if duration is not None else None,
            segments=segments,
            full_text="\n".join(f"{item.speaker}: {item.text}" for item in segments),
        )

    def _local_speaker(self, index: int) -> str:
        if self.settings.local_diarization_strategy.lower() == "none":
            return "Unknown"
        return "Speaker A" if index % 2 == 0 else "Speaker B"

    def _apply_local_diarization(self, segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
        if any(item.speaker not in {"Unknown", "Speaker A", "Speaker B"} for item in segments):
            return segments
        for index, segment in enumerate(segments):
            segment.speaker = self._local_speaker(index)
        return segments
