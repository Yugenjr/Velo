import enum
import time
import threading
import asyncio
import queue
from typing import Optional, Union, List, Dict, Any, AsyncIterator
from dataclasses import dataclass, field

from .core import Stream, RtpReceiver, Frame
from .scheduler import AIScheduler, SchedulerClosedError
from .audio_scheduler import AudioScheduler
from .fusion import TemporalFusion, MultimodalContext
from .asr import BaseASRAdapter, Transcript
from .vlm import BaseVLMAdapter, BaseMultimodalAdapter, VLMResponse
from .metrics import RuntimeMetrics, StreamMetrics
from .pipeline import PipelineState


@dataclass
class MultiStreamResult:
    """
    Inference result from a multi-stream pipeline.
    Supports attribute access and tuple unpacking: (stream_id, response).
    """
    stream_id: str
    response: Any
    timestamp: float = field(default_factory=time.time)

    def __iter__(self):
        yield self.stream_id
        yield self.response

    def __getitem__(self, index: int):
        if index == 0:
            return self.stream_id
        elif index == 1:
            return self.response
        raise IndexError("MultiStreamResult index out of range (0 or 1)")


class StreamSession:
    """
    Encapsulates all isolated resources and execution state for a single stream.
    
    A failure within one StreamSession will not compromise other sessions.
    """
    def __init__(
        self,
        stream_id: str,
        stream: Union[Stream, RtpReceiver],
        scheduler: AIScheduler,
        metrics: StreamMetrics,
        audio_scheduler: Optional[AudioScheduler] = None,
        fusion: Optional[TemporalFusion] = None,
        vad: Optional[Any] = None,
        asr: Optional[BaseASRAdapter] = None,
        context_window_s: float = 1.5,
        max_candidate_queue: int = 10,
    ):
        self.stream_id = stream_id
        self.stream = stream
        self.scheduler = scheduler
        self.metrics = metrics
        self.audio_scheduler = audio_scheduler
        self.fusion = fusion
        self.vad = vad
        self.asr = asr
        self.context_window_s = context_window_s

        self.state = PipelineState.CREATED
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()

        # Isolated candidate queue (bounded to prevent runaway memory)
        self.candidate_queue: queue.Queue = queue.Queue(maxsize=max_candidate_queue)

        # Worker threads
        self._video_ingest_thread: Optional[threading.Thread] = None
        self._candidate_thread: Optional[threading.Thread] = None
        self._audio_ingest_thread: Optional[threading.Thread] = None
        self._audio_process_thread: Optional[threading.Thread] = None

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._last_error: Optional[Exception] = None

    def start(self, loop: asyncio.AbstractEventLoop):
        """Start isolated ingestion and processing workers for this session."""
        with self._state_lock:
            if self.state != PipelineState.CREATED:
                raise RuntimeError(f"Session {self.stream_id} cannot start from state {self.state.name}")
            self.state = PipelineState.STARTING

        self._loop = loop
        self._stop_event.clear()

        self._video_ingest_thread = threading.Thread(
            target=self._video_ingest_worker,
            name=f"VeloVideoIngest-{self.stream_id}",
            daemon=True,
        )
        self._candidate_thread = threading.Thread(
            target=self._candidate_worker,
            name=f"VeloCandidate-{self.stream_id}",
            daemon=True,
        )

        self._video_ingest_thread.start()
        self._candidate_thread.start()

        if self.audio_scheduler is not None:
            if hasattr(self.stream, "next_audio"):
                self._audio_ingest_thread = threading.Thread(
                    target=self._audio_ingest_worker,
                    name=f"VeloAudioIngest-{self.stream_id}",
                    daemon=True,
                )
                self._audio_ingest_thread.start()

            self._audio_process_thread = threading.Thread(
                target=self._audio_process_worker,
                name=f"VeloAudioProcess-{self.stream_id}",
                daemon=True,
            )
            self._audio_process_thread.start()

        with self._state_lock:
            self.state = PipelineState.RUNNING
        self.metrics.update_state(pipeline_state=self.state.name)

    def stop(self):
        """Gracefully stop this session without affecting adjacent streams."""
        with self._state_lock:
            if self.state in (PipelineState.STOPPING, PipelineState.STOPPED, PipelineState.FAILED):
                return
            self.state = PipelineState.STOPPING

        self._stop_event.set()

        try:
            self.stream.close()
        except Exception:
            pass

        try:
            self.scheduler.close()
        except Exception:
            pass

        if self.audio_scheduler is not None:
            try:
                self.audio_scheduler.close()
            except Exception:
                pass

        self._join_threads()

        with self._state_lock:
            if self.state != PipelineState.FAILED:
                self.state = PipelineState.STOPPED
        self.metrics.update_state(pipeline_state=self.state.name)

    def fail(self, exc: Exception):
        """Isolate failure to this stream session."""
        with self._state_lock:
            if self.state in (PipelineState.STOPPING, PipelineState.STOPPED, PipelineState.FAILED):
                return
            self.state = PipelineState.FAILED
            self._last_error = exc

        self._stop_event.set()
        self.metrics.record_worker_error()
        self.metrics.update_state(pipeline_state=self.state.name)

        try:
            self.stream.close()
        except Exception:
            pass

        try:
            self.scheduler.close()
        except Exception:
            pass

        if self.audio_scheduler is not None:
            try:
                self.audio_scheduler.close()
            except Exception:
                pass

    def _join_threads(self):
        for t in (self._video_ingest_thread, self._candidate_thread, self._audio_ingest_thread, self._audio_process_thread):
            if t and t.is_alive():
                t.join(timeout=3.0)

    def _video_ingest_worker(self):
        while not self._stop_event.is_set():
            try:
                frame = self.stream.next()
                if frame is None:
                    continue
                frame.arrival_time = time.time()

                if self.fusion is not None:
                    ts = getattr(frame, "timestamp", 0.0)
                    self.fusion.add_video(frame, ts)

                self.scheduler.submit(frame)
            except Exception as e:
                if self._stop_event.is_set():
                    break
                if "StreamClosed" in type(e).__name__ or "closed" in str(e).lower() or "eos" in str(e).lower():
                    try:
                        self.scheduler.close()
                    except Exception:
                        pass
                    break
                self.fail(e)
                break

    def _candidate_worker(self):
        """Paces and acquires candidate frames from the scheduler into the candidate queue."""
        try:
            while not self._stop_event.is_set():
                try:
                    frame = self.scheduler.acquire()
                    if frame is None:
                        continue

                    # Put into candidate queue with backpressure protection
                    while not self._stop_event.is_set():
                        try:
                            self.candidate_queue.put(frame, timeout=0.05)
                            break
                        except queue.Full:
                            # Drop oldest candidate in queue to maintain freshness
                            try:
                                _old = self.candidate_queue.get_nowait()
                                self.scheduler.release()
                            except Exception:
                                pass
                except SchedulerClosedError:
                    break
                except Exception as e:
                    if self._stop_event.is_set():
                        break
                    self.fail(e)
                    break
        finally:
            with self._state_lock:
                if self.state == PipelineState.RUNNING:
                    self.state = PipelineState.STOPPED
            self.metrics.update_state(pipeline_state=self.state.name)

    def _audio_ingest_worker(self):
        while not self._stop_event.is_set():
            try:
                chunk = self.stream.next_audio()
                if chunk is None:
                    continue
                if self.audio_scheduler is not None:
                    self.audio_scheduler.submit(chunk)
            except Exception as e:
                if self._stop_event.is_set():
                    break
                if "StreamClosed" in type(e).__name__ or "closed" in str(e).lower() or "eos" in str(e).lower():
                    if self.audio_scheduler is not None:
                        try:
                            self.audio_scheduler.close()
                        except Exception:
                            pass
                    break
                self.fail(e)
                break

    def _audio_process_worker(self):
        while not self._stop_event.is_set():
            try:
                if self.audio_scheduler is None:
                    break
                chunk = self.audio_scheduler.acquire()
            except SchedulerClosedError:
                break
            except Exception as e:
                self.fail(e)
                break

            try:
                is_speech = True
                if self.vad is not None:
                    if hasattr(self.vad, "analyze"):
                        is_speech = self.vad.analyze(chunk).is_speech
                    elif hasattr(self.vad, "is_speech"):
                        is_speech = self.vad.is_speech(chunk)
                    elif callable(self.vad):
                        res = self.vad(chunk)
                        is_speech = res.is_speech if hasattr(res, "is_speech") else bool(res)

                if is_speech:
                    if self.asr is not None:
                        transcript = self.asr.transcribe(chunk)
                        if self.fusion is not None and transcript is not None:
                            duration = max(0.0, getattr(transcript, "end_timestamp", 0.0) - getattr(transcript, "start_timestamp", 0.0))
                            ts = getattr(transcript, "start_timestamp", getattr(chunk, "timestamp", 0.0))
                            self.fusion.add_audio(payload=transcript, timestamp=ts, duration=duration)
                    elif self.fusion is not None:
                        self.fusion.add_audio(payload=chunk, timestamp=chunk.timestamp, duration=chunk.duration)
            except Exception as e:
                self.fail(e)
                break

    def get_candidate(self, timeout: float = 0.005) -> Optional[Frame]:
        """Poll for an available candidate frame from this session."""
        try:
            return self.candidate_queue.get(block=True, timeout=timeout)
        except queue.Empty:
            return None

    def is_drained(self) -> bool:
        """True if session is stopped/failed and candidate queue is empty."""
        with self._state_lock:
            not_running = self.state in (PipelineState.STOPPED, PipelineState.FAILED)
        return not_running and self.candidate_queue.empty()

    def stats(self) -> Dict[str, Any]:
        """Return comprehensive metrics snapshot for this session."""
        v_stats = self.scheduler.stats() if hasattr(self.scheduler, "stats") else {}
        a_stats = self.audio_scheduler.stats() if self.audio_scheduler and hasattr(self.audio_scheduler, "stats") else {}
        f_stats = self.fusion.stats() if self.fusion and hasattr(self.fusion, "stats") else {}
        q_depth = self.candidate_queue.qsize()

        return self.metrics.snapshot(
            pipeline_state=self.state.name,
            video_scheduler_stats=v_stats,
            audio_scheduler_stats=a_stats,
            fusion_stats=f_stats,
            inference_queue_depth=q_depth,
        )


class MultiStreamPipeline:
    """
    Asynchronous multi-stream orchestration runtime.
    
    Coordinates N independent media streams sharing a single GPU VLM instance,
    providing fair scheduling, bounded queues, isolated failure domains,
    and hierarchical observability.
    """
    def __init__(
        self,
        vlm: BaseVLMAdapter,
        max_streams: Optional[int] = None,
        gpu_memory_headroom_mb: float = 100.0,
    ):
        self.vlm = vlm
        self.max_streams = max_streams
        self.gpu_memory_headroom_mb = gpu_memory_headroom_mb

        self._sessions: Dict[str, StreamSession] = {}
        self._sessions_lock = threading.Lock()
        self._metrics = RuntimeMetrics(default_stream_id=None)

        self.state = PipelineState.CREATED
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._result_queue: Optional[asyncio.Queue] = None
        self._coordinator_thread: Optional[threading.Thread] = None

        self._current_prompt: str = ""
        self._consumer_active = False

    @property
    def active_stream_count(self) -> int:
        with self._sessions_lock:
            return len(self._sessions)

    @property
    def active_stream_ids(self) -> List[str]:
        with self._sessions_lock:
            return list(self._sessions.keys())

    def add_stream(
        self,
        stream_id: str,
        stream: Union[Stream, RtpReceiver],
        scheduler: Optional[AIScheduler] = None,
        audio_scheduler: Optional[AudioScheduler] = None,
        fusion: Optional[TemporalFusion] = None,
        vad: Optional[Any] = None,
        asr: Optional[BaseASRAdapter] = None,
        target_fps: float = 30.0,
        context_window_s: float = 1.5,
        max_candidate_queue: int = 10,
    ) -> StreamSession:
        """
        Dynamically register a new stream session into the multi-stream runtime.
        
        Validates GPU memory budget and stream limits before allocating.
        """
        with self._sessions_lock:
            if stream_id in self._sessions:
                raise ValueError(f"Stream '{stream_id}' is already registered in multi-stream pipeline.")

            if self.max_streams is not None and len(self._sessions) >= self.max_streams:
                raise RuntimeError(
                    f"Cannot add stream '{stream_id}': maximum stream limit ({self.max_streams}) reached."
                )

            # Check GPU memory budget
            gpu_info = RuntimeMetrics.get_gpu_metrics()
            if gpu_info.get("available", False):
                free_mb = gpu_info.get("free_memory_mb", 0.0)
                if free_mb < self.gpu_memory_headroom_mb:
                    raise RuntimeError(
                        f"Cannot add stream '{stream_id}': GPU memory headroom insufficient "
                        f"({free_mb:.1f} MB free < {self.gpu_memory_headroom_mb:.1f} MB required)."
                    )

            # Create default scheduler if not provided
            if scheduler is None:
                scheduler = AIScheduler(target_fps=target_fps)

            # Register stream metrics
            stream_metrics = self._metrics.register_stream(stream_id)

            session = StreamSession(
                stream_id=stream_id,
                stream=stream,
                scheduler=scheduler,
                metrics=stream_metrics,
                audio_scheduler=audio_scheduler,
                fusion=fusion,
                vad=vad,
                asr=asr,
                context_window_s=context_window_s,
                max_candidate_queue=max_candidate_queue,
            )
            self._sessions[stream_id] = session

        # If pipeline is already RUNNING, start the new session immediately
        with self._state_lock:
            if self.state == PipelineState.RUNNING and self._loop is not None:
                session.start(self._loop)

        return session

    def remove_stream(self, stream_id: str):
        """Gracefully stop and unregister a stream session."""
        with self._sessions_lock:
            session = self._sessions.pop(stream_id, None)

        if session is not None:
            session.stop()
            self._metrics.unregister_stream(stream_id)

    def get_stream(self, stream_id: str) -> Optional[StreamSession]:
        with self._sessions_lock:
            return self._sessions.get(stream_id)

    def has_stream(self, stream_id: str) -> bool:
        with self._sessions_lock:
            return stream_id in self._sessions

    async def start(self):
        """Start the multi-stream coordinator and all registered sessions."""
        with self._state_lock:
            if self.state != PipelineState.CREATED:
                raise RuntimeError(f"Cannot start MultiStreamPipeline from state {self.state.name}")
            self.state = PipelineState.STARTING

        self._loop = asyncio.get_running_loop()
        self._result_queue = asyncio.Queue()
        self._stop_event.clear()

        # Start all existing sessions
        with self._sessions_lock:
            for session in self._sessions.values():
                session.start(self._loop)

        # Start coordinator inference worker
        self._coordinator_thread = threading.Thread(
            target=self._coordinator_worker,
            name="VeloMultiStreamCoordinator",
            daemon=True,
        )
        self._coordinator_thread.start()

        with self._state_lock:
            self.state = PipelineState.RUNNING

    async def stop(self):
        """Gracefully stop all sessions and the multi-stream coordinator."""
        with self._state_lock:
            if self.state in (PipelineState.STOPPING, PipelineState.STOPPED, PipelineState.FAILED):
                return
            self.state = PipelineState.STOPPING

        self._stop_event.set()

        # Stop all sessions
        with self._sessions_lock:
            for session in list(self._sessions.values()):
                session.stop()

        if self._loop and self._result_queue:
            self._loop.call_soon_threadsafe(self._result_queue.put_nowait, StopIteration)

        if self._coordinator_thread and self._coordinator_thread.is_alive():
            if self._loop:
                await self._loop.run_in_executor(None, self._coordinator_thread.join, 5.0)

        with self._state_lock:
            if self.state != PipelineState.FAILED:
                self.state = PipelineState.STOPPED

    def _coordinator_worker(self):
        """
        Fair round-robin inference coordinator.
        
        Iterates over active stream sessions, dispatching ready candidates to the shared VLM.
        Prevents high-FPS streams from starving low-FPS streams.
        """
        while not self._stop_event.is_set():
            with self._sessions_lock:
                sessions_list = list(self._sessions.values())

            if not sessions_list:
                time.sleep(0.01)
                continue

            active_or_pending = False
            processed_any = False

            for session in sessions_list:
                if self._stop_event.is_set():
                    break

                if session.state == PipelineState.FAILED:
                    continue

                if not session.is_drained():
                    active_or_pending = True

                candidate = session.get_candidate(timeout=0.001)
                if candidate is None:
                    continue

                processed_any = True
                self._dispatch_inference(session, candidate)

            # If all registered sessions are drained and no candidates were processed, signal EOF
            if not active_or_pending and not processed_any:
                break

            if not processed_any:
                time.sleep(0.005)

        # Signal termination to consumers
        if self._loop and self._result_queue:
            self._loop.call_soon_threadsafe(self._result_queue.put_nowait, StopIteration)

    def _dispatch_inference(self, session: StreamSession, frame: Frame):
        """Dispatch a single candidate frame to the shared VLM model safely."""
        t_infer_start = time.time()
        try:
            # Retrieve multimodal context if available
            ctx = None
            transcripts = []
            if session.fusion is not None:
                t_fuse_start = time.time()
                ctx = session.fusion.context(
                    frame.timestamp,
                    window=session.context_window_s,
                )
                t_fuse_end = time.time()

                skew_ms = None
                if hasattr(session.fusion, "_latest_skew_ms"):
                    skew_ms = session.fusion._latest_skew_ms
                elif ctx and hasattr(ctx, "audio_video_skew_ms"):
                    skew_ms = ctx.audio_video_skew_ms

                session.metrics.record_fusion(
                    lookup_latency_ms=(t_fuse_end - t_fuse_start) * 1000.0,
                    skew_ms=skew_ms,
                )

                if ctx and hasattr(ctx, "audio"):
                    for obs in ctx.audio:
                        if isinstance(obs.payload, Transcript):
                            transcripts.append(obs.payload)
                        elif hasattr(obs.payload, "text"):
                            transcripts.append(obs.payload)

            # Run inference via standard BaseVLMAdapter
            if hasattr(self.vlm, "generate"):
                import inspect
                try:
                    sig = inspect.signature(self.vlm.generate)
                    gen_kwargs = {}
                    if "context" in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                        gen_kwargs["context"] = ctx
                    if "transcripts" in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                        gen_kwargs["transcripts"] = transcripts
                    response = self.vlm.generate(frame, self._current_prompt, **gen_kwargs)
                except TypeError:
                    response = self.vlm.generate(frame, self._current_prompt)
            elif hasattr(self.vlm, "infer"):
                response = self.vlm.infer(frame, context=ctx)
            else:
                response = VLMResponse(text="", latency_ms=0.0)

            t_infer_end = time.time()
            total_lat_ms = (t_infer_end - t_infer_start) * 1000.0
            prep_lat_ms = getattr(response, "preprocessing_latency_ms", 0.0)
            infer_lat_ms = getattr(response, "latency_ms", total_lat_ms)

            arrival_t = getattr(frame, "arrival_time", None) or t_infer_start
            e2e_lat_ms = max(total_lat_ms, (t_infer_end - arrival_t) * 1000.0)

            # Record per-stream and global metrics
            session.metrics.record_inference(
                inference_latency_ms=infer_lat_ms,
                preprocessing_latency_ms=prep_lat_ms,
                end_to_end_latency_ms=e2e_lat_ms,
            )

            # Feedback latency to adaptive scheduler
            if hasattr(session.scheduler, "record_inference"):
                if total_lat_ms > 0:
                    session.scheduler.record_inference(total_lat_ms)

            # Yield result to consumer
            result = MultiStreamResult(
                stream_id=session.stream_id,
                response=response,
                timestamp=t_infer_end,
            )
            if self._loop and self._result_queue:
                self._loop.call_soon_threadsafe(self._result_queue.put_nowait, result)

        except Exception as e:
            session.metrics.record_inference_error()
            session.scheduler.release()
            # Fault isolation: stream failure does not crash coordinator or other streams
            session.fail(e)
            return

        session.scheduler.release()

    async def run_inference(self, prompt: str = "") -> AsyncIterator[MultiStreamResult]:
        """
        Asynchronous generator yielding MultiStreamResults from all active streams.
        """
        if self.state != PipelineState.RUNNING:
            raise RuntimeError(f"MultiStreamPipeline must be RUNNING, current state is {self.state.name}")

        if self._consumer_active:
            raise RuntimeError("Only one active multi-stream inference consumer is permitted.")

        self._current_prompt = prompt
        self._consumer_active = True

        try:
            while True:
                result = await self._result_queue.get()
                if result is StopIteration:
                    break
                if isinstance(result, Exception):
                    raise result
                yield result
        finally:
            self._consumer_active = False

    def metrics(self) -> RuntimeMetrics:
        """
        Return the hierarchical RuntimeMetrics object supporting:
        - metrics.stream(stream_id)
        - metrics.global()
        """
        with self._sessions_lock:
            for sid, sess in self._sessions.items():
                sess.stats()
        return self._metrics

    def get_metrics(self) -> Dict[str, Any]:
        """Return the global metrics snapshot dict."""
        return self.metrics().global_snapshot()
