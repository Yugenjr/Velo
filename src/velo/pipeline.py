import enum
import threading
import asyncio
import time
from typing import Optional, Union, List, Any
from .scheduler import AIScheduler, SchedulerClosedError
from .vlm import BaseVLMAdapter, BaseMultimodalAdapter, VLMResponse
from .exceptions import StreamClosedError, VeloError
from .fusion import TemporalFusion, MultimodalContext
from .asr import BaseASRAdapter, Transcript
from .audio_scheduler import AudioScheduler
from .metrics import RuntimeMetrics, RollingStats

class PipelineState(enum.Enum):
    CREATED = "CREATED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    FAILED = "FAILED"

class AIPipeline:
    """
    A high-level asynchronous orchestration layer for Velo.
    
    Composes Stream (video and/or audio), AIScheduler, AudioScheduler,
    TemporalFusion, VAD, ASR, and VLMAdapter/BaseMultimodalAdapter, managing their
    blocking OS worker threads safely and exposing a clean asynchronous generator for agents.
    """
    def __init__(
        self,
        stream,
        scheduler: AIScheduler,
        vlm: BaseVLMAdapter,
        audio_scheduler: Optional[AudioScheduler] = None,
        fusion: Optional[TemporalFusion] = None,
        vad: Optional[Any] = None,
        asr: Optional[BaseASRAdapter] = None,
        context_window_s: float = 1.5,
    ):
        self.stream = stream
        self.scheduler = scheduler
        self.vlm = vlm
        self.audio_scheduler = audio_scheduler
        self.fusion = fusion
        self.vad = vad
        self.asr = asr
        self.context_window_s = context_window_s
        
        self.state = PipelineState.CREATED
        self._state_lock = threading.Lock()
        self._metrics = RuntimeMetrics()
        
        self._ingest_thread: Optional[threading.Thread] = None
        self._audio_ingest_thread: Optional[threading.Thread] = None
        self._audio_process_thread: Optional[threading.Thread] = None
        self._inference_thread: Optional[threading.Thread] = None
        
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._result_queue: Optional[asyncio.Queue] = None
        
        self._stop_event = threading.Event()
        
        # Concurrency control for inference
        self._inference_consumer_active = False
        self._current_prompt: str = ""

    async def start(self):
        """Start the background ingestion and inference threads."""
        with self._state_lock:
            if self.state != PipelineState.CREATED:
                raise RuntimeError(f"Cannot start pipeline from state {self.state.name}")
            self.state = PipelineState.STARTING
            
        self._loop = asyncio.get_running_loop()
        self._result_queue = asyncio.Queue()
        
        self._ingest_thread = threading.Thread(target=self._ingest_worker, name="VeloVideoIngest", daemon=True)
        self._inference_thread = threading.Thread(target=self._inference_worker, name="VeloInference", daemon=True)
        
        self._ingest_thread.start()
        self._inference_thread.start()
        
        if self.audio_scheduler is not None:
            if hasattr(self.stream, "next_audio"):
                self._audio_ingest_thread = threading.Thread(target=self._audio_ingest_worker, name="VeloAudioIngest", daemon=True)
                self._audio_ingest_thread.start()
            self._audio_process_thread = threading.Thread(target=self._audio_process_worker, name="VeloAudioProcess", daemon=True)
            self._audio_process_thread.start()
        
        with self._state_lock:
            self.state = PipelineState.RUNNING

    async def stop(self):
        """Gracefully stop the pipeline and all background threads."""
        with self._state_lock:
            if self.state in (PipelineState.STOPPING, PipelineState.STOPPED, PipelineState.FAILED):
                return
            self.state = PipelineState.STOPPING
            
        self._stop_event.set()
        
        # Cascading shutdown triggers
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
            
        if self._loop and self._result_queue:
            self._loop.call_soon_threadsafe(self._result_queue.put_nowait, StopIteration)

        if self._loop:
            await self._loop.run_in_executor(None, self._join_threads)
            
        with self._state_lock:
            if self.state != PipelineState.FAILED:
                self.state = PipelineState.STOPPED

    def _join_threads(self):
        for t in (self._ingest_thread, self._audio_ingest_thread, self._audio_process_thread, self._inference_thread):
            if t and t.is_alive():
                t.join(timeout=5.0)

    def _fail(self, exc: Exception):
        """Transition to FAILED and initiate cascading shutdown."""
        with self._state_lock:
            if self.state in (PipelineState.STOPPING, PipelineState.STOPPED, PipelineState.FAILED):
                return
            self.state = PipelineState.FAILED
            
        self._stop_event.set()
        self._metrics.record_worker_error()
        
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
            
        if self._loop and self._result_queue:
            # Inject exception into async stream to surface to consumer
            self._loop.call_soon_threadsafe(self._result_queue.put_nowait, exc)

    def _ingest_worker(self):
        """Pulls frames from WebRTC/NVDEC and submits them to the scheduler and fusion buffer."""
        while not self._stop_event.is_set():
            try:
                frame = self.stream.next()
                if self.fusion is not None:
                    ts = getattr(frame, "timestamp", 0.0)
                    self.fusion.add_video(frame, ts)
                self.scheduler.submit(frame)
            except StreamClosedError:
                # Normal network disconnect
                break
            except Exception as e:
                self._fail(e)
                break
                
        # End of stream means we should shut down the scheduler
        try:
            self.scheduler.close()
        except Exception:
            pass

    def _audio_ingest_worker(self):
        """Pulls audio chunks from WebRTC/Opus and submits them to the AudioScheduler."""
        while not self._stop_event.is_set():
            try:
                chunk = self.stream.next_audio()
                self.audio_scheduler.submit(chunk)
            except StreamClosedError:
                break
            except Exception as e:
                self._fail(e)
                break
                
        try:
            if self.audio_scheduler is not None:
                self.audio_scheduler.close()
        except Exception:
            pass

    def _audio_process_worker(self):
        """Processes audio chunks from AudioScheduler with VAD and ASR, updating TemporalFusion."""
        while not self._stop_event.is_set():
            try:
                chunk = self.audio_scheduler.acquire()
            except SchedulerClosedError:
                break
            except Exception as e:
                self._fail(e)
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
                            duration = max(0.0, transcript.end_timestamp - transcript.start_timestamp)
                            self.fusion.add_audio(payload=transcript, timestamp=transcript.start_timestamp, duration=duration)
                    elif self.fusion is not None:
                        self.fusion.add_audio(payload=chunk, timestamp=chunk.timestamp, duration=chunk.duration)
            except Exception as e:
                self._fail(e)
                break

    def _inference_worker(self):
        """Acquires paced/changed frames from scheduler, assembles multimodal context, and invokes the model."""
        while not self._stop_event.is_set():
            if not self._inference_consumer_active:
                time.sleep(0.05)
                continue
                
            try:
                frame = self.scheduler.acquire()
            except SchedulerClosedError:
                break
            except Exception as e:
                self._fail(e)
                break
                
            prompt = self._current_prompt
            t_infer_start = time.time()
            
            try:
                ts = getattr(frame, "timestamp", None)
                context = None
                transcripts = None
                
                if self.fusion is not None and ts is not None:
                    t_f0 = time.time()
                    context = self.fusion.context(timestamp=ts, window=self.context_window_s)
                    t_f1 = time.time()
                    fusion_lat = (t_f1 - t_f0) * 1000.0
                    
                    skew_ms = None
                    if hasattr(self.fusion, "_latest_skew_ms"):
                        skew_ms = self.fusion._latest_skew_ms
                    self._metrics.record_fusion(fusion_lat, skew_ms=skew_ms)
                    
                    transcripts = []
                    for obs in context.audio:
                        if isinstance(obs.payload, Transcript):
                            transcripts.append(obs.payload)
                        elif hasattr(obs.payload, "text"):
                            transcripts.append(obs.payload)
                
                try:
                    import inspect
                    sig = inspect.signature(self.vlm.generate)
                    gen_kwargs = {}
                    if "context" in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                        gen_kwargs["context"] = context
                    if "transcripts" in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                        gen_kwargs["transcripts"] = transcripts
                    response = self.vlm.generate(frame, prompt, **gen_kwargs)
                except TypeError:
                    # Fallback for adapters with custom dispatch
                    response = self.vlm.generate(frame, prompt)
                
                t_infer_end = time.time()
                total_latency_ms = (t_infer_end - t_infer_start) * 1000.0
                prep_lat_ms = getattr(response, "preprocessing_latency_ms", 0.0)
                infer_lat_ms = getattr(response, "latency_ms", total_latency_ms)
                
                # End-to-end latency: from frame arrival to response delivery
                e2e_lat_ms = total_latency_ms
                if ts is not None and ts > 0:
                    e2e_lat_ms = max(total_latency_ms, (time.time() - ts) * 1000.0)

                self._metrics.record_inference(
                    inference_latency_ms=infer_lat_ms,
                    preprocessing_latency_ms=prep_lat_ms,
                    end_to_end_latency_ms=e2e_lat_ms,
                )
                
                # Feedback latency to adaptive scheduler
                if hasattr(self.scheduler, "record_inference"):
                    if total_latency_ms > 0:
                        self.scheduler.record_inference(total_latency_ms)
                        
                if self._loop and self._result_queue:
                    self._loop.call_soon_threadsafe(self._result_queue.put_nowait, response)
            except Exception as e:
                self._metrics.record_inference_error()
                self.scheduler.release()
                self._fail(e)
                break
                
            self.scheduler.release()
            
        # If we exited the loop normally and we aren't failed, signal EOF to consumer
        with self._state_lock:
            if self.state not in (PipelineState.FAILED, PipelineState.STOPPING, PipelineState.STOPPED):
                self.state = PipelineState.STOPPED
            if self.state != PipelineState.FAILED and self._loop and self._result_queue:
                 self._loop.call_soon_threadsafe(self._result_queue.put_nowait, StopIteration)

    async def run_inference(self, prompt: str):
        """
        Async generator yielding VLMResponses based on the current prompt.
        
        Args:
            prompt: The text prompt describing what the model should look for.
        """
        if self.state != PipelineState.RUNNING:
            raise RuntimeError(f"Pipeline must be RUNNING, current state is {self.state.name}")
            
        if self._inference_consumer_active:
            raise RuntimeError("Only one active inference consumer is permitted.")
            
        self._current_prompt = prompt
        self._inference_consumer_active = True
        
        try:
            while True:
                result = await self._result_queue.get()
                if result is StopIteration:
                    break
                if isinstance(result, Exception):
                    raise result
                yield result
        finally:
            self._inference_consumer_active = False

    def metrics(self) -> Dict[str, Any]:
        """
        Return a thread-safe snapshot of pipeline performance metrics.
        """
        v_stats = self.scheduler.stats() if hasattr(self.scheduler, "stats") else {}
        a_stats = self.audio_scheduler.stats() if self.audio_scheduler and hasattr(self.audio_scheduler, "stats") else {}
        f_stats = self.fusion.stats() if self.fusion and hasattr(self.fusion, "stats") else {}
        q_depth = self._result_queue.qsize() if self._result_queue else 0

        return self._metrics.snapshot(
            pipeline_state=self.state.name,
            video_scheduler_stats=v_stats,
            audio_scheduler_stats=a_stats,
            fusion_stats=f_stats,
            inference_queue_depth=q_depth,
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Alias for metrics()."""
        return self.metrics()
