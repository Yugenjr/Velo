import enum
import threading
import asyncio
import time
from typing import Optional, Union
from .scheduler import AIScheduler, SchedulerClosedError
from .vlm import BaseVLMAdapter, VLMResponse
from .exceptions import StreamClosedError, VeloError

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
    
    Composes a Stream, AIScheduler, and VLMAdapter, managing their blocking
    OS threads safely and exposing a clean asynchronous generator for agents.
    """
    def __init__(self, stream, scheduler: AIScheduler, vlm: BaseVLMAdapter):
        self.stream = stream
        self.scheduler = scheduler
        self.vlm = vlm
        
        self.state = PipelineState.CREATED
        self._state_lock = threading.Lock()
        
        self._ingest_thread: Optional[threading.Thread] = None
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
        
        self._ingest_thread = threading.Thread(target=self._ingest_worker, name="VeloIngest", daemon=True)
        self._inference_thread = threading.Thread(target=self._inference_worker, name="VeloInference", daemon=True)
        
        self._ingest_thread.start()
        self._inference_thread.start()
        
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
            
        if self._loop and self._result_queue:
            self._loop.call_soon_threadsafe(self._result_queue.put_nowait, StopIteration)

        if self._loop:
            await self._loop.run_in_executor(None, self._join_threads)
            
        with self._state_lock:
            if self.state != PipelineState.FAILED:
                self.state = PipelineState.STOPPED

    def _join_threads(self):
        if self._ingest_thread and self._ingest_thread.is_alive():
            self._ingest_thread.join(timeout=5.0)
        if self._inference_thread and self._inference_thread.is_alive():
            self._inference_thread.join(timeout=5.0)

    def _fail(self, exc: Exception):
        """Transition to FAILED and initiate cascading shutdown."""
        with self._state_lock:
            if self.state in (PipelineState.STOPPING, PipelineState.STOPPED, PipelineState.FAILED):
                return
            self.state = PipelineState.FAILED
            
        self._stop_event.set()
        
        try:
            self.stream.close()
        except Exception:
            pass
            
        try:
            self.scheduler.close()
        except Exception:
            pass
            
        if self._loop and self._result_queue:
            # Inject exception into async stream to surface to consumer
            self._loop.call_soon_threadsafe(self._result_queue.put_nowait, exc)

    def _ingest_worker(self):
        """Pulls frames from WebRTC/NVDEC and submits them to the scheduler."""
        while not self._stop_event.is_set():
            try:
                frame = self.stream.next()
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

    def _inference_worker(self):
        """Acquires paced/changed frames from scheduler and invokes the VLM."""
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
            
            try:
                response = self.vlm.generate(frame, prompt)
                
                # Feedback latency to adaptive scheduler
                if hasattr(self.scheduler, "record_inference"):
                    total_latency = getattr(response, "latency_ms", 0.0) + getattr(response, "preprocessing_latency_ms", 0.0)
                    if total_latency > 0:
                        self.scheduler.record_inference(total_latency)
                        
                if self._loop and self._result_queue:
                    self._loop.call_soon_threadsafe(self._result_queue.put_nowait, response)
            except Exception as e:
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
