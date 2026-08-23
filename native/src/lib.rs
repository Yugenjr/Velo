use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use pyo3::create_exception;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use tokio::runtime::Runtime;
use crossbeam_channel::{bounded, Receiver};
use webrtc::api::media_engine::{MediaEngine, MIME_TYPE_H264};
use webrtc::api::APIBuilder;
use webrtc::api::interceptor_registry::register_default_interceptors;
use webrtc::interceptor::registry::Registry;
use webrtc::peer_connection::configuration::RTCConfiguration;
use webrtc::peer_connection::sdp::session_description::RTCSessionDescription;
use webrtc::rtp_transceiver::rtp_codec::{RTCRtpCodecCapability, RTCRtpCodecParameters, RTPCodecType};
use webrtc::rtp_transceiver::RTCPFeedback;
use rtp::codecs::h264::H264Packet;
use std::thread::{self, JoinHandle};

// ---------------------------------------------------------------------------
// Python exception hierarchy
// ---------------------------------------------------------------------------
create_exception!(velo, VeloError, pyo3::exceptions::PyException);
create_exception!(velo, VeloConnectionError, VeloError);
create_exception!(velo, StreamClosedError, VeloError);
create_exception!(velo, DecodeError, VeloError);

// ---------------------------------------------------------------------------
// NVDEC decoder wrapper (calls PyNvVideoCodec through the Python C-API)
// ---------------------------------------------------------------------------
struct NvDecoder {
    decoder_obj: PyObject,
}

impl NvDecoder {
    fn new(py: Python) -> PyResult<Self> {
        let nvc = PyModule::import_bound(py, "PyNvVideoCodec")?;
        let kwargs = PyDict::new_bound(py);
        kwargs.set_item("usedevicememory", true)?;
        let output_color_type = nvc.getattr("OutputColorType")?;
        kwargs.set_item("outputColorType", output_color_type.getattr("RGB")?)?;

        let decoder_obj = nvc.getattr("CreateDecoder")?.call((), Some(&kwargs))?.into_py(py);
        Ok(Self { decoder_obj })
    }

    /// Decode a buffer of Annex-B NAL data. Returns decoded GPU frames.
    fn decode(&self, py: Python, nal_data: &[u8]) -> PyResult<Vec<PyObject>> {
        let nvc = PyModule::import_bound(py, "PyNvVideoCodec")?;
        let packet = nvc.getattr("PacketData")?.call0()?;
        packet.setattr("bsl", nal_data.len())?;
        packet.setattr("bsl_data", nal_data.as_ptr() as usize)?;

        let decoder_bound = self.decoder_obj.bind(py);
        let frames = decoder_bound.call_method1("Decode", (packet,))?;

        let frame_list: Bound<'_, PyList> = frames.downcast_into()?;
        let mut results = Vec::new();
        for i in 0..frame_list.len() {
            results.push(frame_list.get_item(i)?.into_py(py));
        }
        Ok(results)
    }
}

// ---------------------------------------------------------------------------
// NativeStream — the PyO3-exposed stream object
// ---------------------------------------------------------------------------
#[pyclass]
struct NativeStream {
    /// Receives decoded GPU frames from the worker thread.
    frame_rx: Receiver<PyObject>,
    /// Keeps the NVDEC decoder (and its CUDA context) alive for DLPack validity.
    decoder_obj: PyObject,
    /// Signals the worker thread to shut down.
    shutdown: Arc<AtomicBool>,
    /// Handle to the worker OS thread. Taken by close().
    worker_handle: std::sync::Mutex<Option<JoinHandle<()>>>,
    /// True once close() has been called.
    closed: AtomicBool,
    /// Number of frames dropped by the bounded queue (drop-oldest policy).
    dropped_count: Arc<AtomicU64>,
    /// Number of NVDEC decode errors encountered.
    error_count: Arc<AtomicU64>,
}

#[pymethods]
impl NativeStream {
    /// Block until the next decoded GPU frame arrives.
    /// Returns (DecodedFrame, decoder_ref) as a tuple.
    /// Releases the Python GIL while waiting.
    fn next_frame(&self, py: Python) -> PyResult<PyObject> {
        if self.closed.load(Ordering::SeqCst) {
            return Err(StreamClosedError::new_err("Stream is closed"));
        }

        // Release GIL so the worker thread can acquire it for decoding,
        // and so other Python threads can proceed.
        let frame_opt = py.allow_threads(|| self.frame_rx.recv().ok());

        match frame_opt {
            Some(frame) => {
                // Return (frame, decoder_ref) so Python keeps the decoder alive
                let decoder_ref = self.decoder_obj.clone_ref(py);
                let result = PyTuple::new_bound(
                    py,
                    &[
                        frame.into_bound(py).into_any(),
                        decoder_ref.into_bound(py).into_any(),
                    ],
                );
                Ok(result.into_py(py))
            }
            // Channel closed — worker thread exited or was shut down
            None => Err(StreamClosedError::new_err(
                "Stream closed or peer disconnected",
            )),
        }
    }

    /// Gracefully shut down the stream.
    ///
    /// 1. Signal the worker thread to stop.
    /// 2. Release the GIL and join the worker thread (so it can finish
    ///    its current GIL-holding decode without deadlocking).
    /// 3. Drain any remaining frames from the queue under the GIL.
    ///
    /// Idempotent — safe to call multiple times.
    fn close(&self, py: Python) {
        // Atomically set closed; if already true, return immediately.
        if self.closed.swap(true, Ordering::SeqCst) {
            return;
        }

        // 1. Signal the worker to exit its loops
        self.shutdown.store(true, Ordering::SeqCst);

        // 2. Join the worker thread with GIL released.
        //    The worker may be inside Python::with_gil(decode); releasing here
        //    lets it finish that call, see the shutdown flag, and exit.
        let handle = self.worker_handle.lock().unwrap().take();
        if let Some(handle) = handle {
            py.allow_threads(|| {
                let _ = handle.join();
            });
        }

        // 3. Drain remaining PyObject frames under the GIL (safe destruction)
        while let Ok(_frame) = self.frame_rx.try_recv() {
            // Dropped here with GIL held — correct ref-count decrement
        }
    }

    /// Number of frames dropped by the bounded queue.
    #[getter]
    fn dropped_frames(&self) -> u64 {
        self.dropped_count.load(Ordering::Relaxed)
    }

    /// Number of NVDEC decode errors encountered.
    #[getter]
    fn decode_errors(&self) -> u64 {
        self.error_count.load(Ordering::Relaxed)
    }
}

impl Drop for NativeStream {
    fn drop(&mut self) {
        // If the user forgot to call close(), signal the worker to stop.
        if !self.closed.load(Ordering::SeqCst) {
            self.shutdown.store(true, Ordering::SeqCst);
        }
    }
}

// Helper to run connection logic in worker thread without panic propagation
async fn run_connection_setup(
    sdp_offer: &str,
    shutdown: Arc<AtomicBool>,
    dropped_count: Arc<AtomicU64>,
    error_count: Arc<AtomicU64>,
    nvdec: Arc<NvDecoder>,
    frame_tx: crossbeam_channel::Sender<PyObject>,
    drain_rx: crossbeam_channel::Receiver<PyObject>,
    sdp_tx: crossbeam_channel::Sender<Result<String, String>>,
) {
    // ---- WebRTC setup ----
    let mut m = MediaEngine::default();
    let rtcp_fb = vec![
        RTCPFeedback { typ: "goog-remb".into(), parameter: "".into() },
        RTCPFeedback { typ: "ccm".into(), parameter: "fir".into() },
        RTCPFeedback { typ: "nack".into(), parameter: "".into() },
        RTCPFeedback { typ: "nack".into(), parameter: "pli".into() },
    ];

    // Register multiple H.264 profiles
    for (fmtp, pt) in [
        ("level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=42e01f", 125u8),
        ("level-asymmetry-allowed=1;packetization-mode=0;profile-level-id=42e01f", 108),
        ("level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=42001f", 127),
    ] {
        if let Err(e) = m.register_codec(
            RTCRtpCodecParameters {
                capability: RTCRtpCodecCapability {
                    mime_type: MIME_TYPE_H264.to_owned(),
                    clock_rate: 90000,
                    channels: 0,
                    sdp_fmtp_line: fmtp.to_owned(),
                    rtcp_feedback: rtcp_fb.clone(),
                },
                payload_type: pt,
                ..Default::default()
            },
            RTPCodecType::Video,
        ) {
            let _ = sdp_tx.send(Err(format!("Failed to register codec: {:?}", e)));
            return;
        }
    }

    let mut registry = Registry::new();
    let mut m = m;
    let registry = match register_default_interceptors(registry, &mut m) {
        Ok(r) => r,
        Err(e) => {
            let _ = sdp_tx.send(Err(format!("Interceptor setup failed: {:?}", e)));
            return;
        }
    };

    let api = APIBuilder::new()
        .with_media_engine(m)
        .with_interceptor_registry(registry)
        .build();

    let config = RTCConfiguration {
        ice_servers: vec![webrtc::ice_transport::ice_server::RTCIceServer {
            urls: vec!["stun:stun.l.google.com:19302".into()],
            ..Default::default()
        }],
        ..Default::default()
    };

    let pc = match api.new_peer_connection(config).await {
        Ok(p) => Arc::new(p),
        Err(e) => {
            let _ = sdp_tx.send(Err(format!("Peer connection creation failed: {:?}", e)));
            return;
        }
    };
    let pc_for_cleanup = pc.clone();

    // ---- Track handler ----
    let tx = frame_tx;
    let nvdec_track = nvdec.clone();
    let shutdown_track = shutdown.clone();

    pc.on_track(Box::new(move |track, _receiver, _transceiver| {
        let track = track.clone();
        let tx = tx.clone();
        let drain_rx = drain_rx.clone();
        let nvdec_t = nvdec_track.clone();
        let shutdown_t = shutdown_track.clone();
        let dropped_t = dropped_count.clone();
        let errors_t = error_count.clone();

        Box::pin(async move {
            let codec = track.codec().capability.mime_type.to_lowercase();
            if codec != MIME_TYPE_H264.to_lowercase() {
                return;
            }

            tokio::spawn(async move {
                let mut depacketizer = H264Packet::default();
                let mut nal_buffer: Vec<u8> = Vec::with_capacity(65536);
                let mut current_ts: u32 = 0;
                let mut first_packet = true;

                while let Ok((rtp_pkt, _)) = track.read_rtp().await {
                    if shutdown_t.load(Ordering::Relaxed) {
                        break;
                    }

                    let ts = rtp_pkt.header.timestamp;
                    let payload = rtp_pkt.payload;

                    let nal_bytes = match rtp::packetizer::Depacketizer::depacketize(
                        &mut depacketizer,
                        &payload,
                    ) {
                        Ok(b) if !b.is_empty() => b,
                        _ => continue,
                    };

                    if !first_packet && ts != current_ts && !nal_buffer.is_empty() {
                        Python::with_gil(|py| {
                            match nvdec_t.decode(py, &nal_buffer) {
                                Ok(frames) => {
                                    for frame in frames {
                                        while tx.is_full() {
                                            if let Ok(_old) = drain_rx.try_recv() {
                                                dropped_t.fetch_add(1, Ordering::Relaxed);
                                            }
                                        }
                                        let _ = tx.send(frame);
                                    }
                                }
                                Err(e) => {
                                    errors_t.fetch_add(1, Ordering::Relaxed);
                                    eprintln!("[velo] NVDEC decode error: {}", e);
                                }
                            }
                        });
                        nal_buffer.clear();
                    }

                    first_packet = false;
                    current_ts = ts;
                    nal_buffer.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
                    nal_buffer.extend_from_slice(&nal_bytes);
                }

                if !nal_buffer.is_empty() {
                    Python::with_gil(|py| {
                        match nvdec_t.decode(py, &nal_buffer) {
                            Ok(frames) => {
                                for frame in frames {
                                    while tx.is_full() {
                                        if let Ok(_old) = drain_rx.try_recv() {
                                            dropped_t.fetch_add(1, Ordering::Relaxed);
                                        }
                                    }
                                    let _ = tx.send(frame);
                                }
                            }
                            Err(e) => {
                                errors_t.fetch_add(1, Ordering::Relaxed);
                                eprintln!("[velo] NVDEC decode error (final flush): {}", e);
                            }
                        }
                    });
                }
            });
        })
    }));

    // ---- SDP exchange ----
    let desc = match RTCSessionDescription::offer(sdp_offer.to_string()) {
        Ok(d) => d,
        Err(e) => {
            let _ = sdp_tx.send(Err(format!("Invalid SDP offer: {:?}", e)));
            return;
        }
    };

    if let Err(e) = pc.set_remote_description(desc).await {
        let _ = sdp_tx.send(Err(format!("set_remote_description failed: {:?}", e)));
        return;
    }

    let answer = match pc.create_answer(None).await {
        Ok(a) => a,
        Err(e) => {
            let _ = sdp_tx.send(Err(format!("create_answer failed: {:?}", e)));
            return;
        }
    };

    let mut gather_complete = pc.gathering_complete_promise().await;
    if let Err(e) = pc.set_local_description(answer).await {
        let _ = sdp_tx.send(Err(format!("set_local_description failed: {:?}", e)));
        return;
    }
    let _ = gather_complete.recv().await;

    let local_desc = match pc.local_description().await {
        Some(ld) => ld,
        None => {
            let _ = sdp_tx.send(Err("No local description after gathering".to_string()));
            return;
        }
    };

    // Send SDP answer back to main thread
    if sdp_tx.send(Ok(local_desc.sdp)).is_err() {
        return; // main thread hung up
    }

    // Keep the Tokio runtime alive on this thread and await shutdown
    while !shutdown.load(Ordering::Relaxed) {
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }

    // Graceful cleanup of WebRTC
    let _ = pc_for_cleanup.close().await;

    // Small delay to ensure WebRTC task terminates cleanly
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
}

// ---------------------------------------------------------------------------
// connect() — the main entry point
// ---------------------------------------------------------------------------
#[pyfunction]
fn connect(py: Python, sdp_offer: String) -> PyResult<PyObject> {
    // ---- 1. Initialize NVDEC decoder (allocates CUDA context) ----
    let nvdec = Arc::new(NvDecoder::new(py)?);
    let decoder_obj_for_stream = nvdec.decoder_obj.clone_ref(py);

    // ---- 2. Bounded frame queue ----
    let (frame_tx, frame_rx) = bounded::<PyObject>(3);
    let drain_rx = frame_rx.clone();

    // ---- 3. Shutdown and metrics ----
    let shutdown = Arc::new(AtomicBool::new(false));
    let shutdown_worker = shutdown.clone();

    let dropped_count = Arc::new(AtomicU64::new(0));
    let error_count = Arc::new(AtomicU64::new(0));

    // ---- 4. SDP answer channel ----
    let (sdp_tx, sdp_rx) = bounded::<Result<String, String>>(1);

    let nvdec_for_worker = nvdec.clone();
    let dropped_worker = dropped_count.clone();
    let errors_worker = error_count.clone();

    // ---- 5. Spawn dedicated OS worker thread ----
    let worker_handle = thread::spawn(move || {
        let rt = Runtime::new().expect("[velo] failed to create Tokio runtime");
        rt.block_on(async {
            run_connection_setup(
                &sdp_offer,
                shutdown_worker,
                dropped_worker,
                errors_worker,
                nvdec_for_worker,
                frame_tx,
                drain_rx,
                sdp_tx,
            ).await;
        });
    });

    // ---- 6. Wait for SDP answer or connection error ----
    let sdp_result = sdp_rx.recv().map_err(|_| {
        pyo3::exceptions::PyRuntimeError::new_err(
            "Native WebRTC runtime failed before producing an SDP answer",
        )
    })?;

    let sdp_answer = match sdp_result {
        Ok(answer) => answer,
        Err(err_msg) => {
            // Join the thread to clean up before throwing the exception
            let _ = worker_handle.join();
            return Err(VeloConnectionError::new_err(err_msg));
        }
    };

    // ---- Build the NativeStream ----
    let native_stream = NativeStream {
        frame_rx,
        decoder_obj: decoder_obj_for_stream,
        shutdown,
        worker_handle: std::sync::Mutex::new(Some(worker_handle)),
        closed: AtomicBool::new(false),
        dropped_count,
        error_count,
    };

    let result = PyTuple::new_bound(
        py,
        &[
            native_stream.into_py(py).into_bound(py).into_any(),
            sdp_answer.into_py(py).into_bound(py).into_any(),
        ],
    );
    Ok(result.into_py(py))
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------
#[pymodule]
fn _velo_native(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(connect, m)?)?;
    m.add("VeloError", py.get_type_bound::<VeloError>())?;
    m.add("VeloConnectionError", py.get_type_bound::<VeloConnectionError>())?;
    m.add("StreamClosedError", py.get_type_bound::<StreamClosedError>())?;
    m.add("DecodeError", py.get_type_bound::<DecodeError>())?;
    Ok(())
}
