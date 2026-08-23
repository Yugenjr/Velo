use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use std::sync::Arc;
use tokio::runtime::Runtime;
use crossbeam_channel::{bounded, Sender, Receiver};
use webrtc::api::media_engine::{MediaEngine, MIME_TYPE_H264};
use webrtc::api::APIBuilder;
use webrtc::api::interceptor_registry::register_default_interceptors;
use webrtc::interceptor::registry::Registry;
use webrtc::peer_connection::configuration::RTCConfiguration;
use webrtc::peer_connection::sdp::session_description::RTCSessionDescription;
use webrtc::rtp_transceiver::rtp_codec::{RTCRtpCodecCapability, RTCRtpCodecParameters, RTPCodecType};
use webrtc::rtp_transceiver::RTCPFeedback;
use rtp::codecs::h264::H264Packet;
use std::thread;
use pyo3::create_exception;

/// Custom Python exception mappings
create_exception!(velo, VeloError, pyo3::exceptions::PyException);
create_exception!(velo, StreamClosedError, VeloError);

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

#[pyclass]
struct NativeStream {
    frame_rx: Receiver<PyObject>,
    decoder_obj: PyObject,
}

#[pymethods]
impl NativeStream {
    fn next_frame(&self, py: Python) -> PyResult<PyObject> {
        // Release GIL while waiting for the next frame
        let frame_opt = py.allow_threads(|| {
            self.frame_rx.recv().ok()
        });
        
        match frame_opt {
            Some(frame) => {
                let decoder_ref = self.decoder_obj.clone_ref(py);
                let result_tuple = PyTuple::new_bound(py, &[frame.into_bound(py).into_any(), decoder_ref.into_bound(py).into_any()]);
                Ok(result_tuple.into_py(py))
            },
            None => Err(StreamClosedError::new_err("Stream closed or peer disconnected")),
        }
    }
    
    fn close(&self) {
        // When dropped, receivers close, signalling shutdown eventually
    }
}

#[pyfunction]
fn connect(py: Python, sdp_offer: String) -> PyResult<PyObject> {
    // 1. Initialize NvDecoder natively (allocates CUDA context)
    let nvdec = Arc::new(NvDecoder::new(py)?);
    let decoder_obj = nvdec.decoder_obj.clone_ref(py);

    // 2. Setup drop-oldest crossbeam channel for GPU frames
    // Capacity 3: If Python is slow, we drop the oldest frame to maintain realtime latency
    let (frame_tx, frame_rx) = bounded::<PyObject>(3);

    // 3. Spawn background Tokio runtime in a dedicated OS thread
    let (sdp_tx, sdp_rx) = bounded::<String>(1);
    
    let nvdec_clone = nvdec.clone();
    let rx_for_thread = frame_rx.clone();
    
    thread::spawn(move || {
        let rt = Runtime::new().unwrap();
        rt.block_on(async {
            // Setup WebRTC
            let mut m = MediaEngine::default();
            let video_rtcp_feedback = vec![
                RTCPFeedback { typ: "goog-remb".to_owned(), parameter: "".to_owned() },
                RTCPFeedback { typ: "ccm".to_owned(), parameter: "fir".to_owned() },
                RTCPFeedback { typ: "nack".to_owned(), parameter: "".to_owned() },
                RTCPFeedback { typ: "nack".to_owned(), parameter: "pli".to_owned() },
            ];

            let _ = m.register_codec(
                RTCRtpCodecParameters {
                    capability: RTCRtpCodecCapability {
                        mime_type: MIME_TYPE_H264.to_owned(),
                        clock_rate: 90000,
                        channels: 0,
                        sdp_fmtp_line: "level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=42e01f".to_owned(),
                        rtcp_feedback: video_rtcp_feedback.clone(),
                    },
                    payload_type: 125,
                    ..Default::default()
                },
                RTPCodecType::Video,
            );

            let mut registry = Registry::new();
            registry = register_default_interceptors(registry, &mut m).unwrap();
            let api = APIBuilder::new()
                .with_media_engine(m)
                .with_interceptor_registry(registry)
                .build();

            let config = RTCConfiguration {
                ice_servers: vec![webrtc::ice_transport::ice_server::RTCIceServer {
                    urls: vec!["stun:stun.l.google.com:19302".to_owned()],
                    ..Default::default()
                }],
                ..Default::default()
            };

            let pc = Arc::new(api.new_peer_connection(config).await.unwrap());

            // Track handler
            let tx = frame_tx.clone();
            let rx = rx_for_thread.clone();
            let nvdec_thread = nvdec_clone.clone();
            
            pc.on_track(Box::new(move |track, _receiver, _transceiver| {
                let track = track.clone();
                let tx = tx.clone();
                let rx = rx.clone();
                let nvdec_thread = nvdec_thread.clone();
                
                Box::pin(async move {
                    if track.codec().capability.mime_type.to_lowercase() == MIME_TYPE_H264.to_lowercase() {
                        tokio::spawn(async move {
                            let mut h264_depacketizer = H264Packet::default();
                            
                            while let Ok((rtp_packet, _)) = track.read_rtp().await {
                                let payload = rtp_packet.payload;
                                if let Ok(nal_bytes) = rtp::packetizer::Depacketizer::depacketize(&mut h264_depacketizer, &payload) {
                                    if !nal_bytes.is_empty() {
                                        // Annex-B wrapping
                                        let mut buffer = Vec::with_capacity(4 + nal_bytes.len());
                                        buffer.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
                                        buffer.extend_from_slice(&nal_bytes);
                                        
                                        // Decode via NVDEC (requires GIL briefly)
                                        let frames_res = Python::with_gil(|py| {
                                            nvdec_thread.decode(py, &buffer)
                                        });
                                        
                                        if let Ok(frames) = frames_res {
                                            for frame in frames {
                                                // Bounded push: drop oldest if full
                                                while tx.is_full() {
                                                    let _ = rx.try_recv(); // discard
                                                }
                                                let _ = tx.send(frame);
                                            }
                                        }
                                    }
                                }
                            }
                        });
                    }
                })
            }));

            // Handle SDP
            let desc = RTCSessionDescription::offer(sdp_offer).unwrap();
            pc.set_remote_description(desc).await.unwrap();
            let answer = pc.create_answer(None).await.unwrap();
            
            let mut gather_complete = pc.gathering_complete_promise().await;
            pc.set_local_description(answer).await.unwrap();
            let _ = gather_complete.recv().await;

            let local_desc = pc.local_description().await.unwrap();
            let _ = sdp_tx.send(local_desc.sdp);
            
            // Keep Tokio runtime alive
            let mut close_rx = frame_tx.clone();
            // wait indefinitely or until channel drops
            tokio::signal::ctrl_c().await.unwrap();
        });
    });

    let sdp_answer = sdp_rx.recv().map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("Failed to get SDP answer"))?;
    
    // Return a tuple of (NativeStream, sdp_answer)
    let native_stream = NativeStream {
        frame_rx,
        decoder_obj,
    };
    
    let result = PyTuple::new_bound(py, &[native_stream.into_py(py).into_bound(py).into_any(), sdp_answer.into_py(py).into_bound(py).into_any()]);
    Ok(result.into_py(py))
}

#[pymodule]
fn _velo_native(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(connect, m)?)?;
    m.add("VeloError", py.get_type_bound::<VeloError>())?;
    m.add("StreamClosedError", py.get_type_bound::<StreamClosedError>())?;
    Ok(())
}
