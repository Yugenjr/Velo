use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// Simulates WebRTC receiving H264 bytes natively,
/// passing them to NVDEC directly, and returning the GPU frame.
#[pyfunction]
fn decode_webrtc_h264(py: Python, video_path: &str) -> PyResult<PyObject> {
    // 1. Read file into native Rust buffer (simulating WebRTC payload)
    let data = std::fs::read(video_path)?;
    println!("[Rust] Read {} bytes into native Vec<u8>.", data.len());

    // 2. Import Python NVDEC binding natively via PyO3
    println!("[Rust] Loading PyNvVideoCodec native binding...");
    let nvc = PyModule::import_bound(py, "PyNvVideoCodec")?;

    // 3. Create Decoder specifying GPU memory
    println!("[Rust] Initializing GPU decoder...");
    let kwargs = PyDict::new_bound(py);
    kwargs.set_item("usedevicememory", true)?;
    let output_color_type = nvc.getattr("OutputColorType")?;
    kwargs.set_item("outputColorType", output_color_type.getattr("RGB")?)?;
    
    let decoder = nvc.getattr("CreateDecoder")?.call((), Some(&kwargs))?;

    // 4. Create PacketData pointing to Rust memory
    println!("[Rust] Constructing PacketData pointing to Rust Vec<u8>...");
    let packet = nvc.getattr("PacketData")?.call0()?;
    packet.setattr("bsl", data.len())?;
    packet.setattr("bsl_data", data.as_ptr() as usize)?;

    // 5. Decode frame directly from Rust memory
    // In a real async environment, we might use allow_threads to release GIL here if we were calling C++,
    // but since we are calling the Python wrapper, PyNvVideoCodec manages its own GIL internally.
    println!("[Rust] Invoking NVDEC Decode()...");
    let frames = decoder.call_method1("Decode", (packet,))?;

    // 6. Retrieve the decoded frames
    let frame_list: Bound<'_, PyList> = frames.downcast_into()?;
    let num_frames = frame_list.len();
    println!("[Rust] Successfully decoded {} frames.", num_frames);

    if num_frames == 0 {
        // Try flushing
        println!("[Rust] Flushing decoder...");
        let empty_packet = nvc.getattr("PacketData")?.call0()?;
        empty_packet.setattr("bsl", 0)?;
        empty_packet.setattr("bsl_data", 0)?;
        let flushed_frames = decoder.call_method1("Decode", (empty_packet,))?;
        let flushed_list: Bound<'_, PyList> = flushed_frames.downcast_into()?;
        if flushed_list.len() == 0 {
            return Err(pyo3::exceptions::PyRuntimeError::new_err("No frames decoded"));
        }
        return Ok(flushed_list.get_item(0)?.into_py(py));
    }

    // Return the first DecodedFrame (which contains the DLPack capsule) and the decoder to preserve lifetime
    let frame = frame_list.get_item(0)?;
    Ok((frame, decoder).into_py(py))
}

#[pymodule]
fn velo_gpu(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(decode_webrtc_h264, m)?)?;
    Ok(())
}
