use anyhow::Result;
use axum::{
    extract::State,
    http::StatusCode,
    response::{Html, IntoResponse},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::Write;
use std::sync::Arc;
use tokio::sync::Mutex;
use webrtc::api::interceptor_registry::register_default_interceptors;
use webrtc::api::media_engine::{MediaEngine, MIME_TYPE_H264};
use webrtc::api::APIBuilder;
use webrtc::ice_transport::ice_server::RTCIceServer;
use webrtc::interceptor::registry::Registry;
use webrtc::peer_connection::configuration::RTCConfiguration;
use webrtc::peer_connection::peer_connection_state::RTCPeerConnectionState;
use webrtc::peer_connection::sdp::session_description::RTCSessionDescription;
use webrtc::rtp_transceiver::RTCRtpTransceiver;
use webrtc::rtp_transceiver::rtp_receiver::RTCRtpReceiver;
use webrtc::track::track_remote::TrackRemote;
use rtp::codecs::h264::H264Packet;
use std::path::PathBuf;
use std::env;

#[derive(Deserialize)]
struct SdpOffer {
    sdp: String,
    #[serde(rename = "type")]
    sdp_type: String,
}

#[derive(Serialize)]
struct SdpAnswer {
    sdp: String,
    #[serde(rename = "type")]
    sdp_type: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("Starting V0.2-B Native WebRTC Server...");
    
    // Clear out old file in the requested directory
    let out_path = r"..\temp\native_webrtc_experiment\output.h264";
    if std::path::Path::new(out_path).exists() {
        std::fs::remove_file(out_path)?;
    }

    let app = Router::new()
        .route("/", get(index_html))
        .route("/offer", post(handle_offer));

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await?;
    println!("Listening on http://localhost:8080");
    
    axum::serve(listener, app).await?;
    
    Ok(())
}

async fn index_html() -> impl IntoResponse {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("index.html");
    match std::fs::read_to_string(path) {
        Ok(html) => Html(html).into_response(),
        Err(_) => (StatusCode::NOT_FOUND, "index.html not found").into_response(),
    }
}

async fn handle_offer(Json(offer): Json<SdpOffer>) -> Result<Json<SdpAnswer>, StatusCode> {
    let answer = process_webrtc_offer(offer).await.map_err(|e| {
        println!("Error processing offer: {:?}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;
    Ok(Json(answer))
}

async fn process_webrtc_offer(offer: SdpOffer) -> Result<SdpAnswer> {
    use webrtc::rtp_transceiver::rtp_codec::{RTCRtpCodecCapability, RTCRtpCodecParameters, RTPCodecType};
    use webrtc::api::media_engine::MIME_TYPE_H264;
    use webrtc::rtcp::payload_feedbacks::picture_loss_indication::PictureLossIndication;
    use webrtc::rtp_transceiver::RTCPFeedback;

    let mut m = MediaEngine::default();
    
    let video_rtcp_feedback = vec![
        RTCPFeedback {
            typ: "goog-remb".to_owned(),
            parameter: "".to_owned(),
        },
        RTCPFeedback {
            typ: "ccm".to_owned(),
            parameter: "fir".to_owned(),
        },
        RTCPFeedback {
            typ: "nack".to_owned(),
            parameter: "".to_owned(),
        },
        RTCPFeedback {
            typ: "nack".to_owned(),
            parameter: "pli".to_owned(),
        },
    ];

    m.register_codec(
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
    )?;

    m.register_codec(
        RTCRtpCodecParameters {
            capability: RTCRtpCodecCapability {
                mime_type: MIME_TYPE_H264.to_owned(),
                clock_rate: 90000,
                channels: 0,
                sdp_fmtp_line: "level-asymmetry-allowed=1;packetization-mode=0;profile-level-id=42e01f".to_owned(),
                rtcp_feedback: video_rtcp_feedback.clone(),
            },
            payload_type: 108,
            ..Default::default()
        },
        RTPCodecType::Video,
    )?;

    m.register_codec(
        RTCRtpCodecParameters {
            capability: RTCRtpCodecCapability {
                mime_type: MIME_TYPE_H264.to_owned(),
                clock_rate: 90000,
                channels: 0,
                sdp_fmtp_line: "level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=42001f".to_owned(),
                rtcp_feedback: video_rtcp_feedback.clone(),
            },
            payload_type: 127,
            ..Default::default()
        },
        RTPCodecType::Video,
    )?;

    m.register_codec(
        RTCRtpCodecParameters {
            capability: RTCRtpCodecCapability {
                mime_type: MIME_TYPE_H264.to_owned(),
                clock_rate: 90000,
                channels: 0,
                sdp_fmtp_line: "level-asymmetry-allowed=1;packetization-mode=0;profile-level-id=42001f".to_owned(),
                rtcp_feedback: video_rtcp_feedback.clone(),
            },
            payload_type: 127,
            ..Default::default()
        },
        RTPCodecType::Video,
    )?;
    
    // 2. Setup Interceptors
    let mut registry = Registry::new();
    registry = register_default_interceptors(registry, &mut m)?;

    // 3. Build the API
    let api = APIBuilder::new()
        .with_media_engine(m)
        .with_interceptor_registry(registry)
        .build();

    // 4. Configure PeerConnection
    let config = RTCConfiguration {
        ice_servers: vec![RTCIceServer {
            urls: vec!["stun:stun.l.google.com:19302".to_owned()],
            ..Default::default()
        }],
        ..Default::default()
    };

    let pc = Arc::new(api.new_peer_connection(config).await?);

    // 5. Setup Track Event Handler
    pc.on_track(Box::new(|track: Arc<TrackRemote>, _receiver: Arc<RTCRtpReceiver>, _transceiver: Arc<RTCRtpTransceiver>| {
        println!("Negotiated Track Codec: {}", track.codec().capability.mime_type);
        
        let track_clone = track.clone();
        Box::pin(async move {
            if track_clone.codec().capability.mime_type.to_lowercase() == MIME_TYPE_H264.to_lowercase() {
                println!("--- ENCODED H.264 TRACK DETECTED ---");
                tokio::spawn(async move {
                    if let Err(e) = process_h264_rtp(track_clone).await {
                        println!("Error processing RTP: {:?}", e);
                    }
                });
            } else {
                println!("WARNING: Unexpected codec negotiated. Expected H264.");
            }
        })
    }));

    // Set connection state handler
    pc.on_peer_connection_state_change(Box::new(|s: RTCPeerConnectionState| {
        println!("PeerConnection State: {}", s);
        Box::pin(async {})
    }));

    // 6. Handle SDP Offer/Answer
    let desc = RTCSessionDescription::offer(offer.sdp).unwrap();
    pc.set_remote_description(desc).await?;

    let answer = pc.create_answer(None).await?;
    let mut gather_complete = pc.gathering_complete_promise().await;
    pc.set_local_description(answer).await?;
    let _ = gather_complete.recv().await;

    let local_desc = pc.local_description().await.unwrap();

    Ok(SdpAnswer {
        sdp: local_desc.sdp,
        sdp_type: local_desc.sdp_type.to_string(),
    })
}

async fn process_h264_rtp(track: Arc<TrackRemote>) -> Result<()> {
    // Ensure directory exists
    std::fs::create_dir_all(r"..\temp\native_webrtc_experiment")?;
    
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(r"..\temp\native_webrtc_experiment\output.h264")?;

    println!("Ready to receive RTP packets and extract H.264 NAL Units...");
    
    let mut packet_count = 0;
    let mut bytes_received = 0;
    let mut sps_count = 0;
    let mut pps_count = 0;
    let mut idr_count = 0;
    
    // We use H264Packet to depacketize (this natively resolves Single NAL, STAP-A, FU-A etc.)
    let mut h264_depacketizer = H264Packet::default();
    
    while let Ok((rtp_packet, _)) = track.read_rtp().await {
        packet_count += 1;
        
        let payload = rtp_packet.payload;
        let payload_len = payload.len();
        
        // This extraction proves we have ENCODED media, bypassing CPU decoding
        match rtp::packetizer::Depacketizer::depacketize(&mut h264_depacketizer, &payload) {
            Ok(nal_bytes) => {
                if !nal_bytes.is_empty() {
                    bytes_received += nal_bytes.len();
                    
                    // Inspect NAL type (last 5 bits of the first byte)
                    let nal_type = nal_bytes[0] & 0x1F;
                    if nal_type == 7 {
                        sps_count += 1;
                    } else if nal_type == 8 {
                        pps_count += 1;
                    } else if nal_type == 5 {
                        idr_count += 1;
                    }
                    
                    // Prepend the Annex-B start code and write to file
                    file.write_all(&[0x00, 0x00, 0x00, 0x01])?;
                    file.write_all(&nal_bytes)?;
                    
                    // Print summary periodically
                    if packet_count % 30 == 0 {
                        println!(
                            "--- RTP Reception Summary ---\n\
                             Total H264 RTP Packets: {}\n\
                             Payload bytes received: {}\n\
                             SPS observed: {}\n\
                             PPS observed: {}\n\
                             IDR observed: {}\n\
                             CPU Video Decoding Used: NO",
                            packet_count, bytes_received, sps_count, pps_count, idr_count
                        );
                    }
                    
                    // Stop after 200 packets to limit file size for debugging
                    if packet_count >= 200 {
                        println!("Capture limit reached. Stopping track reading.");
                        break;
                    }
                }
            }
            Err(_e) => {
                // Fragmented packets that haven't completed a full NAL will return an error or wait.
                // webrtc-rs handles standard fragmented reassembly here if fed correctly,
                // but real streams use a jitter buffer (media::SampleBuilder).
                // For V0.2-B, manual extraction suffices to prove the architecture.
            }
        }
    }
    
    Ok(())
}
