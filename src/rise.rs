//! Independent causal onset path. Contract: dist/train_short_onset.py.
use anyhow::{ensure, Context, Result};
use ort::{session::Session, value::Value};
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::{
    collections::VecDeque,
    sync::{
        mpsc::{sync_channel, SyncSender},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};

#[path = "rise_rescue.rs"]
mod rescue;

#[path = "flux.rs"]
pub(crate) mod flux;

const SR: u64 = 16_000;
const HOP: usize = 256;
const FEATURES: usize = 770;
const HISTORY: usize = 35; // current frame + 30 temporal + 4 rise baseline

// An attack may finish the existing 120ms confirmation, or walk a polyphonic
// grip. Consumption and round boundaries end its life sooner; this is not an
// extra waiting period and it never rearms the detector.
pub const EVENT_LIFETIME: Duration = Duration::from_millis(768);

pub fn enabled() -> bool {
    std::env::var("SOLITITO_ONSET").map_or(true, |v| v != "legacy")
}

/// SOLITITO_FLUX: also report an attack the model did not name, from the
/// pitch-blind flux detector plus a harmonic sum. Off by default.
///
/// `1` takes the default sensitivity; a number is the deviation count the flux
/// threshold uses, so the one variable both switches the experiment on and says
/// how hard it listens. Smaller hears more and reports more that nobody played:
/// swept on AtoA, every value from 5 to 25 finds all 51 notes, while attacks on
/// the user's own recording go 27, 36, 48, 88 as it falls 25, 18, 12, 5.
///
/// An experiment, not a setting. What is not known is how many of the extra
/// reports are notes nobody played: the only labelled material labels 51 notes
/// in 91 seconds and cannot tell an unlabelled note from a false report.
pub fn flux_deviations() -> Option<f32> {
    // Empty counts as off: `SOLITITO_FLUX= solitito` reads as switching it off,
    // and the variable set to nothing must not turn an experiment on.
    let value = std::env::var("SOLITITO_FLUX").ok()?;
    match value.trim() {
        "" | "0" => None,
        "1" => Some(flux::DEVIATIONS),
        other => other.parse::<f32>().ok().filter(|v| v.is_finite() && *v > 0.0),
    }
}

pub fn flux_enabled() -> bool {
    flux_deviations().is_some()
}

pub fn model_path() -> Result<String> {
    if let Ok(path) = std::env::var("SOLITITO_ONSET_MODEL") {
        return Ok(path);
    }
    let primary = crate::model_path();
    if crate::onnx_model::is_combined(&primary)? {
        Ok(primary)
    } else {
        Ok("short_onset_rise.onnx".into())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Event {
    pub frame: u64,
    pub pc: usize,
    pub at: Instant,
}

// Rise may detect an attack before the 128 ms RMS window opens the noise gate.
// Retain that strong event for at most its 96 ms training target, and only while
// the model still confirms it. Preserve its original identity and audio time:
// opening the gate must not turn an old attack into one from the next exercise.
#[derive(Default)]
pub(crate) struct GateAdmission {
    pending: [Option<Event>; 12],
}

impl GateAdmission {
    pub(crate) fn admit(
        &mut self,
        at: Instant,
        gate_open: bool,
        probabilities: &[f32; 12],
        threshold: f32,
        events: impl IntoIterator<Item = Event>,
    ) -> Vec<Event> {
        for (pc, pending) in self.pending.iter_mut().enumerate() {
            if pending.is_some_and(|event| {
                !at.checked_duration_since(event.at)
                    .is_some_and(|age| age <= Duration::from_millis(96))
                    || probabilities[pc] < threshold
            }) {
                *pending = None;
            }
        }
        let mut admitted = Vec::new();
        for event in events {
            self.pending[event.pc] = None;
            if gate_open {
                admitted.push(event);
            } else if probabilities[event.pc] >= threshold {
                // Weak rescue keeps its existing immediate gate policy.
                self.pending[event.pc] = Some(event);
            }
        }
        if gate_open {
            admitted.extend(self.pending.iter_mut().filter_map(Option::take));
        }
        admitted
    }
}

#[derive(Default)]
pub struct Mailbox {
    pub enabled: bool,
    pub generation: u64,
    pub frame: u64,
    pub probabilities: [f32; 12],
    pub events: VecDeque<Event>,
    pub invalidated_at: Option<Instant>,
}

impl Mailbox {
    pub fn invalidate(&mut self) {
        self.generation += 1;
        self.frame = 0;
        self.events.clear();
        self.probabilities = [0.0; 12];
        self.invalidated_at = Some(Instant::now());
    }
}

pub struct Detector {
    session: Session,
    threshold: f32,
    spectra: Vec<Spectrum>,
    audio: VecDeque<f32>,
    history: VecDeque<[f32; FEATURES]>,
    /// Pitch-blind attack detection, measured alongside the model and not yet
    /// consulted by anything. See `flux.rs` for what it is for.
    flux: flux::Flux,
    last_flux: (f32, bool),
    flux_events: bool,
    /// Pitch classes the last hop reported through flux rather than the model,
    /// so a trace can tell the two apart.
    pub(crate) from_flux: Vec<usize>,
    pub frame: u64,
    armed: [bool; 12],
    peaks: [f32; 12],
    rescue: Option<rescue::Rescue>,
    rescued: [bool; 12],
}

struct Spectrum {
    fft: Arc<dyn Fft<f64>>,
    window: Vec<f64>,
    scale: f64,
    buffer: Vec<Complex<f64>>,
    scratch: Vec<Complex<f64>>,
}

impl Spectrum {
    fn new(size: usize) -> Self {
        let fft = FftPlanner::new().plan_fft_forward(size);
        let window: Vec<_> = (0..size)
            .map(|i| {
                0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (size - 1) as f64).cos())
            })
            .collect();
        Self {
            scale: 2.0 / window.iter().sum::<f64>(),
            window,
            buffer: vec![Complex::default(); size],
            scratch: vec![Complex::default(); fft.get_inplace_scratch_len()],
            fft,
        }
    }
}

// The training cache rounds features to binary16. Keep that contract without
// introducing a dependency just for positive, finite feature quantization.
fn cache_precision(value: f64) -> f32 {
    let step = if value < 2f64.powi(-14) {
        2f64.powi(-24)
    } else {
        2f64.powi(value.log2().floor() as i32 - 10)
    };
    ((value / step).round_ties_even() * step) as f32
}

impl Detector {
    pub fn new(path: &str) -> Result<Self> {
        let (session, combined) = crate::onnx_model::load(path, crate::onnx_model::Branch::Onset)?;
        let threshold = crate::onnx_model::onset_threshold(&session, combined)?;
        ensure!(
            session.inputs.len() == 1
                && session.inputs[0].name == "short_features"
                && session.outputs.iter().any(|o| o.name == "onset_logits"),
            "Not a short onset model: {path}"
        );
        Ok(Self {
            session,
            threshold,
            spectra: vec![Spectrum::new(1024), Spectrum::new(2048)],
            audio: VecDeque::from(vec![0.0; 2048]),
            history: VecDeque::from(vec![[0.0; FEATURES]; HISTORY - 1]),
            flux: flux::Flux::with_deviations(flux_deviations().unwrap_or(flux::DEVIATIONS)),
            last_flux: (0.0, false),
            flux_events: flux_enabled(),
            from_flux: Vec::new(),
            frame: 0,
            armed: [true; 12],
            peaks: [0.0; 12],
            rescue: (std::env::var("SOLITITO_ONSET_RESCUE").as_deref() == Ok("1"))
                .then(rescue::Rescue::new),
            rescued: [false; 12],
        })
    }

    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    pub fn process_hop(&mut self, samples: &[f32]) -> Result<([f32; 12], Vec<usize>, f32)> {
        ensure!(
            samples.len() == HOP && samples.iter().all(|v| v.is_finite()),
            "Invalid onset audio hop"
        );
        self.audio.drain(..HOP);
        self.audio.extend(samples);
        let audio = self.audio.make_contiguous();
        let rms = (audio.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / 2048.0).sqrt() as f32;
        let mut features = [0.0; FEATURES];
        // The same magnitudes the features are built from, kept linear: the
        // flux detector reads them before the log compression, which is what
        // lets it see a small absolute rise.
        let mut magnitudes = [0.0f32; FEATURES];
        let mut offset = 0;
        for spectrum in &mut self.spectra {
            let size = spectrum.window.len();
            for (i, sample) in audio[2048 - size..].iter().enumerate() {
                spectrum.buffer[i] = Complex::new(*sample as f64 * spectrum.window[i], 0.0);
            }
            spectrum
                .fft
                .process_with_scratch(&mut spectrum.buffer, &mut spectrum.scratch);
            let bins = size / 4 + 1;
            for (i, value) in spectrum.buffer[..bins].iter().enumerate() {
                let magnitude = value.norm() * spectrum.scale;
                magnitudes[offset + i] = magnitude as f32;
                features[offset + i] =
                    cache_precision((1000.0 * magnitude).ln_1p() / 1001f64.ln());
            }
            offset += bins;
        }
        self.last_flux = self.flux.push(&magnitudes);
        if self.history.len() == HISTORY {
            self.history.pop_front();
        }
        self.history.push_back(features);
        let time = self.history.len();
        let mut flat = vec![0.0; FEATURES * time];
        for (t, row) in self.history.iter().enumerate() {
            for (bin, &value) in row.iter().enumerate() {
                flat[bin * time + t] = value;
            }
        }
        let input = Value::from_array((vec![1i64, FEATURES as i64, time as i64], flat))?;
        let output = self.session.run(ort::inputs!["short_features" => input])?;
        let (shape, logits) = output["onset_logits"].try_extract_tensor::<f32>()?;
        ensure!(
            shape.as_ref() == [1, 12, time as i64],
            "Invalid onset output shape: {shape:?}"
        );
        let mut probabilities = [0.0; 12];
        for (pc, p) in probabilities.iter_mut().enumerate() {
            *p = 1.0 / (1.0 + (-logits[pc * time + time - 1]).exp());
        }
        ensure!(
            probabilities.iter().all(|p| p.is_finite()),
            "Nonfinite onset output"
        );
        let confirmations = if let Some(rescue) = &mut self.rescue {
            let magnitude: Vec<_> = self.spectra[1].buffer[..513]
                .iter()
                .map(|v| v.norm())
                .collect();
            rescue.confirm_with_threshold(&magnitude, &probabilities, self.threshold)
        } else {
            [None; 12]
        };
        let mut events = confirmed_crossings(
            &probabilities,
            &confirmations,
            &mut self.armed,
            &mut self.peaks,
            self.threshold,
        );
        self.rescued = [false; 12];
        for &pc in &events {
            self.rescued[pc] = probabilities[pc] < self.threshold;
        }
        // The experiment. It joins the ordinary event stream rather than
        // running beside it, so the boundary, lifetime and one-credit-per-event
        // rules that `state.rs` applies to the model's events apply to these
        // unchanged. The model's own events are never removed or altered.
        self.from_flux.clear();
        if self.flux_events && self.last_flux.1 {
            if let Some(pc) = flux::harmonic_pitch(&magnitudes[257..]) {
                if !events.contains(&pc) {
                    events.push(pc);
                    self.from_flux.push(pc);
                }
            }
        }
        self.frame += 1;
        Ok((probabilities, events, rms))
    }
}

#[cfg(test)]
fn crossings(values: &[f32; 12], armed: &mut [bool; 12], peaks: &mut [f32; 12]) -> Vec<usize> {
    confirmed_crossings(values, &[None; 12], armed, peaks, 0.8)
}

fn confirmed_crossings(
    values: &[f32; 12],
    confirmations: &[Option<f32>; 12],
    armed: &mut [bool; 12],
    peaks: &mut [f32; 12],
    threshold: f32,
) -> Vec<usize> {
    let mut events = Vec::new();
    for pc in 0..12 {
        if armed[pc] && (values[pc] >= threshold || confirmations[pc].is_some()) {
            armed[pc] = false;
            peaks[pc] = values[pc].max(confirmations[pc].unwrap_or(0.0));
            events.push(pc);
        } else if values[pc] < (0.3 * peaks[pc]).max(0.1) {
            armed[pc] = true;
        }
    }
    events
}

fn events_for_trace(flags: &[bool; 12]) -> Vec<usize> {
    (0..12).filter(|&pc| flags[pc]).collect()
}

/// Confirmation must not make audio captured before a round boundary look new.
fn event_time(at: Instant, rescued: bool) -> Instant {
    if rescued {
        at - Duration::from_millis(16)
    } else {
        at
    }
}

/// Same causal interpolation as the trainer, invariant to callback boundaries.
#[derive(Default)]
pub struct Resampler {
    input: VecDeque<f32>,
    base: u64,
    received: u64,
    produced: u64,
}
impl Resampler {
    pub fn push(&mut self, samples: &[f32], rate: u32) -> Vec<f32> {
        self.input
            .extend(samples.iter().map(|&v| if v.is_finite() { v } else { 0.0 }));
        self.received += samples.len() as u64;
        let total = self.received * SR / rate as u64;
        let mut output = Vec::with_capacity((total - self.produced) as usize);
        while self.produced < total {
            let position = self.produced as f64 * (rate as f64 / SR as f64)
                - if rate == SR as u32 { 0.0 } else { 1.0 };
            let lower = position.floor() as i64;
            let fraction = position - lower as f64;
            let sample = |i: i64| -> f64 {
                if i < 0 {
                    0.0
                } else {
                    self.input[(i as u64 - self.base) as usize] as f64
                }
            };
            let a = sample(lower);
            let b = if fraction == 0.0 {
                a
            } else {
                sample(lower + 1)
            };
            output.push((a + fraction * (b - a)) as f32);
            self.produced += 1;
        }
        let next = (self.produced as f64 * rate as f64 / SR as f64 - 1.0)
            .floor()
            .max(0.0) as u64;
        let keep_from = next.min(self.received);
        let count = keep_from.saturating_sub(self.base) as usize;
        self.input.drain(..count);
        self.base = keep_from;
        output
    }
}

/// Optional diagnostic copy of the exact 16 kHz hops seen by the detector.
/// A discontinuous capture is explicitly unsuitable for uninterrupted replay.
struct Capture {
    writer: Option<hound::WavWriter<std::io::BufWriter<std::fs::File>>>,
    path: std::path::PathBuf,
    frames: u64,
    discontinuities: u64,
    failed: bool,
}
impl Capture {
    fn create(prefix: &std::path::Path, generation: u64) -> Result<Self> {
        let mut name = prefix.as_os_str().to_os_string();
        name.push(format!("-g{generation}.wav"));
        let path = std::path::PathBuf::from(name);
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .with_context(|| format!("Cannot create onset recording {}", path.display()))?;
        let writer = hound::WavWriter::new(
            std::io::BufWriter::new(file),
            hound::WavSpec {
                channels: 1,
                sample_rate: SR as u32,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            },
        )?;
        eprintln!("RISE_CAPTURE start generation={generation} path={} rate={SR} hop={HOP} format=float32 gain=1", path.display());
        Ok(Self {
            writer: Some(writer),
            path,
            frames: 0,
            discontinuities: 0,
            failed: false,
        })
    }

    fn write_hop(&mut self, hop: &[f32]) -> Result<()> {
        if let Some(writer) = self.writer.as_mut() {
            for &sample in hop {
                writer.write_sample(sample)?;
            }
            self.frames += 1;
            // Keep the header readable even if the process is interrupted.
            if self.frames % 64 == 0 {
                writer.flush()?;
            }
        }
        Ok(())
    }

    fn record(&mut self, hop: &[f32]) {
        if self.failed {
            return;
        }
        if let Err(error) = self.write_hop(hop) {
            self.failed = true;
            eprintln!(
                "RISE_CAPTURE error path={} error={error:#}",
                self.path.display()
            );
        }
    }

    fn discontinuity(&mut self) {
        self.discontinuities += 1;
        eprintln!(
            "RISE_CAPTURE discontinuity after_frame={} path={} replay_contiguous=false",
            self.frames,
            self.path.display()
        );
    }
}
impl Drop for Capture {
    fn drop(&mut self) {
        if let Some(writer) = self.writer.take() {
            if let Err(error) = writer.finalize() {
                self.failed = true;
                eprintln!(
                    "RISE_CAPTURE finalize_error path={} error={error:#}",
                    self.path.display()
                );
            }
        }
        eprintln!("RISE_CAPTURE end path={} frames={} discontinuities={} replay_contiguous={} complete={}",
            self.path.display(), self.frames, self.discontinuities,
            self.discontinuities == 0, !self.failed);
    }
}

struct Packet {
    samples: Vec<f32>,
    start: u64,
    end: Instant,
}
pub struct Input {
    sender: Option<SyncSender<Packet>>,
    sent: u64,
    recording_worker: Option<std::thread::JoinHandle<()>>,
}
impl Input {
    pub fn push(&mut self, samples: &[f32]) {
        let packet = Packet {
            samples: samples.to_vec(),
            start: self.sent,
            end: Instant::now(),
        };
        self.sent += samples.len() as u64;
        // Never wait for inference in the audio callback. A missing packet is
        // detected by its sample position and resets the worker's context.
        if let Some(sender) = &self.sender {
            let _ = sender.try_send(packet);
        }
    }
}

impl Drop for Input {
    fn drop(&mut self) {
        // Recording must finish before normal stream/application shutdown.
        // Only diagnostics join; normal audio callbacks never wait for a worker.
        self.sender.take();
        if let Some(worker) = self.recording_worker.take() {
            if worker.join().is_err() {
                eprintln!("RISE_CAPTURE worker panicked; capture incomplete");
            }
        }
    }
}

pub fn start(shared: Arc<Mutex<crate::audio::AudioAnalysis>>, rate: u32) -> Result<Option<Input>> {
    ensure!(rate >= 8000, "Rise requires sample rate >=8000 Hz");
    let use_rise = enabled();
    let generation = {
        let mut state = shared
            .lock()
            .map_err(|_| anyhow::anyhow!("Audio state poisoned"))?;
        state.rise.invalidate();
        state.rise.enabled = use_rise;
        state.rise.generation
    };
    if !use_rise {
        println!("Onset detector: legacy");
        return Ok(None);
    }
    let path = model_path()?;
    let mut detector =
        Detector::new(&path).with_context(|| format!("Cannot load rise model {path}"))?;
    // Validate dimensions by running it before opening the audio stream.
    detector.process_hop(&[0.0; HOP])?;
    detector.history = VecDeque::from(vec![[0.0; FEATURES]; HISTORY - 1]);
    detector.frame = 0;
    detector.armed = [true; 12];
    detector.peaks = [0.0; 12];
    if let Some(rescue) = &mut detector.rescue {
        rescue.reset();
    }
    let threshold = detector.threshold();
    let (sender, receiver) = sync_channel::<Packet>(8);
    let trace = std::env::var("SOLITITO_ONSET_TRACE").is_ok();
    let mut capture = std::env::var_os("SOLITITO_ONSET_RECORD")
        .map(|prefix| Capture::create(std::path::Path::new(&prefix), generation))
        .transpose()?;
    let recording = capture.is_some();
    let worker = std::thread::Builder::new()
        .name("solitito-onset".into())
        .spawn(move || {
            let mut resampler = Resampler::default();
            let mut pending = VecDeque::new();
            let mut expected = 0;
            let mut sequence = 0;
            let mut admission = GateAdmission::default();
            while let Ok(packet) = receiver.recv() {
                if packet.start != expected || packet.end.elapsed() > Duration::from_millis(100) {
                    if let Some(capture) = &mut capture { capture.discontinuity(); }
                    // No discontinuity may manufacture an onset from old audio.
                    resampler = Resampler::default();
                    pending.clear();
                    admission = GateAdmission::default();
                    detector.audio.make_contiguous().fill(0.0);
                    detector.history = VecDeque::from(vec![[0.0; FEATURES]; HISTORY - 1]);
                    detector.flux = flux::Flux::with_deviations(flux_deviations().unwrap_or(flux::DEVIATIONS));
        detector.flux = flux::Flux::with_deviations(flux_deviations().unwrap_or(flux::DEVIATIONS));
                    detector.armed = [false; 12];
                    if let Some(rescue) = &mut detector.rescue { rescue.reset(); }
                    if let Ok(mut state) = shared.lock() {
                        if state.rise.generation != generation {
                            return;
                        }
                        state.rise.events.clear();
                        state.rise.invalidated_at = Some(Instant::now());
                        state.rise.probabilities = [0.0; 12];
                    }
                    eprintln!("Rise: audio discontinuity; discarded pending onsets");
                }
                expected = packet.start + packet.samples.len() as u64;
                if packet.end.elapsed() > Duration::from_millis(100) {
                    continue;
                }
                pending.extend(resampler.push(&packet.samples, rate));
                while pending.len() >= HOP {
                    let hop: Vec<_> = pending.drain(..HOP).collect();
                    let at = packet.end - Duration::from_secs_f64(pending.len() as f64 / SR as f64);
                    match detector.process_hop(&hop) {
                        Ok((probabilities, pcs, rms)) => {
                            sequence += 1;
                            if let Some(capture) = &mut capture { capture.record(&hop); }
                            if let Ok(mut state) = shared.lock() {
                                if state.rise.generation != generation {
                                    return;
                                }
                                let gate = state.noise_gate;
                                let mailbox = &mut state.rise;
                                mailbox.frame = sequence;
                                mailbox.probabilities = probabilities;
                                // Include subthreshold answers: a crossings-only log cannot
                                // distinguish a weak model response from a lost event.
                                if trace {
                                    eprintln!("RISE_FRAME generation={generation} frame={sequence} pcs={pcs:?} probabilities={probabilities:.4?} armed_after={:?} rms={rms:.6} gate={gate:.6} admitted={} from_flux={:?} delivery_ms={:.1}",
                                        detector.armed, rms > gate, detector.from_flux,
                                        at.elapsed().as_secs_f64() * 1000.0);
                                }
                                if trace && detector.rescued.iter().any(|&v| v) {
                                    eprintln!("RISE_RESCUE generation={generation} frame={sequence} pcs={:?} evidence_age_ms=16", events_for_trace(&detector.rescued));
                                }
                                let admitted = admission.admit(
                                    at, rms > gate, &probabilities, threshold,
                                    pcs.into_iter().map(|pc| Event {
                                            frame: sequence,
                                            pc,
                                            at: event_time(at, detector.rescued[pc]),
                                    }),
                                );
                                for event in admitted {
                                    if trace && event.frame != sequence {
                                        eprintln!("RISE_GATE_RELEASE generation={generation} frame={sequence} event_frame={} pc={} evidence_age_ms={:.1}",
                                            event.frame, event.pc, at.duration_since(event.at).as_secs_f64() * 1000.0);
                                    }
                                    if mailbox.events.len() == 256 {
                                        mailbox.events.pop_front();
                                    }
                                    mailbox.events.push_back(event);
                                }
                            }
                        }
                        Err(error) => {
                            eprintln!("Rise inference failed: {error:#}");
                            if let Ok(mut state) = shared.lock() {
                                if state.rise.generation == generation {
                                    state.rise.events.clear();
                                    state.rise.probabilities = [0.0; 12];
                                    state.rise.invalidated_at = Some(Instant::now());
                                }
                            }
                            return;
                        }
                    }
                }
            }
        })?;
    println!(
        "Onset detector: rise ({path}), threshold {threshold}, hop 16 ms; weak rescue={}, flux={}",
        std::env::var("SOLITITO_ONSET_RESCUE").as_deref() == Ok("1"),
        match flux_deviations() {
            Some(d) => format!("on, {d} deviations"),
            None => "off".into(),
        }
    );
    Ok(Some(Input {
        sender: Some(sender),
        sent: 0,
        recording_worker: recording.then_some(worker),
    }))
}

/// WAV replay uses the first input channel, as a selected live input does.
pub fn read_wav(path: &str) -> Result<(Vec<f32>, u32)> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    ensure!(
        spec.sample_rate >= 8000 && spec.channels > 0,
        "Invalid WAV format"
    );
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = 2f64.powi(spec.bits_per_sample as i32 - 1) as f32;
            reader
                .samples::<i32>()
                .map(|v| v.map(|v| v as f32 / scale))
                .collect::<Result<_, _>>()?
        }
    };
    Ok((
        samples
            .chunks_exact(spec.channels as usize)
            .map(|s| s[0])
            .collect(),
        spec.sample_rate,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_wait_expires_on_time_probability_drop_or_reset() {
        let start = Instant::now();
        let event = Event { frame: 7, pc: 4, at: start };
        for cause in ["time", "probability", "reset"] {
            let mut gate = GateAdmission::default();
            let high = [0.9; 12];
            assert!(gate.admit(start, false, &high, 0.8, [event]).is_empty());
            let mut at = start + Duration::from_millis(32);
            match cause {
                "time" => at = start + Duration::from_millis(112),
                "probability" => {
                    gate.admit(start + Duration::from_millis(16), false, &[0.7; 12], 0.8, []);
                }
                _ => gate = GateAdmission::default(),
            }
            assert!(gate.admit(at, true, &high, 0.8, []).is_empty(), "{cause}");
        }
    }

    #[test]
    fn gate_wait_keeps_polyphonic_events_and_never_repeats_held_probabilities() {
        let mut gate = GateAdmission::default();
        let start = Instant::now();
        let high = [0.9; 12];
        let events = [0, 4, 7].map(|pc| Event { frame: 1, pc, at: start });
        assert!(gate.admit(start, false, &high, 0.8, events).is_empty());
        let admitted = gate.admit(start + Duration::from_millis(32), true, &high, 0.8, []);
        assert_eq!(admitted.iter().map(|e| e.pc).collect::<Vec<_>>(), [0, 4, 7]);
        assert!(admitted.iter().all(|e| e.frame == 1 && e.at == start));
        for hop in 3..100 {
            assert!(gate.admit(start + Duration::from_millis(hop * 16), true,
                &high, 0.8, []).is_empty());
        }
        let at = start + Duration::from_secs(2);
        assert_eq!(gate.admit(at, true, &high, 0.8,
            [Event { frame: 125, pc: 0, at }]).len(), 1);
    }

    #[test]
    fn weak_rescue_and_already_open_gate_keep_their_existing_policy() {
        let start = Instant::now();
        let weak = Event { frame: 1, pc: 7, at: start - Duration::from_millis(16) };
        let mut gate = GateAdmission::default();
        assert!(gate.admit(start, false, &[0.7; 12], 0.8, [weak]).is_empty());
        assert!(gate.admit(start + Duration::from_millis(16), true,
            &[0.9; 12], 0.8, []).is_empty());
        let admitted = gate.admit(start, true, &[0.7; 12], 0.8, [weak]);
        assert_eq!(admitted.len(), 1);
        assert_eq!(admitted[0].at, weak.at);
    }

    #[test]
    #[ignore = "diagnostic: requires SOLITITO_GATE_INPUT and SOLITITO_GATE_OUTPUT"]
    fn replay_gate_admission() -> Result<()> {
        let input = std::fs::read(std::env::var("SOLITITO_GATE_INPUT")?)?;
        let document: serde_json::Value = serde_json::from_slice(&input)?;
        let rows = document["rows"].as_array().context("missing rows")?;
        let threshold: f32 = std::env::var("SOLITITO_GATE_THRESHOLD")
            .unwrap_or_else(|_| "0.8".into()).parse()?;
        let gate_db: f32 = std::env::var("SOLITITO_GATE_DB")
            .unwrap_or_else(|_| "-34".into()).parse()?;
        let level = 10f32.powf(gate_db / 20.0);
        let base = Instant::now();
        let mut admission = GateAdmission::default();
        let mut baseline = Vec::new();
        let mut candidate = Vec::new();
        for (index, row) in rows.iter().enumerate() {
            let t = row["t"].as_f64().context("missing time")?;
            let at = base + Duration::from_secs_f64(t);
            let probabilities: [f32; 12] = serde_json::from_value(row["probabilities"].clone())?;
            let pcs: Vec<usize> = serde_json::from_value(row["pcs"].clone())?;
            let rescued: [bool; 12] = serde_json::from_value(row["rescued"].clone())?;
            let open = row["rms"].as_f64().context("missing rms")? > level as f64;
            let events: Vec<_> = pcs.into_iter().map(|pc| Event {
                frame: index as u64 + 1, pc, at: event_time(at, rescued[pc]),
            }).collect();
            let record = |event: Event| serde_json::json!({
                "frame": event.frame, "pc": event.pc,
                "t": event.at.duration_since(base).as_secs_f64(), "delivered": t,
            });
            if open { baseline.extend(events.iter().copied().map(record)); }
            candidate.extend(admission.admit(at, open, &probabilities, threshold, events)
                .into_iter().map(record));
        }
        std::fs::write(std::env::var("SOLITITO_GATE_OUTPUT")?,
            serde_json::to_vec(&serde_json::json!({"baseline": baseline, "candidate": candidate}))?)?;
        Ok(())
    }

    #[test]
    fn weak_and_strong_evidence_share_one_latch() {
        let mut armed = [true; 12];
        let mut peaks = [0.0; 12];
        let mut values = [0.0; 12];
        let mut confirmations = [None; 12];
        values[4] = 0.65;
        confirmations[4] = Some(0.72);
        assert_eq!(
            confirmed_crossings(&values, &confirmations, &mut armed, &mut peaks, 0.8),
            [4]
        );
        values[4] = 0.95;
        assert!(confirmed_crossings(&values, &[None; 12], &mut armed, &mut peaks, 0.8).is_empty());
        values[4] = 0.01;
        assert!(confirmed_crossings(&values, &[None; 12], &mut armed, &mut peaks, 0.8).is_empty());
        values[4] = 0.9;
        assert_eq!(
            confirmed_crossings(&values, &[None; 12], &mut armed, &mut peaks, 0.8),
            [4]
        );
    }

    #[test]
    fn selected_threshold_keeps_the_repetition_latch() {
        for threshold in [0.5, 0.7, 0.9] {
            let mut armed = [true; 12];
            let mut peaks = [0.0; 12];
            let mut values = [0.0; 12];
            values[7] = threshold - 0.01;
            assert!(
                confirmed_crossings(&values, &[None; 12], &mut armed, &mut peaks, threshold)
                    .is_empty()
            );
            values[7] = threshold;
            assert_eq!(
                confirmed_crossings(&values, &[None; 12], &mut armed, &mut peaks, threshold),
                [7]
            );
            for _ in 0..100 {
                assert!(confirmed_crossings(
                    &values,
                    &[None; 12],
                    &mut armed,
                    &mut peaks,
                    threshold
                )
                .is_empty());
            }
            values[7] = 0.01;
            assert!(
                confirmed_crossings(&values, &[None; 12], &mut armed, &mut peaks, threshold)
                    .is_empty()
            );
            values[7] = threshold;
            assert_eq!(
                confirmed_crossings(&values, &[None; 12], &mut armed, &mut peaks, threshold),
                [7]
            );
        }
    }

    #[test]
    fn confirmation_does_not_move_a_pre_boundary_attack_into_the_next_round() {
        let candidate = Instant::now();
        let boundary = candidate + Duration::from_millis(8);
        let confirmation = candidate + Duration::from_millis(16);
        assert!(event_time(confirmation, true) < boundary);
        assert_eq!(event_time(confirmation, false), confirmation);
    }

    #[test]
    fn diagnostic_capture_preserves_samples_and_never_overwrites() -> Result<()> {
        let dir = std::env::temp_dir().join(format!(
            "solitito-capture-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        ));
        std::fs::create_dir(&dir)?;
        let prefix = dir.join("trace");
        let samples: Vec<f32> = (0..HOP).map(|i| i as f32 / 64.0 - 2.0).collect();
        let path;
        {
            let mut capture = Capture::create(&prefix, 2)?;
            path = capture.path.clone();
            capture.record(&samples);
            assert!(!capture.failed);
            assert!(Capture::create(&prefix, 2).is_err());
            capture.discontinuity();
            assert_eq!(capture.discontinuities, 1);
        }
        let (read, rate) = read_wav(path.to_str().context("invalid test path")?)?;
        assert_eq!(rate, SR as u32);
        assert_eq!(read, samples); // Includes values outside [-1, 1]; no clipping.
        let next = Capture::create(&prefix, 3)?;
        assert_ne!(next.path, path);
        drop(next);
        std::fs::remove_dir_all(dir)?;
        Ok(())
    }

    #[test]
    fn callback_boundaries_do_not_change_resampling() {
        let wave: Vec<_> = (0..192_013).map(|n| (n as f32 * 0.031).sin()).collect();
        for rate in [8000, 16000, 44100, 48000, 96000] {
            let expected = Resampler::default().push(&wave, rate);
            assert_eq!(expected.len(), wave.len() * 16000 / rate as usize);
            for chunk in [1, 127, 1024] {
                let mut stream = Resampler::default();
                let actual: Vec<_> = wave
                    .chunks(chunk)
                    .flat_map(|s| stream.push(s, rate))
                    .collect();
                assert_eq!(actual, expected, "rate={rate}, chunk={chunk}");
            }
        }
    }

    #[test]
    fn interpolation_has_the_trainers_one_sample_delay() {
        let got = Resampler::default().push(&[0.2, 0.4, 0.6, 0.8, 1.0, 0.0], 48000);
        assert_eq!(got, [0.0, 0.6]);
        let same = [0.2, 0.4, 0.6];
        assert_eq!(Resampler::default().push(&same, 16000), same);
    }

    #[test]
    fn plateau_and_polyphony_match_the_offline_latch() {
        let mut armed = [true; 12];
        let mut peaks = [0.0; 12];
        let mut values = [0.0; 12];
        for pc in [0, 4, 7] {
            values[pc] = 0.9;
        }
        assert_eq!(crossings(&values, &mut armed, &mut peaks), [0, 4, 7]);
        for _ in 0..100 {
            assert!(crossings(&values, &mut armed, &mut peaks).is_empty());
        }
        values[7] = 0.2;
        assert!(crossings(&values, &mut armed, &mut peaks).is_empty());
        values[7] = 0.85;
        assert_eq!(crossings(&values, &mut armed, &mut peaks), [7]);
    }

    #[test]
    fn binary16_rounding_includes_subnormals_and_ties() {
        assert_eq!(cache_precision(0.0), 0.0);
        assert_eq!(cache_precision(1.0 + 2f64.powi(-11)), 1.0);
        assert_eq!(cache_precision(1.0 + 3.0 * 2f64.powi(-11)), 1.001953125);
        assert_eq!(cache_precision(2f64.powi(-25)), 0.0);
        assert_eq!(cache_precision(3.0 * 2f64.powi(-25)), 2f32.powi(-23));
    }

    #[test]
    #[ignore = "diagnostic: requires SOLITITO_RISE_WAV and SOLITITO_RISE_OUTPUT"]
    fn export_streaming_predictions() -> Result<()> {
        let (audio, rate) = read_wav(&std::env::var("SOLITITO_RISE_WAV")?)?;
        let mut resampler = Resampler::default();
        let mut detector = Detector::new(&model_path()?)?;
        let mut pending = VecDeque::new();
        let mut rows = Vec::new();
        let mut features = Vec::new();
        let mut timings = Vec::new();
        for chunk in audio.chunks(511) {
            pending.extend(resampler.push(chunk, rate));
            while pending.len() >= HOP {
                let hop: Vec<_> = pending.drain(..HOP).collect();
                let start = Instant::now();
                let (probabilities, pcs, rms) = detector.process_hop(&hop)?;
                timings.push(start.elapsed().as_secs_f64() * 1000.0);
                rows.push(serde_json::json!({"t": detector.frame as f64 * 0.016,
                    "probabilities": probabilities, "pcs": pcs, "rms": rms,
                    "flux": detector.last_flux.0, "flux_attack": detector.last_flux.1,
                    "from_flux": detector.from_flux.clone(),
                    "rescued": detector.rescued}));
                // Enough frames to cover startup and full receptive history.
                if detector.frame <= 100 {
                    features.push(
                        detector
                            .history
                            .back()
                            .context("missing features")?
                            .to_vec(),
                    );
                }
            }
        }
        std::fs::write(
            std::env::var("SOLITITO_RISE_OUTPUT")?,
            serde_json::to_vec(&serde_json::json!({
                "rows": rows, "features": features, "compute_ms": timings,
            }))?,
        )?;
        Ok(())
    }
}
