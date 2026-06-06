# Thai Call Center ASR Technique Summary

## Reference Notebooks

- `notebook/call_center_asr/ref/gold_thai_asr_whisper_pipeline.ipynb`
- `notebook/call_center_asr/ref/gold_thai_asr_whisper_pipeline_tboat.ipynb`

The two notebooks describe the same Thai ASR workflow. The `tboat` version adds Codex guide notes around the original cells.

## Core Technique

The pipeline is a Whisper-based automatic speech recognition system for Thai call-center audio. It uses three practical stages:

1. Inspect and normalize audio input.
2. Optionally reduce noise before recognition.
3. Run a Thai Whisper model in batches and write a submission CSV.

## Audio Handling

- Input audio is loaded from a folder of `.wav` files.
- The reference notebooks use `librosa`, `soundfile`, and `torchaudio` for waveform loading, resampling, and temporary WAV writing.
- Whisper inference is chunked with `chunk_length_s=20`, which helps process longer calls without feeding the full waveform to the model at once.
- The production-style path normalizes audio for model expectations:
  - Resample to 48 kHz before a DFSMN denoiser.
  - Resample back to 16 kHz before Whisper.

## Noise Reduction

The references test two denoising approaches:

- Spectral noise reduction with `noisereduce.reduce_noise(y, sr, prop_decrease=0.8)`.
- Wavelet threshold denoising with `pywt`, using Daubechies-4 (`db4`) and a median absolute deviation threshold.

The notebook also sketches a stronger production denoising option with ModelScope acoustic noise suppression using a DFSMN model. That path is useful when a compatible model is available locally, but it is more environment-dependent than the pure Python denoise path.

## ASR Models

The references compare two Thai Whisper choices:

- `biodatlab/whisper-th-medium-combined`: smaller Thai Whisper model, faster and lighter.
- Pathumma Whisper Thai Large v3: larger model, expected to be more accurate but needs more GPU memory.

For Thai transcription, the references force Thai decoding with either:

- `generate_kwargs={"language": "<|th|>", "task": "transcribe"}`, or
- `tokenizer.get_decoder_prompt_ids(language="th", task="transcribe")` assigned to `model.config.forced_decoder_ids`.

The second style is convenient for batch processing because it avoids repeating generation options on every call.

## Batch Submission Strategy

The target dataset has `sample_submission.csv` with columns:

- `file_name`
- `text`

The implementation should iterate over `sample_submission.file_name`, resolve each file under `audio_final/audio`, transcribe it, and preserve the same row order in the output CSV.

Important implementation details:

- Load the ASR model once before the loop.
- Use GPU if available, with `float16`/`bfloat16` where supported; fall back to CPU and `float32`.
- Catch per-file exceptions and continue, so one bad audio file does not stop the full submission.
- Write UTF-8 CSV output because Thai text must round-trip correctly.

## Practical Choice for This Repository

The notebook implementation in `notebook/call_center_asr/thai_call_center_asr_submission.ipynb` uses the reference technique but adapts it for the local dataset:

- Reads `dataset/Individual-Test-Thai-Call-Center-ASR/sample_submission.csv`.
- Resolves audio files from `dataset/Individual-Test-Thai-Call-Center-ASR/audio_final/audio`.
- Defaults to `biodatlab/whisper-th-medium-combined` because it can be downloaded from Hugging Face and is less hardware-heavy.
- Keeps Pathumma or local model paths configurable through `ASR_MODEL_ID`.
- Makes denoising optional with `ENABLE_DENOISE`.
- Writes `submission_call_center_asr.csv` with the required `file_name,text` columns.
