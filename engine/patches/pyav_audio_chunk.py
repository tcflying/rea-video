"""Runtime shim: make SaveVideo's audio mux survive a muted / oversized waveform.

Root cause of the ComfyUI node-7 failure in our runs (evidence:
engine/logs/engine.log:23074-23120 for arm A and :28042-28090 for arm B):

    av.error.MemoryError: [Errno 12] Cannot allocate memory
      G:/comfyui/comfy_api/latest/_input_impl/video_types.py:1295  output.mux(audio_stream.encode(frame))
      av/audio/stream.py:41        packets = self.codec_context.encode(frame)
      av/codec/context.py:702      for frame in self._prepare_and_time_rebase_frames_for_encode(frame):
      av/audio/codeccontext.py:31  frames = self.resampler.resample(frame)
      av/audio/resampler.py:127    self.graph.push(frame)
      av/filter/graph.py:296       ctx.push(frame)
      av/filter/context.py:125     err_check(res)          <- av_buffersrc_write_frame -> ENOMEM

SaveVideo hands the audio to the AAC encoder as ONE AudioFrame. When that frame
carries no samples (our graph runs the director with audioMode="mute", so the
waveform is empty) the encoder's internal resampler pushes an empty frame into
the FFmpeg filter graph and the graph's buffersrc fails to allocate -> ENOMEM.
The video track is already on disk at that point, which is why the mp4 exists
but the prompt is marked as failed.

Fix (in-process only, no shared file is modified):
  * a frame with 0 samples is skipped instead of pushed into the encoder,
  * a frame larger than the encoder frame size is split into frame-sized chunks.

The deliverable audio track is produced afterwards by engine/finalize_video.sh
(ffmpeg -f lavfi anullsrc), so skipping the empty frame loses nothing.
"""
import sys
from fractions import Fraction


def install():
    try:
        import av
        import av.audio.stream as ast
    except Exception as exc:  # pragma: no cover
        print("[pyav_audio_chunk] av import failed: %r" % (exc,), file=sys.stderr, flush=True)
        return False

    orig = ast.AudioStream.encode
    if getattr(orig, "_chunked", False):
        return True

    def encode(self, frame=None):
        if frame is None:
            return orig(self, frame)
        try:
            fs = int(self.codec_context.frame_size or 0)
        except Exception:
            fs = 0
        if fs <= 0:
            fs = 1024  # aac frame size
        samples = int(frame.samples)
        print(
            "[pyav_audio_chunk] encode frame samples=%d fmt=%s layout=%s rate=%d fs=%d"
            % (samples, frame.format.name if frame.format else "?",
               frame.layout.name if frame.layout else "?", frame.sample_rate, fs),
            file=sys.stderr,
            flush=True,
        )
        if samples == 0:
            print("[pyav_audio_chunk] empty frame -> skipped", file=sys.stderr, flush=True)
            return []
        if samples <= fs:
            return orig(self, frame)

        data = frame.to_ndarray()
        tb = frame.time_base or Fraction(1, int(frame.sample_rate))
        base_pts = frame.pts or 0
        packets = []
        for start in range(0, samples, fs):
            part = av.AudioFrame.from_ndarray(
                data[:, start:start + fs], format="fltp", layout=frame.layout.name
            )
            part.sample_rate = frame.sample_rate
            part.time_base = tb
            part.pts = base_pts + start
            packets.extend(orig(self, part))
        print(
            "[pyav_audio_chunk] split %d samples -> %d chunks of %d"
            % (samples, -(-samples // fs), fs),
            file=sys.stderr,
            flush=True,
        )
        return packets

    encode._chunked = True
    ast.AudioStream.encode = encode
    print("[pyav_audio_chunk] installed", file=sys.stderr, flush=True)
    return True


install()
