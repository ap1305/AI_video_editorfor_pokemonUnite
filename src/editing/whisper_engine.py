import os
from faster_whisper import WhisperModel

class WhisperEngine:
    def __init__(self, model_size: str = "base"):
        """
        Initializes the local Whisper model for dynamic subtitling. 
        'base' provides excellent accuracy for English commentary while maintaining high speed on CPU.
        """
        print(f"[Whisper] Booting local transcription model: {model_size}...")
        # compute_type="int8" reduces memory usage with almost zero accuracy loss
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def generate_viral_subtitles(self, audio_path: str, output_ass_path: str) -> str:
        """
        Transcribes an audio file and generates an Advanced SubStation Alpha (.ass) 
        file formatted specifically for high-retention vertical Shorts.
        
        Args:
            audio_path: Path to the extracted temporary audio file.
            output_ass_path: Where to save the generated .ass file.
            
        Returns:
            The absolute path to the generated subtitle file.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found for transcription: {audio_path}")

        print(f"[Whisper] Transcribing audio and extracting word-level timestamps...")
        
        # word_timestamps=True is the critical parameter for the "karaoke" effect
        segments, info = self.model.transcribe(audio_path, word_timestamps=True)
        
        # The ASS Header defines the viral visual style (Bold, White Text, Thick Black Outline, Centered)
        ass_header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: ViralStyle,Arial,90,&H00FFFFFF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,8,0,5,10,10,800,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        
        with open(output_ass_path, "w", encoding="utf-8") as f:
            f.write(ass_header)
            
            for segment in segments:
                for word in segment.words:
                    start_time = self._format_ass_time(word.start)
                    end_time = self._format_ass_time(word.end)
                    clean_word = word.word.strip().upper() 
                    
                    if clean_word:
                        f.write(f"Dialogue: 0,{start_time},{end_time},ViralStyle,,0,0,0,,{clean_word}\n")

        print(f"[Whisper] Karaoke subtitles generated at: {output_ass_path}")
        return output_ass_path

    def _format_ass_time(self, seconds: float) -> str:
        """Converts float seconds into the strict ASS timestamp format (H:MM:SS.cs)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}:{minutes:02d}:{secs:05.2f}"


if __name__ == "__main__":
    # Local module test
    engine = WhisperEngine()
    # engine.generate_viral_subtitles("temp_audio.wav", "output_subs.ass")