class VoxCPMSynthesizer(BaseSynthesizer):
    """Synthesizer for VoxCPM models with streaming and emotion control support."""
    
    def __init__(
        self,
        model: Any,
        model_version: str,
        voices_path: str,
        voice_manager: VoiceManager,
        model_type: str = "voxcpm",
        device: str = "cuda",
        **kwargs
    ):
        super().__init__(model_type, device, **kwargs)
        self.model = model
        self.model_version = model_version
        self.voices_path = voices_path
        self.voice, self.speaker = self._load_voice_model()
        self.voice_manager = voice_manager
        self.model_path = os.path.join("./data/models", f"voxcpm_{model_version}.pth")
        self._check_model_exists()
    
    def _check_model_exists(self):
        """Verify that the model file exists at the specified path."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
