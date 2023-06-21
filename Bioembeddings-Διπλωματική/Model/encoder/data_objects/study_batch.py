import numpy as np
from typing import List
from encoder.data_objects.study import Study

class StudyBatch:
    def __init__(self, studies: List[Study], samples_per_study: int, n_frames: int):
        self.studies = studies
        self.partials = {s: s.random_partial(samples_per_study, n_frames) for s in studies}
        
        # Array of shape (n_speakers * n_utterances, n_frames, mel_n), e.g. for 3 speakers with
        # 4 utterances each of 160 frames of 40 mel coefficients: (12, 160, 40)
        self.data = np.array([frames for s in studies for _, frames, _ in self.partials[s]])
