from encoder.data_objects.random_cycler import RandomCycler
from encoder.data_objects.study_batch import StudyBatch
from encoder.data_objects.study import Study
from encoder.params_data import partials_n_frames
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class StudyVerificationDataLoader(DataLoader):
    def __init__(self, dataset, studies_per_batch, samples_per_study, sampler=None, 
                 batch_sampler=None, num_workers=0, pin_memory=False, timeout=0, 
                 worker_init_fn=None):
        self.samples_per_study = samples_per_study

        super().__init__(
            dataset=dataset, 
            batch_size=studies_per_batch, 
            shuffle=False, 
            sampler=sampler, 
            batch_sampler=batch_sampler, 
            num_workers=0,
            collate_fn=self.collate, 
            pin_memory=pin_memory, 
            drop_last=False, 
            timeout=timeout, 
            worker_init_fn=worker_init_fn
        )

    def collate(self, studies):
        return StudyBatch(studies, self.samples_per_study, partials_n_frames)
    
class StudyVerificationDataset(Dataset):
    def __init__(self, datasets_root: Path):
        self.root = datasets_root
        studies_dirs = [f for f in self.root.glob("*") if f.is_dir()]
        if len(studies_dirs) == 0:
            raise Exception("No speakers found. Make sure you are pointing to the directory "
                            "containing all preprocessed speaker directories.")
        self.studies = [Study(study_dir) for study_dir in studies_dirs]
        self.study_cycler = RandomCycler(self.studies)

    def __len__(self):
        return int(1e10)
        
    def __getitem__(self, index):
        return next(self.study_cycler)
    
    def get_logs(self):
        log_string = ""
        for log_fpath in self.root.glob("*.txt"):
            with log_fpath.open("r") as log_file:
                log_string += "".join(log_file.readlines())
        return log_string
    