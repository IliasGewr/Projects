
# coding: utf-8

# In[ ]:


from encoder.visualizations import Visualizations
from encoder.data_objects import StudyVerificationDataLoader, StudyVerificationDataset
from encoder.params_data import *
from encoder.model import BioDataomeEncoder
from utils.profiler import Profiler
from pathlib import Path
import torch

def sync(device: torch.device):
    # FIXME
    return 
    # For correct profiling (cuda operations are async)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

def train(run_id: str, clean_data_root: Path, models_dir: Path, save_every: int,
          backup_every: int,  force_restart: bool):
    # Create a dataset and a dataloader
    dataset = StudyVerificationDataset(clean_data_root)
    loader = StudyVerificationDataLoader(
        dataset,
        studies_per_batch,
        samples_per_study,
        num_workers=8,
    )
    
    # Setup the device on which to run the forward pass and the loss. These can be different, 
    # because the forward pass is faster on the GPU whereas the loss is often (depending on your
    # hyperparameters) faster on the CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # FIXME: currently, the gradient is None if loss_device is cuda
    loss_device = torch.device("cpu")
    
    # Create the model and the optimizer
    model = BioDataomeEncoder(device, loss_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_init)
    init_step = 1
    
    # Configure file path for the model
    state_fpath = models_dir.joinpath(run_id + ".pt")
    backup_dir = models_dir.joinpath(run_id + "_backups")

    # Load any existing model
    if not force_restart:
        if state_fpath.exists():
            print("Found existing model \"%s\", loading it and resuming training." % run_id)
            checkpoint = torch.load(state_fpath)
            init_step = checkpoint["step"]
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            optimizer.param_groups[0]["lr"] = learning_rate_init
        else:
            print("No model \"%s\" found, starting training from scratch." % run_id)
    else:
        print("Starting the training from scratch.")
    model.train()
    
#     # Initialize the visualization environment
#     vis = Visualizations(run_id, vis_every, server=visdom_server, disabled=no_visdom)
#     vis.log_dataset(dataset)
#     vis.log_params()
#     device_name = str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
#     vis.log_implementation({"Device": device_name})
    
    # Training loop
    profiler = Profiler(summarize_every=10, disabled=False)
    for step, StudyBatch in enumerate(loader, init_step):
        profiler.tick("Blocking, waiting for batch (threaded)")
        
        # Forward pass
        inputs = torch.from_numpy(StudyBatch.data).to(device)
        
        inputs =inputs[:,:,1:1000].float()
        inputs = torch.nan_to_num(inputs, nan=0.0, posinf=0.0, neginf=0.0)
        inputs=torch.nn.functional.normalize(inputs, p=2.0, dim=0, eps=1e-12, out=inputs)
        inputs=inputs[:,:,:input_shape]
        inputs=torch.reshape(inputs, (batch_size,input_shape))
        
        sync(device)
        profiler.tick("Data to %s" % device)
        outputs = model(inputs)
        rec_loss_train = model.loss_fn2(inputs, outputs)
        
        sync(device)
        profiler.tick("Forward pass")
        
        embeds0 = model.encoder_hidden_layer1(inputs)
        embeds0 = torch.relu(embeds0)
        embeds1 = model.encoder_hidden_layer2(embeds0)
        embeds1 = torch.relu(embeds1)
        embeds2 = model.encoder_output_layer(embeds1)
        embeds2 = torch.nn.functional.silu(embeds2)
        
        #Save embeddings for visualitation
        
        
        embeds2 = embeds2 / torch.norm(embeds2, dim=1, keepdim=True)
         
             
        
        embeds=torch.reshape(embeds2,
(studies_per_batch,samples_per_study,layer3_size))
        
        loss_sim_train=model.loss(embeds)
        loss_sum_train = 0.003*loss_sim_train+rec_loss_train
        
        sync(loss_device)
        profiler.tick("Loss")

        # Backward pass
        model.zero_grad()
        loss_sum_train.backward()
        profiler.tick("Backward pass")
        model.do_gradient_ops()
        optimizer.step()
        profiler.tick("Parameter update")
        
        print("Train loss:",loss_sum_train.item(),"Reconstruction loss:",rec_loss_train.item(),"Similarity loss:",loss_sim_train.item(),"Epoch:",step)
        
        
        # Update visualizations
        # learning_rate = optimizer.param_groups[0]["lr"]
        #vis.update(loss.item(), eer, step)
        
        # Draw projections and save them to the backup folder
#         if umap_every != 0 and step % umap_every == 0:
#             print("Drawing and saving projections (step %d)" % step)
#             backup_dir.mkdir(exist_ok=True)
#             projection_fpath = backup_dir.joinpath("%s_umap_%06d.png" % (run_id, step))
#             embeds = embeds.detach().cpu().numpy()
#             vis.draw_projections(embeds, utterances_per_speaker, step, projection_fpath)
#             vis.save()

        # Overwrite the latest version of the model
        if save_every != 0 and step % save_every == 0:
            print("Saving the model (step %d)" % step)
            torch.save({
                "step": step + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, state_fpath)
            
        # Make a backup
        if backup_every != 0 and step % backup_every == 0:
            print("Making a backup (step %d)" % step)
            backup_dir.mkdir(exist_ok=True)
            backup_fpath = backup_dir.joinpath("%s_bak_%06d.pt" % (run_id, step))
            torch.save({
                "step": step + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, backup_fpath)
            
        profiler.tick("Extras (visualizations, saving)")

