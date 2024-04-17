import subprocess
from omegaconf import OmegaConf
import hydra
import os

# Set environment variable by reading from secret file
# This is a good practice to avoid exposing your API key
# You can also set this in your bashrc or zshrc file
with open("secret.txt", "r") as f:
    os.environ['WANDB_API_KEY'] = f.read().strip()

# 
@hydra.main(config_name="config.yaml", config_path="./", version_base="1.3")
def main(config):
    print(f"configuration: \n {OmegaConf.to_yaml(config)}")
    
    command = f"""bsub -q {config.bsub.queue} 
                -J {config.bsub.name} 
                -gpu "num={config.bsub.gpu_num}"
                -n {config.bsub.cpu_num}
                -R "rusage[mem={config.bsub.cpu_mem}GB]"
                -R "span[hosts=1]"
                -W 24:00
                -B 
                -N 
                -o lsf_logs/gpu_%J.out
                -e lsf_logs/gpu_%J.err
                -env "all" 
                python3 main.py
                hyper.lr={config.hyper.lr} 
                hyper.epochs={config.hyper.epochs}
                hyper.batch_size={config.hyper.batch_size}
                hyper.n_depth={config.hyper.n_depth}
                hyper.patience={config.hyper.patience}
                hyper.n_neurons={config.hyper.n_neurons}
                hyper.with_skip_connections={config.hyper.with_skip_connections}
                
                data.train_size={config.hyper.train_size}
                data.in_memory={config.hyper.in_memory}
                data.static_test={config.hyper.static_test}
                data.normalize={config.hyper.normalize}
                data.p_flip_horizontal={config.hyper.p_flip_horizontal}
                data.sampling_height={config.hyper.sampling_height}
                data.sampling_width={config.hyper.sampling_width}
                data.random_train_test_split={config.hyper.random_train_test_split}
                data.detector={config.data.detector}
                
                compute.hpc={config.compute.hpc}
                
                miscellaneous.save_as={config.miscellaneous.save_as}
                
                constants.seed={config.constants.seed}
                constants.n_classes={config.constants.n_classes}
                
                wandb.track={config.wandb.track}
                
                """
    command = command.replace("\n", " ")
    command = " ".join(command.split())
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    output, error = process.communicate()
    
    if error:
        print(f"Error: {error}")
    else:
        print(f"Output: {output.decode('utf-8')}")
        
if __name__ == "__main__":
    main()
