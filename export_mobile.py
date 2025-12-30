import torch
import torch.utils.mobile_optimizer
from model import GarbageClassifier
import config

def export_to_mobile():
    checkpoint_path = f"{config.CHECKPOINT_DIR}/best_model.ckpt"
    lightning_model = GarbageClassifier.load_from_checkpoint(checkpoint_path)
    lightning_model.eval()
    
    model = lightning_model.model
    model.eval()
    model.cpu() 

    example_input = torch.rand(1, 3, config.IMG_SIZE, config.IMG_SIZE)
    
    traced_script_module = torch.jit.trace(model, example_input)

    optimized_traced_model = torch.utils.mobile_optimizer.optimize_for_mobile(traced_script_module)

    output_path = "garbage_classifier_mobile.ptl"
    optimized_traced_model._save_for_lite_interpreter(output_path)
    
    print(f"Output: {output_path}")

if __name__ == "__main__":
    export_to_mobile()