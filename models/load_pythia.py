"""
Load Pythia-410M checkpoints from HuggingFace
EleutherAI provides 154 checkpoints from training
"""

import torch
from pathlib import Path
from typing import Optional, Tuple
from transformer_lens import HookedTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PythiaCheckpointLoader:
    # Mapping from step number to HuggingFace revision name
    STEP_TO_REVISION = {
        0: "step0",
        1: "step1", 
        2: "step2",
        4: "step4",
        8: "step8",
        16: "step16",
        32: "step32",
        64: "step64",
        128: "step128",
        256: "step256",
        512: "step512",
        1000: "step1000",
        2000: "step2000",
        3000: "step3000",
        4000: "step4000",
        5000: "step5000",
        6000: "step6000",
        7000: "step7000",
        8000: "step8000",
        9000: "step9000",
        10000: "step10000",
        11000: "step11000",
        12000: "step12000",
        13000: "step13000",
        14000: "step14000",
        15000: "step15000",
        16000: "step16000",
        17000: "step17000",
        18000: "step18000",
        19000: "step19000",
        20000: "step20000",
        21000: "step21000",
        22000: "step22000",
        23000: "step23000",
        24000: "step24000",
        25000: "step25000",
        26000: "step26000",
        27000: "step27000",
        28000: "step28000",
        29000: "step29000",
        30000: "step30000",
        31000: "step31000",
        32000: "step32000",
        33000: "step33000",
        34000: "step34000",
        35000: "step35000",
        36000: "step36000",
        37000: "step37000",
        38000: "step38000",
        39000: "step39000",
        40000: "step40000",
        41000: "step41000",
        42000: "step42000",
        43000: "step43000",
        44000: "step44000",
        45000: "step45000",
        46000: "step46000",
        47000: "step47000",
        48000: "step48000",
        49000: "step49000",
        50000: "step50000",
        51000: "step51000",
        52000: "step52000",
        53000: "step53000",
        54000: "step54000",
        55000: "step55000",
        56000: "step56000",
        57000: "step57000",
        58000: "step58000",
        59000: "step59000",
        60000: "step60000",
        61000: "step61000",
        62000: "step62000",
        63000: "step63000",
        64000: "step64000",
        65000: "step65000",
        66000: "step66000",
        67000: "step67000",
        68000: "step68000",
        69000: "step69000",
        70000: "step70000",
        71000: "step71000",
        72000: "step72000",
        73000: "step73000",
        74000: "step74000",
        75000: "step75000",
        76000: "step76000",
        77000: "step77000",
        78000: "step78000",
        79000: "step79000",
        80000: "step80000",
        81000: "step81000",
        82000: "step82000",
        83000: "step83000",
        84000: "step84000",
        85000: "step85000",
        86000: "step86000",
        87000: "step87000",
        88000: "step88000",
        89000: "step89000",
        90000: "step90000",
        91000: "step91000",
        92000: "step92000",
        93000: "step93000",
        94000: "step94000",
        95000: "step95000",
        96000: "step96000",
        97000: "step97000",
        98000: "step98000",
        99000: "step99000",
        100000: "step100000",
        101000: "step101000",
        102000: "step102000",
        103000: "step103000",
        104000: "step104000",
        105000: "step105000",
        106000: "step106000",
        107000: "step107000",
        108000: "step108000",
        109000: "step109000",
        110000: "step110000",
        111000: "step111000",
        112000: "step112000",
        113000: "step113000",
        114000: "step114000",
        115000: "step115000",
        116000: "step116000",
        117000: "step117000",
        118000: "step118000",
        119000: "step119000",
        120000: "step120000",
        121000: "step121000",
        122000: "step122000",
        123000: "step123000",
        124000: "step124000",
        125000: "step125000",
        126000: "step126000",
        127000: "step127000",
        128000: "step128000",
        129000: "step129000",
        130000: "step130000",
        131000: "step131000",
        132000: "step132000",
        133000: "step133000",
        134000: "step134000",
        135000: "step135000",
        136000: "step136000",
        137000: "step137000",
        138000: "step138000",
        139000: "step139000",
        140000: "step140000",
        141000: "step141000",
        142000: "step142000",
        143000: "step143000",
    }
    
    def __init__(
        self, 
        model_name: str = "EleutherAI/pythia-410m",
        cache_dir: Optional[str] = "./cache"
    ):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def load_checkpoint(
        self, 
        step: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[HookedTransformer, any]:
        # Validate step
        if step not in self.STEP_TO_REVISION:
            available = list(self.STEP_TO_REVISION.keys())
            raise ValueError(
                f"Invalid checkpoint step: {step}. "
                f"Available steps: {available[:10]}...{available[-5:]}"
            )
        
        revision = self.STEP_TO_REVISION[step]
        logger.info(f"Loading {self.model_name} at {revision} (step {step})")
        
        # Load with TransformerLens
        try:
            model = HookedTransformer.from_pretrained(
                self.model_name,
                revision=revision,
                device=device,
                cache_dir=str(self.cache_dir) if self.cache_dir else None,
                torch_dtype=dtype,
            )
            
            tokenizer = model.tokenizer
            
            logger.info(f"- Successfully loaded checkpoint at step {step}")
            logger.info(f"  Model: {model.cfg.n_layers} layers, {model.cfg.d_model} hidden dim")
            logger.info(f"  Vocab size: {model.cfg.d_vocab}")
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint {step}: {e}")
            raise
    
    @staticmethod
    def get_available_steps() -> list:
        return sorted(PythiaCheckpointLoader.STEP_TO_REVISION.keys())
    
    @staticmethod
    def get_closest_step(target_step: int) -> int:
        available = PythiaCheckpointLoader.get_available_steps()
        closest = min(available, key=lambda x: abs(x - target_step))
        return closest


def get_available_checkpoints() -> list:
    return PythiaCheckpointLoader.get_available_steps()


# Example usage and testing
if __name__ == "__main__":
    print("Available checkpoint steps:")
    steps = get_available_checkpoints()
    print(f"  Total: {len(steps)} checkpoints")
    print(f"  Range: {steps[0]} to {steps[-1]}")
    print(f"  First 10: {steps[:10]}")
    print(f"  Last 10: {steps[-10:]}")
    
    # Test loading
    loader = PythiaCheckpointLoader()
    print("\nTesting checkpoint loading...")
    try:
        model, tokenizer = loader.load_checkpoint(step=0, device="cpu")
        print(f"- Successfully loaded step 0 (random init)")
        print(f"  Model type: {type(model)}")
        print(f"  Device: {next(model.parameters()).device}")
    except Exception as e:
        print(f"- Failed to load: {e}")
