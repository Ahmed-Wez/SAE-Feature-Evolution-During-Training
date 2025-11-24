"""
Prepare dataset for activation collection
Uses a subset of The Pile for efficiency
"""

import torch
from datasets import load_dataset
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PileDatasetLoader:
    """
    Load and prepare The Pile dataset for activation collection
    """
    
    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def load_pile_subset(
        self,
        n_samples: int = 100000,
        context_length: int = 2048,
        split: str = "train",
    ):
        """
        Load a subset of The Pile
        
        Args:
            n_samples: Number of samples to load
            context_length: Maximum sequence length
            split: Dataset split to use
            
        Returns:
            texts: List of text strings
        """
        
        logger.info(f"Loading {n_samples} samples from The Pile...")
        
        try:
            # Load The Pile
            dataset = load_dataset(
                "EleutherAI/pile",
                split=split,
                streaming=True,  # Stream to avoid loading everything
                cache_dir=str(self.cache_dir),
            )
            
            texts = []
            for i, example in enumerate(dataset):
                if i >= n_samples:
                    break
                
                text = example['text']
                
                # Basic cleaning
                text = text.strip()
                if len(text) > 10:  # Skip very short texts
                    texts.append(text)
                
                if (i + 1) % 10000 == 0:
                    logger.info(f"  Loaded {i+1}/{n_samples} samples")
            
            logger.info(f"✓ Loaded {len(texts)} text samples")
            return texts
            
        except Exception as e:
            logger.error(f"Failed to load Pile dataset: {e}")
            logger.info("Falling back to dummy data for testing...")
            return self._generate_dummy_data(n_samples)
    
    def _generate_dummy_data(self, n_samples: int):
        """Generate dummy data if Pile fails to load"""
        logger.warning("Using dummy data - only for testing!")
        
        templates = [
            "The quick brown fox jumps over the lazy dog. ",
            "Machine learning is a subset of artificial intelligence. ",
            "Python is a high-level programming language. ",
            "Neural networks consist of interconnected nodes. ",
            "Data science combines statistics and computer science. ",
        ]
        
        texts = []
        for i in range(n_samples):
            # Create varied texts
            text = templates[i % len(templates)] * (i % 10 + 1)
            texts.append(text)
        
        return texts


if __name__ == "__main__":
    # Test loading
    loader = PileDatasetLoader()
    texts = loader.load_pile_subset(n_samples=100)
    
    print(f"\nLoaded {len(texts)} samples")
    print(f"Example text (first 200 chars):")
    print(texts[0][:200])
