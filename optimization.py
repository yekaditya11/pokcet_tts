import os
import torch
import logging

logger = logging.getLogger(__name__)

def optimize_model(tts_model, level: int = 1):
    """
    Apply optimizations to the TTS model based on the specified level.
    
    Level 0: No optimization.
    Level 1: Threading optimization (Safe).
    Level 2: Dynamic Quantization (Fastest, potential slight quality trade-off).
    """
    logger.info(f"Applying CPU optimization level: {level}")
    
    if level == 0:
        logger.info("Level 0: Skipping all optimizations.")
        return

    # Level 1: Threading Optimization
    # -------------------------------
    # PyTorch defaults to using all available cores, which can cause contention
    # and context switching overhead for small/medium models like this.
    # We set it to a reasonable default (4) or respect OMP_NUM_THREADS.
    if level >= 1:
        num_threads = int(os.getenv("OMP_NUM_THREADS", "4"))
        torch.set_num_threads(num_threads)
        logger.info(f"Level 1: Set torch intra-op threads to {num_threads}")

    # Level 2: Dynamic Quantization
    # -----------------------------
    # Helper to quantize specific submodules
    if level >= 2:
        logger.info("Level 2: Applying dynamic quantization (int8)...")
        
        try:
            # Quantize FlowLM linear layers
            # We specifically target Linear layers which dominate compute.
            logger.info("Quantizing FlowLM...")
            tts_model.flow_lm = torch.quantization.quantize_dynamic(
                tts_model.flow_lm, {torch.nn.Linear}, dtype=torch.qint8
            )
            
            # Quantize Mimi (neural codec) linear layers
            # Mimi is smaller but runs at high frequency (24kHz / hop_length).
            logger.info("Quantizing Mimi...")
            tts_model.mimi = torch.quantization.quantize_dynamic(
                tts_model.mimi, {torch.nn.Linear}, dtype=torch.qint8
            )
            
            logger.info("Dynamic quantization applied successfully.")
        except Exception as e:
            logger.error(f"Failed to apply quantization: {e}")
            logger.warning("Reverting to unquantized model is not possible without reloading.")
