"""
Utility functions for the stock prediction package.
"""
 
import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
 
 
def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> logging.Logger:
    """Setup logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file. If None, only logs to console.
        log_format: Custom log format string
 
    Returns:
        Configured logger
    """
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
 
    # Create logger
    logger = logging.getLogger('stock_prediction')
    logger.setLevel(getattr(logging, level.upper()))
 
    # Remove existing handlers
    logger.handlers = []
 
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
 
    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
 
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
 
        logger.info(f"Logging to file: {log_file}")
 
    return logger
 
 
def get_logger(name: str) -> logging.Logger:
    """Get a logger instance.
 
    Args:
        name: Logger name
 
    Returns:
        Logger instance
    """
    return logging.getLogger(f'stock_prediction.{name}')
 
 
def create_output_directory(base_dir: str = "output") -> Path:
    """Create output directory with timestamp.
 
    Args:
        base_dir: Base directory name
 
    Returns:
        Path to created directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(base_dir) / timestamp
    output_path.mkdir(parents=True, exist_ok=True)
 
    logger = get_logger(__name__)
    logger.info(f"Created output directory: {output_path}")
 
    return output_path
 
 
def save_model_results(
    output_dir: Path,
    model_name: str,
    results: dict
) -> None:
    """Save model evaluation results to file.
 
    Args:
        output_dir: Output directory path
        model_name: Name of the model
        results: Dictionary containing evaluation results
    """
    import json
 
    results_file = output_dir / f"{model_name}_results.json"
 
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    for key, value in results.items():
        if hasattr(value, 'tolist'):
            serializable_results[key] = value.tolist()
        elif isinstance(value, dict):
            serializable_results[key] = {
                k: v.tolist() if hasattr(v, 'tolist') else v
                for k, v in value.items()
            }
        else:
            serializable_results[key] = value
 
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
 
    logger = get_logger(__name__)
    logger.info(f"Saved results to {results_file}")