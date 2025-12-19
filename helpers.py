import subprocess
import time
import logging

logger = logging.getLogger(__name__)

def run_command(cmd, timeout=30):
    """Run a command with timeout."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        logger.error(f"Command timed out: {cmd}")
        return False, "", "Command timed out"
    except Exception as e:
        logger.error(f"Error running command: {e}")
        return False, "", str(e)

def retry_operation(operation, max_retries=3, delay=2, *args, **kwargs):
    """Retry an operation with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return operation(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(delay * (2 ** attempt))
            else:
                raise e
