import inspect
import logging


def log(message, level):
    """Log a message with the caller's full module path and function name.

    Args:
        message: The message to log
        level: Log level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
    """
    # Get the calling function's frame
    frame = inspect.currentframe().f_back

    # Get the function name
    func_name = frame.f_code.co_name

    # Get the full module path from the frame's globals
    module_name = frame.f_globals.get('__name__', 'UNKNOWN_MODULE')

    # Format the log message
    formatted_message = f"[{module_name}.{func_name}()] {message}"

    # Get the appropriate logging function
    log_func = getattr(logging, level.lower())
    log_func(formatted_message)


def dummy():
    log("testing", 'WARNING')

if __name__ == "__main__":
    dummy()