from datetime import datetime
from webbrowser import get

def get_timestamp():
    """Get current time with format like 2022/09/22-20:10:53

    Returns:
        timestamp(str): current time
    """
    return datetime.now().strftime('20%y/%m/%d-%H:%M:%S')