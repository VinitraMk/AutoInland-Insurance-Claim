from datetime import datetime

def get_filename(filename):
    now = datetime.now().strftime('%d%m-%H%M%Y')
    return f'{filename}_{now}'
