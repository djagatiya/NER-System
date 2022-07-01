import logging


def set_logger(file_path):

    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(file_path)
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.INFO)

    _format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(_format)
    f_handler.setFormatter(_format)
    
    def add_common_handers(logger_obj):
        logger_obj.setLevel(logging.INFO)
        logger_obj.addHandler(c_handler)
        logger_obj.addHandler(f_handler)

    add_common_handers(logging.getLogger())
    add_common_handers(logging.getLogger('transformers'))