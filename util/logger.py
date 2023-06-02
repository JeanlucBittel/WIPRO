import logging

class CustomLogger():
    
    def __init__(self, logger_name):
        self.log = logging.getLogger(logger_name)
        self.log.propagate = False

        if not self.log.handlers:
            # create console handler and set level to debug
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            # create formatter
            formatter = logging.Formatter('%(levelname)s: %(message)s')
            # add formatter to ch
            ch.setFormatter(formatter)
            # add ch to logger
            self.log.addHandler(ch)
            
    def setLogLevel(self, log_level):
        self.log.setLevel(log_level)
        return self.log