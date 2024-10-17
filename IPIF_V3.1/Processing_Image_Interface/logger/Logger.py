import logging
import os
import datetime

class Logger():
    def __init__(self, folder_name, file_name, name_logger, file_mode, format='%(asctime)s - %(name)s - %(levelname)s: %(message)s', mode="debug"):
        root_folder = "Processing_Image_Interface"
        if not (os.path.exists(path=root_folder)):
            os.mkdir(path=root_folder)
        if not (os.path.exists(path=root_folder + "/" + folder_name)):
            os.mkdir(path=root_folder + "/" + folder_name)
        current_date = datetime.datetime.now()
        str_date = current_date.strftime("%Y%m%d")
        if not (os.path.exists(path=root_folder + "/" + folder_name + "/" + str_date)):
            os.mkdir(path=root_folder + "/" + folder_name + "/" + str_date)
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(filename=root_folder + "/" + folder_name + "/" + str_date + "/" + file_name + ".log", filemode=file_mode, format=format)
        self.logger = logging.getLogger(name_logger)
        if mode == "debug":
            self.logger.setLevel(logging.DEBUG)
        elif mode == "info":
            self.logger.setLevel(logging.INFO)
        elif mode == "warning":
            self.logger.setLevel(logging.WARNING)
        elif mode == "error":
            self.logger.setLevel(logging.ERROR)
        elif mode == "critical":
            self.logger.setLevel(logging.CRITICAL)

        self.mode = mode

    def debug(self, msg):
        self.logger.debug(msg=msg)

    def info(self, msg):
        self.logger.info(msg=msg)

    def warning(self, msg):
        self.logger.warning(msg=msg)

    def error(self, msg):
        self.logger.error(msg=msg)

    def critical(self, msg):
        self.logger.critical(msg=msg)

    def trace(self, msg):
        if self.mode == "debug":
            self.logger.debug(msg=">>>> " + msg)
        elif self.mode == "info":
            self.logger.info(msg=">>>> " + msg)
        elif self.mode == "warning":
            self.logger.warning(msg=">>>> " + msg)
        elif self.mode == "error":
            self.logger.error(msg=">>>> " + msg)
        elif self.mode == "critical":
            self.logger.critical(msg=">>>> " + msg)