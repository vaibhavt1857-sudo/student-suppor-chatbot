import logging

LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
logging.basicConfig(filename="app.log",
                    level=logging.DEBUG,
                    format=LOG_FORMAT)
