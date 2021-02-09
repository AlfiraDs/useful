import boto3
import io
import os
import sys
import logging


s3_stream = io.StringIO()
s3_err_stream = io.StringIO()


def write_logs_to_s3(body, bucket_key):
    logger = logging.getLogger(__name__)
    logger.debug(f'Writing logs to {bucket_key}')
    s3 = boto3.client("s3")
    bucket = bucket_key.split('/')[0]
    key = '/'.join(bucket_key.split('/')[1:])
    s3.put_object(Body=body, Bucket=bucket, Key=key)
    logger.debug(f'Logs are written')


class InfoFilter(logging.Filter):
    def filter(self, rec):
        return rec.levelno in (logging.DEBUG, logging.INFO)


def get_config(log_file_path=None, s3=None):
    logging_config = {
        "version": 1,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(levelname)s - %(name)-28s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            }
        },
        "filters": {
            "info_filter": {
                "()": InfoFilter
            }
        },
        "handlers": {
            "red_stream": {
                "class": "logging.StreamHandler",
                "level": logging.ERROR,
                "formatter": "default",
                "filters": [],
                "stream": sys.stderr
            },
            "out_stream": {
                "class": "logging.StreamHandler",
                "level": logging.DEBUG,
                "formatter": "default",
                "filters": ["info_filter"],
                "stream": sys.stdout
            }
        },
        "loggers": {
            "multiprocessing": {
                "level": logging.DEBUG,
                "filters": [],
                "handlers": ["red_stream", "out_stream"]
            }
        },
        "root": {
            "level": logging.DEBUG,
            "filters": [],
            "handlers": ["red_stream", "out_stream"]
        },
        "incremental": False,
        "disable_existing_loggers": True
    }

    if log_file_path:
        try:
            os.makedirs(os.path.dirname(log_file_path))
        except FileExistsError:
            pass

        logging_config["handlers"]["file_stream"] = {
            "class": "logging.FileHandler",
            "level": logging.DEBUG,
            "formatter": "default",
            "filters": [],
            "filename": log_file_path
        }
        logging_config["handlers"]["file_err_stream"] = {
            "class": "logging.FileHandler",
            "level": logging.ERROR,
            "formatter": "default",
            "filters": [],
            "filename": log_file_path + ".errors"
        }
        logging_config["root"]["handlers"] += ["file_stream", "file_err_stream"]
        # logging_config["loggers"]["multiprocessing"]["handlers"] += ["file_stream"]

    if s3:
        logging_config["handlers"]["s3_stream"] = {
            "class": "logging.StreamHandler",
            "level": logging.DEBUG,
            "formatter": "default",
            "filters": [],
            "stream": s3_stream
        }
        logging_config["handlers"]["s3_err_stream"] = {
            "class": "logging.StreamHandler",
            "level": logging.ERROR,
            "formatter": "default",
            "filters": [],
            "stream": s3_err_stream
        }
        logging_config["root"]["handlers"] += ["s3_stream", "s3_err_stream"]
        # logging_config["loggers"]["multiprocessing"]["handlers"] += ["s3_stream"]

    return logging_config
