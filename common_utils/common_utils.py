import json
import datetime
import logging
import os

logger = logging.getLogger(__name__)


def save_json(dir: str, prefix: str, data) -> True:  # save json data for test
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_dir = os.path.dirname(current_file_dir)
    current_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(
        project_root_dir, "data_for_test", dir, prefix + current_time_str + ".json"
    )
    logger.info(f"save to {filepath}")
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)
