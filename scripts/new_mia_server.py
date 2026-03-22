import logging
from pathlib import Path
from common_utils import network_config
from common_utils.custom_logger import CustomFormatter
from common_utils.workflow_control import BaseWorkflowController, parse_args
from common_utils.socket_communication import (
    NonBlockingJSONSender,
    NonBlockingJSONReceiver,
)

# root logger setup
handler = logging.StreamHandler()
handler.setFormatter(CustomFormatter())
logging.basicConfig(level=logging.DEBUG, handlers=[handler], force=True)
logger = logging.getLogger(__name__)

# Project root dir
PROJECT_ROOT_DIR = Path(__file__).resolve().parents[1]


class MiaWorkflowController(BaseWorkflowController):
    def __init__(self, args) -> None:
        sender = NonBlockingJSONSender(port=network_config.GRASPGEN_TO_ISAACSIM_PORT)
        receiver = NonBlockingJSONReceiver(
            port=network_config.ISAACSIM_TO_GRASPGEN_PORT
        )
        super().__init__(args, sender, receiver)
        self.main_sender = NonBlockingJSONSender(
            port=network_config.GRASPGEN_TO_MIA_PORT
        )
        self.main_receiver = NonBlockingJSONReceiver(
            port=network_config.MIA_TO_GRASPGEN_PORT
        )

    def _send_EOF(self):
        # end of move
        self.sender.send_data(["EOF"])
        response = self.receiver.capture_data()
        while response is None:
            response = self.receiver.capture_data()
        if response["message"] == "EOF and ROS2 Complete":
            logger.warning("Success")
            self.main_sender.send_data({"message": "Success"})
        elif response["message"] == "Abort":
            logger.warning("Abort")
            self.main_sender.send_data({"message": "Fail"})
            raise InterruptedError("aborted by isaacsim, stop current action")
        else:
            raise ValueError(f"Unknown message {response['message']}")

    def _handle_keyboard_interrupt(self):
        logger.info("Manual stopping current action.")
        self.sender.send_data(["Reset_to_default"])
        self.main_sender.send_data({"message": "Fail"})

    def _grab_command(self) -> str:
        while True:
            task_signal = self.main_receiver.capture_data()
            if task_signal is None:
                continue
            text = task_signal.get("actions")
            if text is None:
                raise ValueError(f"received weird signal {task_signal}")
            return text


def main():
    args = parse_args()
    with MiaWorkflowController(args) as controller:
        while True:
            controller.handle_task_command()


if __name__ == "__main__":
    main()
