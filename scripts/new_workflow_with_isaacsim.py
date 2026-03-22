import logging
from pathlib import Path
from common_utils import network_config
from common_utils.socket_communication import (
    NonBlockingJSONSender,
    NonBlockingJSONReceiver,
)
from common_utils.custom_logger import CustomFormatter
from common_utils.workflow_control import BaseWorkflowController, parse_args

# root logger setup
handler = logging.StreamHandler()
handler.setFormatter(CustomFormatter())
logging.basicConfig(level=logging.DEBUG, handlers=[handler], force=True)
logger = logging.getLogger(__name__)

# Project root dir
PROJECT_ROOT_DIR = Path(__file__).resolve().parents[1]


class CLIWorkflowController(BaseWorkflowController):
    def __init__(self, args) -> None:
        sender = NonBlockingJSONSender(port=network_config.GRASPGEN_TO_ISAACSIM_PORT)
        receiver = NonBlockingJSONReceiver(
            port=network_config.ISAACSIM_TO_GRASPGEN_PORT
        )
        super().__init__(args, sender, receiver)

    def _send_EOF(self):
        # end of move
        self.sender.send_data(["EOF"])
        response = self.receiver.capture_data()
        while response is None:
            response = self.receiver.capture_data()
        if response["message"] == "EOF and ROS2 Complete":
            logger.warning("Success")
        elif response["message"] == "Abort":
            logger.warning("Abort")
            raise InterruptedError("aborted by isaacsim, stop current action")
        else:
            raise ValueError(f"Unknown message {response['message']}")

    def _handle_keyboard_interrupt(self):
        logger.info("Manual stopping current action.")
        self.sender.send_data(["Reset_to_default"])

    def _grab_command(self):
        print("Please provide the command, or type 'end' to end.")
        return input("Command: ")


def main():
    args = parse_args()
    with CLIWorkflowController(args) as controller:
        while True:
            controller.handle_task_command()


if __name__ == "__main__":
    main()
