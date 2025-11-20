import socket
import json
import logging
import struct
import select
# Some example joint configurations to send

logger = logging.getLogger(__name__)


SAMPLE_DATA = {
    "track": ["green cup", "blue cup", "pan"],
    "actions": [
        {
            "target_name": "green cup",
            "qualifier": "cup_qualifier",
            "action": "grab_and_pour_and_place_back",
            "args": ["pan"],
        },
        {
            "target_name": "blue cup",
            "qualifier": "cup_qualifier",
            "action": "grab_and_pour_and_place_back",
            "args": ["pan"],
        },
        {
            "target_name": "blue cup",
            "qualifier": "cup_qualifier",
            "action": "move_to",
            "args": [[326.8, -140.2, 212.6]],
        },
    ],
}


class NonBlockingJSONSender:
    """
    A class to manage connection and sending goals to the robot bridge.
    Connects automatically upon instantiation.
    """

    def __init__(self, port, host="localhost"):
        self.host = host
        self.port = port
        self.socket = None
        self._connect_on_init()  # Attempt connection during initialization

    def _connect_on_init(self):
        """
        Internal method to establish connection. Used during init and for re-connection.
        Returns True on success, False otherwise.
        """
        if self.socket:
            # Already connected or socket object exists, close it first to ensure a fresh connection
            self.disconnect()

        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            logger.info(f"Sender connected to receiver at {self.host}:{self.port}")
            return True
        except ConnectionRefusedError:
            logger.warning(f"No socket currently listening at {self.host}:{self.port}")
            self.socket = None
            return False
        except Exception as e:
            logger.exception(f"An error occurred during connection: {e}")
            self.socket = None
            return False

    def disconnect(self):
        """
        Closes the connection to the robot bridge.
        """
        if self.socket:
            self.socket.close()
            self.socket = None
            logger.warning("Sender disconnected")

    def reconnect(self) -> bool:
        """
        Closes the current connection and establishes a new one.
        """
        logger.info("Attempting to reconnect sender...")
        self.disconnect()
        return self._connect_on_init()

    def send_data(self, data: dict) -> bool:
        """
        Sends a single joint position goal to the robot bridge.
        Attempts to reconnect if the connection is lost.
        Returns True on successful send, False otherwise.
        """
        if not self.socket:
            logger.info("Connection not established. Attempting to reconnect.")
            if not self.reconnect():
                return False

        # Check if the socket is still alive before sending
        try:
            ready_to_read, _, _ = select.select([self.socket], [], [], 0)
            if ready_to_read:
                # If the socket is readable, it might be closed.
                # A recv with MSG_PEEK will not remove data from buffer.
                # If it returns b'', the peer has closed the connection.
                if self.socket.recv(1, socket.MSG_PEEK) == b"":
                    logger.warning("Receiver has closed the connection.")
                    raise BrokenPipeError("Connection closed by peer")
        except BrokenPipeError:
            logger.info("Connection lost. Attempting to reconnect and resend.")
            if self.reconnect():
                return self.send_data(data)  # Retry sending
            else:
                return False
        except Exception as e:
            logger.exception(f"An error occurred while checking socket status: {e}")
            return False

        if not (isinstance(data, dict) or isinstance(data, list)):
            logger.error("data is not a dict or a list")
            return False

        message_bytes = json.dumps(data).encode("utf-8")
        header = struct.pack(">I", len(message_bytes))

        try:
            logger.debug(f"Sending signal: {data}")
            self.socket.sendall(header + message_bytes)
            logger.info("Sent!")
            return True
        except BrokenPipeError:
            logger.warning("Connection lost while sending. Attempting to reconnect.")
            if self.reconnect():  # Try to reconnect
                return self.send_data(data)  # Retry sending
            return False
        except Exception as e:
            logger.exception(f"An error occurred while sending: {e}")
            return False


class NonBlockingJSONReceiver:
    """
    A class to manage connection and receive dict.
    Connects automatically upon instantiation.
    """

    def __init__(self, port, host="localhost"):
        self.host = host
        self.port = port
        self.socket = None
        self.conn = None
        self.buffer = b""
        self.msg_len = None
        self._connect_on_init()  # Attempt connection during initialization

    def _connect_on_init(self):
        """
        Internal method to establish connection. Used during init and for re-connection.
        Returns True on success, False otherwise.
        """
        if self.socket:
            # Already connected or socket object exists, close it first to ensure a fresh connection
            self.disconnect()

        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.setblocking(False)
            self.socket.bind((self.host, self.port))
            self.socket.listen()

            self.conn = None
            logger.info(f"Receiver starts listening at {self.host}:{self.port}")
            return True
        except ConnectionRefusedError:
            logger.exception("Receiver connection failed.")
            self.socket = None
            return False
        except Exception as e:
            logger.exception(f"An error occurred during connection: {e}")
            self.socket = None
            return False

    def disconnect(self):
        if self.socket:
            self.socket.close()
            self.socket = None
            logger.info("receiver disconnected")

    def capture_data(self):
        try:
            if self.conn is None:
                self.conn, addr = self.socket.accept()
                self.conn.setblocking(False)
                self.buffer = b""
                self.msg_len = None
                logger.info(f"Accepted connection from {addr}")

            data = self.conn.recv(4096)
            if not data:
                logger.warning("Sender disconnected.")
                self.conn.close()
                self.conn = None
                return None
            self.buffer += data
        except BlockingIOError:
            pass  # No data available
        except Exception as e:
            logger.exception(f"Socket server error on recv: {e}")
            if self.conn:
                self.conn.close()
            self.conn = None
            return None

        # Process buffer for a complete message
        if self.msg_len is None:
            if len(self.buffer) >= 4:
                self.msg_len = struct.unpack(">I", self.buffer[:4])[0]
                self.buffer = self.buffer[4:]
            else:
                return None  # Not enough data for header

        if len(self.buffer) >= self.msg_len:
            message_bytes = self.buffer[: self.msg_len]
            self.buffer = self.buffer[self.msg_len :]
            self.msg_len = None  # Reset for next message
            try:
                return json.loads(message_bytes.decode("utf-8"))
            except json.JSONDecodeError as e:
                logger.exception(f"JSON decode error: {e}")
                return None

        return None  # Not enough data for full message


class BlockingJSONReceiver:
    """
    A class to manage connection and receive dict.
    Connects automatically upon instantiation.
    """

    def __init__(self, port, host="localhost"):
        self.host = host
        self.port = port
        self.socket = None
        self.conn = None
        self._connect_on_init()  # Attempt connection during initialization

    def _connect_on_init(self):
        """
        Internal method to establish connection. Used during init and for re-connection.
        Returns True on success, False otherwise.
        """
        if self.socket:
            # Already connected or socket object exists, close it first to ensure a fresh connection
            self.disconnect()

        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.host, self.port))
            self.socket.listen()

            self.conn = None
            logger.info(f"Receiver starts listening at {self.host}:{self.port}")
            return True
        except ConnectionRefusedError:
            logger.exception("Receiver connection failed.")
            self.socket = None
            return False
        except Exception as e:
            logger.exception(f"An error occurred during connection: {e}")
            self.socket = None
            return False

    def disconnect(self):
        if self.socket:
            self.socket.close()
            self.socket = None
            logger.info("receiver disconnected")

    def _read_blocking(self, n):
        """Helper to read exactly n bytes from a blocking socket."""
        data = b""
        while len(data) < n:
            packet = self.conn.recv(n - len(data))
            if not packet:
                return None
            data += packet
        return data

    def capture_data(self):
        try:
            if self.conn is None:
                self.conn, addr = self.socket.accept()
                logger.info(f"Accepted connection from {addr}")

            header_data = self._read_blocking(4)
            if not header_data:
                logger.warning("Sender disconnected. Re-accepting... ")
                self.conn.close()
                self.conn = None
                return self.capture_data()  # Wait for new connection

            msg_len = struct.unpack(">I", header_data)[0]

            message_bytes = self._read_blocking(msg_len)
            if not message_bytes:
                logger.warning("Sender disconnected. Re-accepting... ")
                self.conn.close()
                self.conn = None
                return self.capture_data()  # Wait for new connection

            return json.loads(message_bytes.decode("utf-8"))
        except (json.JSONDecodeError, struct.error) as e:
            logger.exception(f"Data format error: {e}")
            if self.conn:
                self.conn.close()
            self.conn = None
            return None
        except Exception as e:
            logger.exception(f"Socket server error: {e}")
            if self.conn:
                self.conn.close()
            self.conn = None
            return None
