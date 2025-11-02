import socket
import time
import json
import logging
# Some example joint configurations to send

logger = logging.getLogger(__name__)


SAMPLE_DATA = {
    "track": ["green cup", "blue cup", "pan"],
    "actions":
    [
        {
            "target_name": "green cup",
            "qualifier": "cup_qualifier",
            "action": "grab_and_pour_and_place_back",
            "args": [
                "pan"
            ]
        },
        {
            "target_name": "blue cup",
            "qualifier": "cup_qualifier",
            "action": "grab_and_pour_and_place_back",
            "args": [
                "pan"
            ]
        },
        {
            "target_name": "blue cup",
            "qualifier": "cup_qualifier",
            "action": "move_to",
            "args": [
                [326.8, -140.2, 212.6]
            ]
        }
    ]
    
    
}

class NonBlockingJSONSender:
    """
    A class to manage connection and sending goals to the robot bridge.
    Connects automatically upon instantiation.
    """
    def __init__(self, host='localhost', port=9870):
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
            logger.exception(f"Connection failed. Is the bridge.py script running on {self.host}:{self.port}?")
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

    def send_data(self, data: dict) -> bool:
        """
        Sends a single joint position goal to the robot bridge.
        Attempts to reconnect if the connection is lost.
        Returns True on successful send, False otherwise.
        """
        if not self.socket:
            logger.warning("Connection not established. Attempting to reconnect.")
            if not self._connect_on_init():  # Try to reconnect
                return False
        if not (isinstance(data, dict) or isinstance(data, list)):
            logger.error(f"data is not a dict or a list")
            return False
        
        signal_str = json.dumps(data)
        try:
            logger.debug(f"Sending signal: {signal_str}")
            # Encode the string to bytes and send it
            self.socket.sendall(signal_str.encode('utf-8'))
            logger.info("Sent!")
            return True
        except BrokenPipeError:
            logger.exception("Connection lost while sending. Attempting to reconnect.")
            self.disconnect()
            if self._connect_on_init():  # Try to reconnect
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
    def __init__(self, host='localhost', port=9870):
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
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.setblocking(False)
            self.socket.bind((self.host, self.port))
            self.socket.listen()

            self.conn = None
            logger.info(f"Receiver starts listening at {self.host}:{self.port}")
            return True
        except ConnectionRefusedError:
            logger.exception(f"Receiver connection failed.")
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
            # Decode the received bytes into a string, and split by comma
            if self.conn is None:
                self.conn, _ = self.socket.accept()
                self.conn.setblocking(False)
            data = self.conn.recv(1024)
            if not data:
                logger.warning("detected sender disconnected, attempt to re-accept connection")
                self.conn.close()
                self.conn = None
                return self.capture_data()
            data_str = data.decode('utf-8').strip()
            data_loaded = json.loads(data_str) 
            return data_loaded
        except BlockingIOError as e:
            #logger.exception(e)
            #logger.info("Nothing Captured, return None.")
            return None
        except Exception as e:
            logger.exception(f'Socket server error: {e}')
            time.sleep(1) # Avoid busy-looping on error

class BlockingJSONReceiver:
    """
    A class to manage connection and receive dict.
    Connects automatically upon instantiation.
    """
    def __init__(self, host='localhost', port=9870):
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
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.host, self.port))
            self.socket.listen()

            self.conn = None
            logger.info(f"Receiver starts listening at {self.host}:{self.port}")
            return True
        except ConnectionRefusedError:
            logger.exception(f"Receiver connection failed.")
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
            # Decode the received bytes into a string, and split by comma
            if self.conn is None:
                self.conn, _ = self.socket.accept()
            data = self.conn.recv(1024)
            if not data:
                logger.warning("detected sender disconnected, attempt to re-accept connection")
                self.conn.close()
                self.conn = None
                return self.capture_data()
            data_str = data.decode('utf-8').strip()
            data_loaded = json.loads(data_str) 
            return data_loaded
        except Exception as e:
            logger.exception(f'Socket server error: {e}')
            time.sleep(1) # Avoid busy-looping on error
