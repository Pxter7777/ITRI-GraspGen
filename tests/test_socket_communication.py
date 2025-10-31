from common_utils.socket_communication import DataSender, DataReceiver
import pytest
import time
from multiprocessing import Process, Queue

SAMPLE_DATA = {
    "track": ["green cup", "orange cup", "pan"],
    "actions":
    [
        {
            "target_name": "orange cup",
            "qualifier": "cup_qualifier",
            "action": "grab_and_pour_and_place_back",
            "args": [
                "pan"
            ]
        },
        {
            "target_name": "green cup",
            "qualifier": "cup_qualifier",
            "action": "grab_and_pour_and_place_back",
            "args": [
                "pan"
            ]
        },
        {
            "target_name": "glass cup",
            "qualifier": "cup_qualifier",
            "action": "move_to",
            "args": [
                [326.8, -140.2, 212.6]
            ]
        }
    ]
}

SAMPLE_DATA_2 = {
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

def test_normal_send_and_receive():
    assert SAMPLE_DATA != {}

    receiver = DataReceiver(port=9876)
    sender = DataSender(port=9876)

    sender.send_data(SAMPLE_DATA)
    received_data = receiver.capture_data()
    
    assert SAMPLE_DATA == received_data

def test_send_and_receive_5_times():
    receiver = DataReceiver(port=9876)
    sender = DataSender(port=9876)
    for _ in range(5):
        sender.send_data(SAMPLE_DATA)
        received_data = receiver.capture_data()
        assert SAMPLE_DATA == received_data

def test_receiver_durability():
    receiver = DataReceiver(port=9876)
    sender = DataSender(port=9876)

    sender.send_data(SAMPLE_DATA)
    received_data = receiver.capture_data()
    assert SAMPLE_DATA == received_data

    # try to capture the next data
    sender.disconnect()
    sender2 = DataSender(port=9876)
    sender2.send_data(SAMPLE_DATA)
    received_data = receiver.capture_data()
    assert SAMPLE_DATA == received_data

# This function will be run in the background process
def receiver_loop(queue, port):
    """
    This function is intended to run in a separate process.
    It initializes a DataReceiver and continuously listens for data,
    putting any received data into a multiprocessing queue.
    It terminates when it receives a specific 'terminate' command.
    """
    receiver = DataReceiver(port=port)
    while True:
        data = receiver.capture_data()
        if data:
            queue.put(data)
            if data.get("action") == "terminate":
                break
    receiver.disconnect()

@pytest.fixture
def receiver_process():
    """
    A pytest fixture that sets up and tears down a DataReceiver running
    in a separate background process.
    
    Yields:
        tuple: A tuple containing the multiprocessing Queue for results
               and the port number used by the receiver.
    """
    port = 9877  # Use a different port to avoid conflicts
    queue = Queue()
    
    # Create and start the receiver process
    process = Process(target=receiver_loop, args=(queue, port))
    process.start()
    
    # Give the process a moment to initialize the socket
    time.sleep(0.5)
    
    yield queue, port
    
    # Teardown: send a termination message and clean up the process
    try:
        # Connect a sender to unblock the receiver's accept() call if it's waiting
        # and send the termination command.
        sender = DataSender(port=port)
        sender.send_data({"action": "terminate"})
        sender.disconnect()
    except ConnectionRefusedError:
        # This can happen if the receiver process has already crashed or exited.
        # It's safe to ignore in the teardown phase.
        pass
    finally:
        process.join(timeout=5)  # Wait for the process to finish
        if process.is_alive():
            process.terminate() # Force kill if it doesn't stop gracefully
            process.join()


def test_receiver_in_background_process(receiver_process):
    """
    Tests the DataSender against a DataReceiver running in a background process.
    """
    queue, port = receiver_process
    
    sender = DataSender(port=port)
    
    # Test sending SAMPLE_DATA
    sender.send_data(SAMPLE_DATA)
    
    # Get the result from the queue (with a timeout to prevent test hangs)
    received_data = queue.get(timeout=3)
    assert received_data == SAMPLE_DATA
    
    # Test sending SAMPLE_DATA_2
    sender.send_data(SAMPLE_DATA_2)
    received_data_2 = queue.get(timeout=3)
    assert received_data_2 == SAMPLE_DATA_2
    
    sender.disconnect()
