from common_utils.socket_communication import (
    NonBlockingJSONSender,
    BlockingJSONReceiver,
    NonBlockingJSONReceiver,
)
import time
from multiprocessing import Process, Queue
import logging
import pytest
from common_utils.custom_logger import CustomFormatter

SAMPLE_DATAS = [{"name": "bobby"}, [1, 2, 3, 4, 5], {"motions": [1, 2, 3, 4, 5]}]
SAMPLE_BIG_DATA = {"big_data": "x" * 1000000}

TIMEOUT = 5

handler = logging.StreamHandler()
handler.setFormatter(CustomFormatter())
logging.basicConfig(level=logging.DEBUG, handlers=[handler], force=True)
logger = logging.getLogger(__name__)


def blocking_receiver_loop(port: int, receiver_type: str, data_queue, error_queue):
    receiver = None
    if receiver_type == "non-blocking_receiver":
        receiver = NonBlockingJSONReceiver(port=port)
    elif receiver_type == "blocking_receiver":
        receiver = BlockingJSONReceiver(port=port)
    start_time = time.time()
    while time.time() - start_time < TIMEOUT:
        data = receiver.capture_data()
        # if the receiver is non-blocking, captured nothing and continue without blocking, should raise a error here
        if data is None:
            error_queue.put("Capture a None data.")
            receiver.disconnect()
            break
        if data:
            data_queue.put(data)
    receiver.disconnect()


def responsive_receiver_loop(port: int, receiver_type: str, data_queue, error_queue):
    """
    A loop that simulates a responsive application.
    It increments a counter on each iteration while also checking for socket data.
    A blocking `capture_data` will prevent the counter from incrementing.

    receiver can be either blocking or non-blocking, but only non-blocking type receiver can pass the assert
    """
    receiver = None
    if receiver_type == "non-blocking_receiver":
        receiver = NonBlockingJSONReceiver(port=port)
    elif receiver_type == "blocking_receiver":
        receiver = BlockingJSONReceiver(port=port)
    target_frame_duration = 1.0 / 60.0  # for ~60 FPS
    start_time = time.time()
    while time.time() - start_time < TIMEOUT:
        frame_start_time = time.time()
        # logger.debug(f"catching")
        data = receiver.capture_data()
        # logger.debug(f"catched {data}")
        if data:
            data_queue.put(data)

        # Ensure the loop runs at roughly 60 FPS
        frame_duration = time.time() - frame_start_time
        if frame_duration > target_frame_duration:
            logger.error(
                f"Error: the receiver is slowing down the process. {frame_duration}"
            )
            error_queue.put("Error: the receiver is slowing down the process.")
            break
        time.sleep(target_frame_duration - frame_duration)
    receiver.disconnect()


def receiver_process(port: int, task_type: str, receiver_type: str):
    """
    A pytest fixture that sets up and tears down a DataReceiver running
    in a separate background process.

    Yields:
        tuple: A tuple containing the multiprocessing Queue for results
               and the port number used by the receiver.
    """
    data_queue = Queue()
    error_queue = Queue()
    # Create and start the receiver process
    if task_type == "non-blocking_task":
        process = Process(
            target=responsive_receiver_loop,
            args=(port, receiver_type, data_queue, error_queue),
        )
    elif task_type == "blocking_task":
        process = Process(
            target=blocking_receiver_loop,
            args=(port, receiver_type, data_queue, error_queue),
        )
    process.start()

    # Give the process a moment to initialize the socket
    time.sleep(0.5)

    return process, data_queue, error_queue

    # Teardown: send a termination message and clean up the process
    try:
        # Connect a sender to unblock the receiver's accept() call if it's waiting
        # and send the termination command.
        sender = NonBlockingJSONSender(port=port)
        sender.send_data({"action": "terminate"})
        sender.disconnect()
    except ConnectionRefusedError:
        # This can happen if the receiver process has already crashed or exited.
        # It's safe to ignore in the teardown phase.
        pass
    finally:
        process.join(timeout=5)  # Wait for the process to finish
        if process.is_alive():
            process.terminate()  # Force kill if it doesn't stop gracefully
            process.join()


def test_Non_to_Non_sendall_first():  # should success
    process, data_queue, error_queue = receiver_process(
        port=9876, task_type="non-blocking_task", receiver_type="non-blocking_receiver"
    )
    sender = NonBlockingJSONSender(port=9876)

    for data in SAMPLE_DATAS:
        succ = sender.send_data(data)
        assert succ
    for data in SAMPLE_DATAS:
        assert data_queue.get(timeout=3) == data

    if process.is_alive():
        process.terminate()  # Force kill if it doesn't stop gracefully
    process.join()


def test_NonBlockingJSONReceiver_on_NonBlockingJSONReceiver_task():  # should success
    process, data_queue, error_queue = receiver_process(
        port=9876, task_type="non-blocking_task", receiver_type="non-blocking_receiver"
    )
    sender = NonBlockingJSONSender(port=9876)

    for data in SAMPLE_DATAS:
        time.sleep(0.2)
        succ = sender.send_data(data)
        assert succ
        assert data_queue.get(timeout=3) == data

    if process.is_alive():
        process.terminate()  # Force kill if it doesn't stop gracefully
    process.join()


def test_NonBlockingJSONReceiver_on_BlockingJSONReceiver_task():  # should fail
    process, data_queue, error_queue = receiver_process(
        port=9876, task_type="blocking_task", receiver_type="non-blocking_receiver"
    )
    sender = NonBlockingJSONSender(port=9876)
    succ = sender.send_data(SAMPLE_DATAS[0])
    assert not succ  # Because the receiver already left
    assert error_queue.get(timeout=3) == "Capture a None data."
    process.join(timeout=5)  # Wait for the process to finish
    if process.is_alive():
        process.terminate()  # Force kill if it doesn't stop gracefully
    process.join()


def test_BlockingJSONReceiver_on_BlockingJSONReceiver_task():  # should success
    process, data_queue, error_queue = receiver_process(
        port=9876, task_type="blocking_task", receiver_type="blocking_receiver"
    )
    sender = NonBlockingJSONSender(port=9876)

    for data in SAMPLE_DATAS:
        time.sleep(0.2)
        succ = sender.send_data(data)
        assert succ
        assert data_queue.get(timeout=3) == data

    # process.join(timeout=5)  # Wait for the process to finish
    if process.is_alive():
        process.terminate()  # Force kill if it doesn't stop gracefully
    process.join()


def test_BlockingJSONReceiver_on_NonBlockingJSONReceiver_task():  # should fail
    process, data_queue, error_queue = receiver_process(
        port=9876, task_type="non-blocking_task", receiver_type="blocking_receiver"
    )
    sender = NonBlockingJSONSender(port=9876)

    succ = sender.send_data(SAMPLE_DATAS[0])
    assert succ
    assert (
        error_queue.get(timeout=3) == "Error: the receiver is slowing down the process."
    )

    process.join(timeout=5)  # Wait for the process to finish
    if process.is_alive():
        process.terminate()  # Force kill if it doesn't stop gracefully
    process.join()


def test_send_to_disconnected_NonBlockingJSONReceiver():
    process, data_queue, error_queue = receiver_process(
        port=9876, task_type="non-blocking_task", receiver_type="non-blocking_receiver"
    )
    sender = NonBlockingJSONSender(port=9876)

    for data in SAMPLE_DATAS:
        time.sleep(0.2)
        succ = sender.send_data(data)
        assert succ
        assert data_queue.get(timeout=3) == data
    # kill the receiver process and starts a new one.
    if process.is_alive():
        process.terminate()
    process.join()
    process, data_queue, error_queue = receiver_process(
        port=9876, task_type="non-blocking_task", receiver_type="non-blocking_receiver"
    )
    for data in SAMPLE_DATAS:
        time.sleep(0.2)
        succ = sender.send_data(data)
        assert succ
        assert data_queue.get(timeout=3) == data
    if process.is_alive():
        process.terminate()  # Force kill if it doesn't stop gracefully
    process.join()


def test_send_to_non_open_socket():
    port = 9881
    sender = NonBlockingJSONSender(port=port)
    succ = sender.send_data(SAMPLE_DATAS[0])
    assert not succ


def test_send_big_data_to_NonBlockingJSONReceiver():
    sender = NonBlockingJSONSender(port=9878)
    receiver = NonBlockingJSONReceiver(port=9878)
    succ = sender.send_data(SAMPLE_BIG_DATA)
    assert succ
    received_data = receiver.capture_data()
    assert received_data == SAMPLE_BIG_DATA


def test_send_big_data_to_BlockingJSONReceiver():
    sender = NonBlockingJSONSender(port=9876)
    receiver = BlockingJSONReceiver(port=9876)
    succ = sender.send_data(SAMPLE_BIG_DATA)
    assert succ
    received_data = receiver.capture_data()
    assert received_data == SAMPLE_BIG_DATA

def test_raise_occupying_socket():
    """Test that raises error if port is occupied."""
    receiver1 = BlockingJSONReceiver(port=9890)
    with pytest.raises(
        ConnectionAbortedError,
        match=r"An error occurred during connecting localhost:9890: \[Errno 98\] Address already in use",
    ):
        _ = BlockingJSONReceiver(port=9890)
    receiver1.disconnect()
    """Test that raises error if port is occupied."""
    receiver2 = NonBlockingJSONReceiver(port=9891)
    with pytest.raises(
        ConnectionAbortedError,
        match=r"An error occurred during connecting localhost:9891: \[Errno 98\] Address already in use",
    ):
        _ = NonBlockingJSONReceiver(port=9891)
    receiver2.disconnect()
    

if __name__ == "__main__":
    test_raise_occupying_socket()
