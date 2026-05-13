"""Test JSON socket sender and receiver classes."""

from __future__ import annotations

import logging
import time
from multiprocessing import Process, Queue

import pytest

from common_utils.log_formatter import CustomLoggingFormatter
from common_utils.socket_communication import (
    BlockingJSONReceiver,
    NonBlockingJSONReceiver,
    NonBlockingJSONSender,
)

SAMPLE_DATAS = [{"name": "bobby"}, [1, 2, 3, 4, 5], {"motions": [1, 2, 3, 4, 5]}]
SAMPLE_BIG_DATA = {"big_data": "x" * 1000000}

TIMEOUT = 5

handler = logging.StreamHandler()
handler.setFormatter(CustomLoggingFormatter())
logging.basicConfig(level=logging.DEBUG, handlers=[handler], force=True)
logger = logging.getLogger(__name__)


def blocking_receiver_loop(
    port: int, receiver_type: str, data_queue: Queue[object], error_queue: Queue[object]
) -> None:
    """Run a receiver loop that blocks on each capture_data call.

    Args:
        port (int): The TCP port to listen on.
        receiver_type (str): Either "non-blocking_receiver" or "blocking_receiver".
        data_queue (Queue[object]): Queue for captured data.
        error_queue (Queue[object]): Queue for error messages.

    Raises:
        TypeError: If receiver_type is unknown.
    """
    receiver = None
    if receiver_type == "non-blocking_receiver":
        receiver = NonBlockingJSONReceiver(port=port)
    elif receiver_type == "blocking_receiver":
        receiver = BlockingJSONReceiver(port=port)
    if receiver is None:
        raise TypeError(f"Unknown receiver_type: {receiver_type}")
    start_time = time.time()
    while time.time() - start_time < TIMEOUT:
        data = receiver.capture_data()
        # if the receiver is non-blocking, captured nothing
        # and continue without blocking, should raise an error
        if data is None:
            error_queue.put("Capture a None data.")
            receiver.disconnect()
            break
        if data:
            data_queue.put(data)
    receiver.disconnect()


def responsive_receiver_loop(
    port: int, receiver_type: str, data_queue: Queue[object], error_queue: Queue[object]
) -> None:
    """Simulate a responsive 60 FPS application loop while checking for socket data.

    A blocking `capture_data` will prevent the counter from incrementing.
    receiver can be either blocking or non-blocking, but only non-blocking type
    receiver can pass the assert.

    Args:
        port (int): The TCP port to listen on.
        receiver_type (str): Either "non-blocking_receiver" or "blocking_receiver".
        data_queue (Queue[object]): Queue for captured data.
        error_queue (Queue[object]): Queue for error messages.

    Raises:
        TypeError: If receiver_type is unknown.
    """
    receiver = None
    if receiver_type == "non-blocking_receiver":
        receiver = NonBlockingJSONReceiver(port=port)
    elif receiver_type == "blocking_receiver":
        receiver = BlockingJSONReceiver(port=port)
    if receiver is None:
        raise TypeError(f"Unknown receiver_type: {receiver_type}")
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


def receiver_process(
    port: int, task_type: str, receiver_type: str
) -> tuple[Process, Queue[object], Queue[object]]:
    """Set up a DataReceiver in a separate background process.

    Args:
        port (int): The TCP port to use.
        task_type (str): Either "non-blocking_task" or "blocking_task".
        receiver_type (str): Either "non-blocking_receiver" or "blocking_receiver".

    Returns:
        tuple[Process, Queue[object], Queue[object]]: The process, data queue,
            and error queue.

    Raises:
        ValueError: If task_type is unknown.
    """
    data_queue: Queue[object] = Queue()
    error_queue: Queue[object] = Queue()
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
    else:
        raise ValueError(f"Unknown task_type: {task_type}")
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


def test_non_to_non_sendall_first():  # should success
    """Send all data before the receiver reads any."""
    process, data_queue, _error_queue = receiver_process(
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


def test_nonblocking_receiver_on_nonblocking_task():  # should success
    """Send and receive data interleaved with a non-blocking receiver."""
    process, data_queue, _error_queue = receiver_process(
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


def test_nonblocking_receiver_on_blocking_task():  # should fail
    """Verify non-blocking receiver fails on a blocking task."""
    process, _data_queue, error_queue = receiver_process(
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


def test_blocking_receiver_on_blocking_task():  # should success
    """Send and receive data interleaved with a blocking receiver."""
    process, data_queue, _error_queue = receiver_process(
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


def test_blocking_receiver_on_nonblocking_task():  # should fail
    """Verify blocking receiver slows down a non-blocking task loop."""
    process, _data_queue, error_queue = receiver_process(
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


def test_send_to_disconnected_nonblocking_receiver():
    """Reconnect and resume sending after the receiver process restarts."""
    process, data_queue, _error_queue = receiver_process(
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
    process, data_queue, _error_queue = receiver_process(
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
    """Return failure when no receiver is listening on the port."""
    port = 9881
    sender = NonBlockingJSONSender(port=port)
    succ = sender.send_data(SAMPLE_DATAS[0])
    assert not succ


def test_send_big_data_to_nonblocking_receiver():
    """Send a large payload through a non-blocking receiver."""
    sender = NonBlockingJSONSender(port=9878)
    receiver = NonBlockingJSONReceiver(port=9878)
    succ = sender.send_data(SAMPLE_BIG_DATA)
    assert succ
    received_data = receiver.capture_data()
    assert received_data == SAMPLE_BIG_DATA


def test_send_big_data_to_blocking_receiver():
    """Send a large payload through a blocking receiver."""
    sender = NonBlockingJSONSender(port=9876)
    receiver = BlockingJSONReceiver(port=9876)
    succ = sender.send_data(SAMPLE_BIG_DATA)
    assert succ
    received_data = receiver.capture_data()
    assert received_data == SAMPLE_BIG_DATA


def test_raise_occupying_socket():
    """Test that raises error if port is occupied."""
    receiver1 = BlockingJSONReceiver(port=9890)
    with pytest.raises(ConnectionAbortedError) as excinfo:
        _ = BlockingJSONReceiver(port=9890)
    assert (
        str(excinfo.value) == "An error occurred during connecting"
        " localhost:9890:"
        " [Errno 98] Address already in use"
    )
    assert isinstance(excinfo.value.__cause__, OSError)
    receiver1.disconnect()
    """Test that raises error if port is occupied."""
    receiver2 = NonBlockingJSONReceiver(port=9891)
    with pytest.raises(ConnectionAbortedError) as excinfo:
        _ = NonBlockingJSONReceiver(port=9891)
    assert (
        str(excinfo.value) == "An error occurred during connecting"
        " localhost:9891:"
        " [Errno 98] Address already in use"
    )
    assert isinstance(excinfo.value.__cause__, OSError)
    receiver2.disconnect()


def test_send_to_opened_captured_and_disconnected_nonblocking_receiver_socket():
    """Verify send fails after receiver disconnects.

    It's here because I found a weird bug, that
    receiver.disconnect only closed the listening socket,
    but kept the conn socket.
    """
    port = 9881
    receiver = NonBlockingJSONReceiver(port=port)
    sender = NonBlockingJSONSender(port=port)
    receiver.capture_data()
    receiver.disconnect()
    succ = sender.send_data(SAMPLE_DATAS[0])
    assert not succ


def test_send_to_opened_captured_and_disconnected_blocking_receiver_socket():
    """Same as above. Without the fix, it would fail."""
    port = 9882
    receiver = BlockingJSONReceiver(port=port)
    sender = NonBlockingJSONSender(port=port)
    sender.send_data(SAMPLE_DATAS[0])
    receiver.capture_data()
    receiver.disconnect()
    succ = sender.send_data(SAMPLE_DATAS[0])
    assert not succ


def test_send_to_connected_but_not_accepted_socket_with_nonblocking_receiver():
    """Handle the case where a socket was connected but not yet accepted.

    Sender will notice the other peer's socket manager has closed, but since
    the sender's socket itself is never accepted, it pops out
    ConnectionResetError, and we should just handle that.
    """
    port = 9883
    receiver = NonBlockingJSONReceiver(port=port)
    sender = NonBlockingJSONSender(port=port)
    # receiver.capture_data()
    receiver.disconnect()
    receiver2 = NonBlockingJSONReceiver(port=port)
    succ = sender.send_data(SAMPLE_DATAS[0])
    assert succ
    data = receiver2.capture_data()
    assert data == SAMPLE_DATAS[0]


def test_send_to_connected_but_not_accepted_socket_with_blocking_receiver():
    """Same as above, and be careful that it will just die if not handle right."""
    port = 9884
    receiver = BlockingJSONReceiver(port=port)
    sender = NonBlockingJSONSender(port=port)
    # receiver.capture_data()
    receiver.disconnect()
    receiver2 = BlockingJSONReceiver(port=port)
    succ = sender.send_data(SAMPLE_DATAS[0])
    assert succ
    data = receiver2.capture_data()
    assert data == SAMPLE_DATAS[0]


if __name__ == "__main__":
    print("HELLO")
