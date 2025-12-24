import socket
import json
import keyboard
import time
HOST = "192.168.1.33"   # 改成顯示電腦的 IP，例如 "192.168.1.123"
PORT = 50008

# Example
TEST_ROWS = [
    [17.68,-43.31,119.41,115.86,-88.3,184.33,0,0,0],
    [14.16,-43.31,119.41,115.86,-88.3,184.33,0,0,0],
    [11.52,-43.31,119.41,115.86,-88.3,184.33,0,0,0],
    [8.87,-43.31,119.41,115.86,-88.3,184.33,0,0,0],
    [6.25,-43.31,119.41,115.86,-88.3,184.33,0,0,0],
    [3.6,-43.31,119.41,115.86,-88.3,184.33,0,0,0],
    [1.41,-43.31,119.41,115.86,-88.3,184.33,0,0,0],
    [-1.24,-43.31,119.41,115.86,-88.3,184.33,0,0,0],
    [-3.43,-43.31,119.41,115.86,-88.3,184.33,0,0,0],
    [-6.5,-43.31,119.41,115.86,-88.3,184.33,0,0,0],
    [-9.15,-43.31,119.41,115.86,-88.3,184.33,0,0,0],
    [-11.35,-43.31,119.41,115.86,-88.3,184.33,0,0,0],
    [-13.99,-43.31,119.41,115.86,-88.3,184.33,0,0,0],
    [-16.19,-43.31,119.41,115.86,-88.3,184.33,0,0,0],
]


def send_traj(rows):
    payload = {
        "type": "traj",
        "unit": "deg",
        "data": rows,
    }
    msg = json.dumps(payload)
    start = time.time()
    with socket.create_connection((HOST, PORT), timeout=3) as s:
        s.sendall(msg.encode("utf-8"))
        resp = s.recv(1024).decode("utf-8", errors="ignore")
        print("Server:", resp.strip())
    end = time.time()
    print(f"Time taken: {end - start:.3f} seconds")



if __name__ == "__main__":
    print(f"Prepared rows: {len(TEST_ROWS)}")
    print("Press '1' to send trajectory. Press 'q' or ESC to quit.")

    sent = False

    while True:
        key = keyboard.read_key(suppress=False)

        if key == "1" and not sent:
            print("[Action] Sending trajectory...")
            send_traj(TEST_ROWS)
            print("[Done] Trajectory sent.")
            sent = True
            print("Press '1' again to re-send, or 'q'/ESC to quit.")
            # 允許重送：把 sent=False 改掉即可
        elif key == "1" and sent:
            # 若你希望每次按 1 都能重送，註解掉 sent 機制即可
            print("[Info] Already sent once. (Remove 'sent' flag to allow re-send.)")
        elif key.lower() == "q" or key == "esc":
            print("Exit.")
            break
