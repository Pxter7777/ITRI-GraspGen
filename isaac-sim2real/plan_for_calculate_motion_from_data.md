# plan_for_calculate_motion_from_data

Mostly works like sync_with_ROS2.py.


- Instead of listening commands from socket, we directly read files in data/order_experiment_data
- Because the startup would take a while, CLI is more preferred. Prompt like: `please provide class:` and `please provide name:`, and let me input in the terminal.

