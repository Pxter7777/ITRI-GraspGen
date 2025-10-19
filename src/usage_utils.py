import time
import torch
import logging

class UsageInspector:
    def __init__(self, enabled=False, log_file="usage_log.txt"):
        self.enabled = enabled
        if not self.enabled:
            return
        self.log_file = log_file
        self.timers = {}
        self.vram_reports = {}
        
        # Create a logger that writes to a file
        self.logger = logging.getLogger('UsageInspector')
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        
        # Prevent adding handlers multiple times if the class is instantiated more than once
        if not self.logger.handlers:
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.logger.info("--- New Usage Inspection Session ---")

    def start(self, name):
        if not self.enabled:
            return
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.timers[name] = time.perf_counter()
        self.logger.info(f"Starting: {name}")
        print(f"Starting: {name}")

    def end(self, name):
        if not self.enabled:
            return
        if name not in self.timers:
            self.logger.warning(f"Timer for '{name}' was not started.")
            return

        elapsed_time = time.perf_counter() - self.timers[name]
        
        vram_usage = 0
        if torch.cuda.is_available():
            vram_usage = torch.cuda.max_memory_allocated() / (1024**2)  # In MB
        
        self.vram_reports[name] = vram_usage
        
        log_message = f"Finished: {name} | Time: {elapsed_time:.4f}s | Peak VRAM: {vram_usage:.2f} MB"
        self.logger.info(log_message)
        print(log_message)

    def report(self):
        if not self.enabled:
            return
        # This could be expanded to provide a summary report.
        self.logger.info("--- End of Session ---")
