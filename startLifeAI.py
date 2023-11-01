#!/usr/bin/env python
# -*- coding: utf-8 -*-
import subprocess
import threading
import queue
import time
import json
import signal

class ProgramManager:
    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        self.processes = {}
        self.threads = {}
        self.command_queue = queue.Queue()
        self.should_be_running = set()

    def start_program(self, name):
        program_info = self.config.get(name)
        if program_info:
            process = subprocess.Popen(program_info['args'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.processes[name] = process
            self.should_be_running.add(name)
            print(f"Started program {name}")
            self.threads[name] = threading.Thread(target=self.monitor_program, args=(name,))
            self.threads[name].start()
        else:
            print(f"No configuration found for program {name}")

    def monitor_program(self, name):
        process = self.processes.get(name)
        while process and self.should_be_running.issuperset({name}):
            returncode = process.poll()
            if returncode is not None:  # Process has exited
                print(f"Program {name} exited with return code {returncode}")
                # If the program should be running, restart it
                if name in self.should_be_running:
                    print(f"Restarting program {name}")
                    self.start_program(name)
                break
            else:
                time.sleep(1)  # Poll every second

    def stop_program(self, name, force_kill_timeout=10):
        if name in self.processes:
            self.should_be_running.discard(name)
            process = self.processes[name]
            process.terminate()  # Send terminate signal
            print(f"Attempting to stop program {name}")

            # Wait for process to terminate gracefully
            for i in range(force_kill_timeout):
                if process.poll() is not None:
                    print(f"Program {name} terminated gracefully")
                    break
                time.sleep(1)
            else:
                # Force kill if not terminated after timeout
                print(f"Force killing program {name}")
                process.kill()  # Send SIGKILL
                process.wait()  # Ensure resources are cleaned up

            self.processes.pop(name, None)
            print(f"Stopped program {name}")

        def run(self):
            # Register signal handlers
            signal.signal(signal.SIGINT, self.signal_handler)
            signal.signal(signal.SIGTERM, self.signal_handler)

        for name in self.config:
            self.start_program(name)

        while True:
            try:
                command = self.command_queue.get_nowait()
                self.process_command(command)
            except queue.Empty:
                pass
            time.sleep(1)

    def process_command(self, command):
        name = command['name']
        if command['action'] == 'stop':
            self.stop_program(name)
        elif command['action'] == 'start':
            self.start_program(name)
        elif command['action'] == 'restart':
            self.stop_program(name)
            time.sleep(3)  # Give some time before restarting
            self.start_program(name)
        elif command['action'] == 'status':
            running = name in self.processes and self.should_be_running.issuperset({name})
            print(f"Program {name} is running: {running}")
        elif command['action'] == 'list':
            print("Running programs:")
            for prog in self.should_be_running:
                print(f"  {prog}")
        elif command['action'] == 'exit':
            for name in list(self.processes.keys()):
                self.stop_program(name)
            exit(0)
        else:
            print(f"Unknown command {command['action']}")

    def command(self, action, name):
        self.command_queue.put({'action': action, 'name': name})

def signal_handler(signal_received, frame):
    # Handle any cleanup or resource releasing here
    print(f"Signal {signal_received} received, shutting down.")
    for name in list(manager.processes.keys()):  # Use list to avoid dictionary size change during iteration
        manager.stop_program(name)
    exit(0)

if __name__ == '__main__':
    manager = ProgramManager('config.json')

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    run_thread = threading.Thread(target=manager.run)
    run_thread.start()

    # Example command line interaction
    while True:
        cmd_input = input("Enter command: ")
        if cmd_input.lower() == 'exit':
            # If the input is 'exit', we should signal the run thread to stop
            for name in list(manager.processes.keys()):
                manager.stop_program(name)
            break
        cmd, prog = cmd_input.split()
        manager.command(cmd, prog)

    run_thread.join()  # Now it's safe to join since we've handled 'exit'
