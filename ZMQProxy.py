#!/usr/bin/env python

# Proxy.py
import zmq
import threading

def run_proxy():
    context = zmq.Context()
    
    # Subscriber socket to receive messages from publishers
    frontend = context.socket(zmq.SUB)
    frontend.bind('tcp://*:5999')
    frontend.setsockopt_string(zmq.SUBSCRIBE, '')  # Subscribe to all topics
    
    # Publisher socket to send messages to subscribers
    backend = context.socket(zmq.PUB)
    backend.bind('tcp://*:6000')
    
    zmq.proxy(frontend, backend)  # Start the proxy

if __name__ == "__main__":
    run_proxy()

