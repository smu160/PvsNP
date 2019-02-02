"""
The client/server classes that keep multiple VLC python bindings players
synchronized.

Author: Saveliy Yusufov, Columbia University, sy2685@columbia.edu
Date: 25 January 2019
"""

import os
import platform
import socket
import threading
import sys
import logging
from concurrent import futures

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# Create formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Add formatter to ch
ch.setFormatter(formatter)

# Add ch to logger
logger.addHandler(ch)


class Server:
    """Data sender server"""

    def __init__(self, host, port, data_queue):

        if platform.system() == "Windows":

            # Create a TCP/IP socket
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

            # Bind the socket to the port
            logger.info("Server started on %s port %s", host, port)
            self.sock.bind((host, port))

        else:
            server_address = "./uds_socket"

            # Make sure the socket does not already exist
            try:
                os.unlink(server_address)
            except OSError:
                if os.path.exists(server_address):
                    raise

            # Create a UDS socket
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

            # Bind the socket to the address
            logger.info("Server starting up on %s", server_address)
            self.sock.bind(server_address)

        # Listen for incoming connections
        self.sock.listen(5)

        self.clients = set()
        self.data_queue = data_queue
        listener_thread = threading.Thread(target=self.listen_for_clients, args=())
        listener_thread.daemon = True
        listener_thread.start()

    def listen_for_clients(self):
        logger.info("Now listening for clients")
        t = threading.Thread(target=self.data_sender, args=())
        t.daemon = True
        t.start()

        while True:
            client, _ = self.sock.accept()
            logger.info("Accepted Connection from: %s", client)
            self.clients.add(client)

    def data_sender(self):
        while True:
            data = "{},".format(self.data_queue.get())

            with futures.ThreadPoolExecutor(max_workers=5) as ex:
                for client in self.clients.copy():
                    ex.submit(self.sendall, client, data.encode())

    def sendall(self, client, data):
        """Wraps socket module's `sendall` function"""
        try:
            client.sendall(data)
        except socket.error:
            logger.exception("Connection to client: %s was broken!", client)
            client.close()
            self.clients.remove(client)


class Client:
    """Data receiver client"""

    def __init__(self, address, port, data_queue):
        self.data_queue = data_queue

        if platform.system() == "Windows":

            # Create a TCP/IP socket
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

            # Connect the socket to the port where the server is listening
            logger.info("Connecting to %s port %s", address, port)
            self.sock.connect((address, port))
        else:

            # Create a UDS socket
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

            # Connect the socket to the port where the server is listening
            server_address = "./uds_socket"
            logger.info("New client connecting to %s", server_address)

            try:
                self.sock.connect(server_address)
            except socket.error:
                logger.exception()
                sys.exit(1)

        thread = threading.Thread(target=self.data_receiver, args=())
        thread.daemon = True
        thread.start()

    def data_receiver(self):
        """Handles receiving, parsing, and queueing data"""
        logger.info("New data receiver thread started.")

        try:
            while True:
                data = self.sock.recv(4096)
                if data:
                    data = data.decode()

                    for char in data.split(','):
                        if char:
                            if char == 'd':
                                self.data_queue.queue.clear()
                            else:
                                self.data_queue.put(char)
        except:
            logger.exception("Closing socket: %s", self.sock)
            self.sock.close()
            return
