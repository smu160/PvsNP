import os
import platform
import socket
import threading
import sys

class Server:

    def __init__(self, host, port, q):

        if platform.system() == "Windows":

            # Create a TCP/IP socket
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

            # Bind the socket to the port
            server_address = (host, port)
            print("Server starting up on {} port {}".format(server_address[0], server_address[1]))
            self.sock.bind(server_address)

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
            print("Starting up on {}".format(server_address))
            self.sock.bind(server_address)

        # Listen for incoming connections
        self.sock.listen(5)

        self.q = q
        self.client_threads = {}
        listener_thread = threading.Thread(target=self.listen_for_clients, args=())
        listener_thread.daemon = True
        listener_thread.start()

    def listen_for_clients(self):
        print("Listening for clients...")

        while True:
            client, addr = self.sock.accept()
            print("Accepted Connection from: {}".format(client))
            t = threading.Thread(target=self.data_sender, args=())
            self.client_threads[client] = (t, addr)
            t.daemon = True
            t.start()

    def data_sender(self):
        while True:
            try:
                while True:
                    data = "{},".format(self.q.get())
                    for client, _ in self.client_threads.items():
                        client.sendall(data.encode())
            except (BrokenPipeError, socket.error) as e:
                print(e, file=sys.stderr)
                print("Connection to client: {} was broken!".format(client), file=sys.stderr)
                print(self.client_threads, file=sys.stderr)
                client.close()
                del self.client_threads[client]