import socket
import threading

class Server:

    def __init__(self, host, port, q):
        self.host = host
        self.port = port
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((self.host, self.port))
        self.server.listen(5)
        self.q = q
        self.client_threads = []
        listener_thread = threading.Thread(target=self.listen_for_clients, args=())
        listener_thread.start()

    def listen_for_clients(self):
        print("Listening...")

        while True:
            client, addr = self.server.accept()
            print("Accepted Connection from: {}: {}".format(addr[0], addr[1]))
            t = threading.Thread(target=self.data_sender, args=())
            self.client_threads.append((t, client, addr))
            t.start()

    def data_sender(self):
        while True:
            try:
                while True:
                    data = str(self.q.get()) + ','
                    for client_thread, client, addr in self.client_threads:
                        # print("Sending {} to {}".format(data, addr), file=sys.stderr)
                        client.send(data.encode())
            except:
                pass
