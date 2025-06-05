import logging
import atexit

class QueueHandler(logging.handlers.QueueHandler):
    def __init__(self, queue):
        super().__init__(queue)

        #logging.handlers.QueueHandler requires
        #us to manually start a thread and register
        #a callback on exit, we do it in this child
        #class' constructor

        self.listener.start()
        atexit.register(self.listener.stop)
