import threading
import time

__delay__ = 1
__micro_delay__ = 0.1


class RunBeforeCompletedError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class ThreadManager:

    def __init__(self):
        self.failures = []
        self.retrial = []
        self.running_ops = []
        self.completed_ops = []
        self.waiting_list_counter = -1
        self.failures_lock = threading.Lock()
        self.retrial_lock = threading.Lock()
        self.running_lock = threading.Lock()
        self.waiting_list_counter_lock = threading.Lock()
        self.last_id = 1

        self.reset_waiting_list_counter()

    def reset_waiting_list_counter(self):
        while not self.waiting_list_counter_lock.acquire():
            time.sleep(__delay__)
        self.waiting_list_counter = 0
        self.waiting_list_counter_lock.release()

    def get_unique_id(self):
        self.last_id += 1
        return self.last_id - 1

    def decrease_waiting_list_counter(self, amount=1):
        while not self.waiting_list_counter_lock.acquire():
            time.sleep(__delay__)
        if amount < 0:
            raise ValueError("Amount must be greater than 0")
        if self.waiting_list_counter < amount:
            raise RuntimeError("Amount cannot be greater than list counter - "
                               "Amount: " + str(amount) + " Counter: " + str(self.waiting_list_counter))
        self.waiting_list_counter -= amount
        # print("Counter decreased: " + str(self.waiting_list_counter))
        self.waiting_list_counter_lock.release()

    def set_waiting_list_counter(self, amount):
        while not self.waiting_list_counter_lock.acquire():
            time.sleep(__delay__)
        if not self.waiting_list_counter == 0:
            raise RuntimeError("Counter needs reset before being set")
        self.waiting_list_counter = amount
        self.waiting_list_counter_lock.release()

    def add_retrial(self, op):
        self.retrial.append(op)

    def add_failure(self, op):
        self.retrial.append(op)

    def clear_retrials(self):
        self.decrease_waiting_list_counter(len(self.retrial))
        self.failures.extend(self.retrial)
        self.retrial = []

    def add_running(self, op):
        self.running_ops.append(op)

    def add_completed_op(self, op):
        self.completed_ops.append(op)

    def remove_retrial(self, op):
        op_id = op.get_id()
        for listed_op in self.retrial:
            if listed_op.get_id() == op_id:
                self.retrial.remove(listed_op)

    def remove_running(self, op):
        op_id = op.get_id()
        for listed_op in self.running_ops:
            if listed_op.get_id() == op_id:
                self.running_ops.remove(listed_op)

    def is_error_happened(self):
        if self.retrial:
            return True
        else:
            return False

    def all_ops_finished(self):
        if self.waiting_list_counter == 0:
            return True
        if self.waiting_list_counter > 0:
            return False
        if self.waiting_list_counter < 0:
            raise RuntimeError("Counter is in illegal state: " + str(self.waiting_list_counter))

    def get_completed_ops(self):
        return self.completed_ops

    def get_failures(self):
        return self.failures

    def print_failures(self):
        for op in self.retrial:
            print(str(op.get_id()) + ' - "' + str(op.get_error().__class__) + ": " + str(op.get_error()))

    def run_all(self, op_list, restore_operation=None, retrial_mode=False, max_threads=10):
        """
        Runs all the given operations, each in a different thread.

        Parameters
        ----------
        op_list: list of Operation
            list of operations to run

        restore_operation: Operation, optional
            defaults to None
            If any Operation belonging to the op_list should fail
            this Operations will be run to try to go back to a
            safe state. This is valuable especially when working with
            API that require authentications subject to expiration,
            such as Spotify's. In such situations the restore operation
            can be used to renew the OAuth token and resume using the API.

        retrial_mode: bool, optional
            defaults to False
            Flag to handle restore operations, should not be of any use
            out of internal mechanics of the class.

        max_threads: int, optional
            defaults to 10
            Let the user choose the maximum amount of concurrent
            threads. After the top has been reached, it will
            wait until a thread ends before running another.

        """
        if max_threads < threading.active_count():
            raise ValueError("Max threads should be greater than active threads."
                             "Currently active: " + str(threading.active_count()))
        if not retrial_mode:
            self.set_waiting_list_counter(len(op_list))
        for op in op_list:
            if op.get_id() == 0:
                op.op_id = self.get_unique_id()
            while threading.active_count() >= max_threads:
                time.sleep(__micro_delay__)
            while self.is_error_happened() and not retrial_mode:
                if restore_operation:
                    self.run_op(restore_operation)
                    restore_operation = restore_operation.clone(restore_operation.error, restore_operation.get_return_value())
                    self.run_all(self.retrial, None, retrial_mode=True)
                    self.clear_retrials()
                else:
                    print("Warning: Could not complete every operation..")
                    self.print_failures()
                    print("\n")
                    self.clear_retrials()

            if retrial_mode:
                while not self.retrial_lock.acquire():
                    time.sleep(__delay__)
                self.remove_retrial(op)
                self.retrial_lock.release()

            self.run_op(op)

        for op in self.running_ops:
            op.join()

    def run_op(self, operation):

        while not self.running_lock.acquire():
            time.sleep(__delay__)
        self.add_running(operation)
        self.running_lock.release()
        print("Before running: " + str(operation.get_id()))
        operation.start()


class Operation(threading.Thread):
    """
    Operation to be performed in a concurrent thread.
    """

    def __init__(self, manager, func, args=None, op_id=0):
        """
        Parameters
        ----------
        manager: ThreadManager
            manager who run this operation
        func: function
            function to be run
        args: dict
            arguments of the function
        op_id: int, optional
            operation id

        Returns
        -------
        unknown
            return value of the given function func
        """
        threading.Thread.__init__(self)
        self.manager = manager
        self.op_id = op_id
        self.func = func
        self.args = args
        self.error = None
        self.return_value = None

    def clone(self, error=None, value=None):
        clone = Operation(self.manager, self.func, self.args, self.get_id()+100000)
        clone.set_error(error)
        clone.return_value = self.return_value
        clone._flag = False
        return clone

    def get_id(self):
        return self.op_id

    def get_error(self):
        return self.error

    def get_return_value(self):
        return self.return_value

    def set_error(self, error):
        self.error = error

    def run(self):
        try:
            print("Starting op: " + str(self.get_id()))
            self.return_value = self.func(self.args)
            while not self.manager.running_lock.acquire():
                time.sleep(__delay__)
            self.manager.remove_running(self)
            self.manager.add_completed_op(self)
            self.manager.running_lock.release()
            self.manager.decrease_waiting_list_counter()
        except Exception as e:
            while not self.manager.running_lock.acquire():
                time.sleep(__delay__)
            self.manager.remove_running(self)
            self.manager.running_lock.release()

            if self.get_error() is None:
                while not self.manager.retrial_lock.acquire():
                    time.sleep(__delay__)
                self.manager.add_retrial(self.clone(e))
                self.manager.retrial_lock.release()
            else:
                while not self.manager.failures_lock.acquire():
                    time.sleep(__delay__)
                self.manager.add_failure(self)
                self.manager.failures_lock.release()


