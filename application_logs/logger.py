from datetime import datetime

class App_Logger:
    """
        This package is responsible for log all the details with particular file.
    """
    def __init__(self):
        pass

    def log(self, file_object, log_message):
        """
            Method Name: log\n
            Description: This method log the details\n
            Output: log the details.\n
            On Failure: Raise Exception\n\n

            :param file_object: file name
            :param log_message: messages
            :return: it's helps to log the messages.
        """
        self.now = datetime.now()
        self.date = self.now.date()
        self.current_time = self.now.strftime("%H:%M:%S")
        file_object.write(
            str(self.date) + "/" + str(self.current_time) + "\t\t" + log_message + '\n')  # log the details with time stamp

if __name__ == '__main__':
    pass
