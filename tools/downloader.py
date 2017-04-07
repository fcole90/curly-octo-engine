import os
import sys
import threading
import time
import requests

thread_lock = threading.Lock()


class DownloaderThread(threading.Thread):
    """
    Thread to download files

    name: str
        the name of the thread

    dir: str
        path to the download directory



    downloader: Downloader
        The Downloader which generated the thread.

    download_element: dict
        contains the following:
          - name: the name of the element
          - dir: the dir where to save the images
          - images: a list of str (url) to images

    """

    def __init__(self, downloader, download_element):
        """
        Initializer

        Parameters
        ----------
        downloader: Downloader
            The Downloader which generated the thread.

        download_element: dict
            contains the following:
              - name: the name of the element
              - dir: the dir where to save the images
              - images: a list of str (url) to images
        """
        threading.Thread.__init__(self)
        self.downloader = downloader
        self.download_element = download_element
        self.name = download_element["name"]
        if "dir" in download_element:
            download_dir = os.path.join("images", download_element["dir"])
        else:
            download_dir = self.name
        self.dir = os.path.join(os.getcwd(), download_dir)

        # Create the download path does not exist
        if os.path.exists(self.dir):
            if not os.path.isdir(self.dir):
                raise NotADirectoryError(self.dir)
        else:
            os.makedirs(self.dir, exist_ok=True)

    def run(self):
        """
        Downloads the files relative to the given urls
        """
        print("Downloading " + self.getName() + "...")

        self.download_all()

        # Release procedure
        while not thread_lock.acquire():
            time.sleep(1)
        self.downloader.detach(self)
        thread_lock.release()

    def download(self, url, save_dir, filename):
        """

        Parameters
        ----------
        url: str
            url to the resource to download

        save_dir: str
            path to the download directory

        filename: str
            name of the saved file

        """
        req = Downloader.fault_tolerant_connection(url, self.name)

        # Try to obtain the extension of the resource
        path = self.res_filename(save_dir, filename +
                                 '.' +
                                 req.headers['Content-Type'].split('/')[1])

        # Save the resource
        file = open(path, 'wb')
        for chunk in req.iter_content(100000):
            file.write(chunk)
        file.close()

    def download_all(self):
        """
        Downloads all the links from the list 'images' of the download_element
        """

        # Download each image url
        for counter, url in enumerate(self.download_element["url_list"]):
            print(self.getName() + " downloading " +
                  str(counter + 1) + " of " +
                  str(len(self.download_element["url_list"])))

            self.download(url, self.dir, "img" + str(counter + 1))

    def getName(self):
        """
        Get the name of the thread

        Returns
        -------
        str
            the name of the thread

        """
        return self.name

    def res_filename(self, file_dir, file_name):
        """
        Adapts the name to avoid duplicates

        Parameters
        ----------
        file_dir: str
            path to the downloads dir

        file_name: str
            name of the file

        Returns
        -------
        str
            updated full path file

        """
        counter = 1

        path = os.path.join(file_dir, file_name)

        while os.path.exists(path):
            name_ext = os.path.splitext(file_name)
            rename = name_ext[0] + "(" + str(counter) + ")" + name_ext[1]
            path = os.path.join(file_dir, rename)
            counter += 1

        return path


class Downloader:
    """Spawns DownloaderThreads

    element_list: list
        dictionary of elements

    sleep_delay: int, optional
        waiting time before retrying connecting
        (default: 5)

    max_threads: int, optional
        maximum number of active threads
        (default: 1)

    active_threads: int
        number of currently active threads

    threads_list: list of DownloaderThreads
        list of currently active threads

    """

    def __init__(self, element_list, max_threads=1, sleep_delay=5):
        """
        Initializer

        Parameters
        ----------
        element_list: list
            dictionary of elements

        sleep_delay: int, optional
            waiting time before retrying connecting
            (default: 5)

        max_threads: int, optional
            maximum number of active threads
            (default: 1)
        """
        self.sleep_delay = sleep_delay
        self.max_threads = max_threads
        self.element_list = element_list

        self.active_threads = 0
        self.threads_list = []

    def download_list(self):
        """
            Spawns a thread for each element
        """
        for element in self.element_list:
            while self.active_threads >= self.max_threads:
                time.sleep(self.sleep_delay)

            self.active_threads += 1
            dt = DownloaderThread(self, element)
            self.threads_list.append(dt)
            dt.start()

        for thread in self.threads_list:
            thread.join()

    def detach(self, element):
        """
            Detaches a thread

        Parameters
        ----------
        element: DownloaderThread
            the element to detach

        """
        self.active_threads -= 1
        self.threads_list.remove(element)
        print(element.getName() + " released...")

    @staticmethod
    def fault_tolerant_connection(url, name, max_retrials=50, delay=5, headers=None):
        """
        Retries to connect if connection fails.

        After each failure i, it waits 2^i seconds before retrying.

        Parameters
        ----------
        url: str
        name: str
        max_retrials: int, optional
            (default: 50)

        delay: int, optional
            (default: 5)
        headers: str, optional
            (default: None)

        Returns
        -------
        request
        """
        for i in range(max_retrials):
            try:
                req = requests.get(url, headers=headers)
                return req

            except:
                if i < max_retrials - 1:
                    sec = delay * (2 ** i)
                    time.sleep(sec)
                    print(name + " failed.. Retrying in " + str(sec) + " (at " +
                          time.strftime("%H:%M:%S", time.localtime(time.time() + sec)) + ")")
                else:
                    raise Exception(name + " failed to connect..")

    def get(self):
        """Downloads all the resources"""
        self.download_list()


def test():
    url_list = []

    for i in range(6)[1:]:
        url_list.append("http://www.periodictable.com/GridImages/big/" + str(i) + ".JPG")

    element_list = []

    for i in range(10):
        element_list.append(
            {
                "name": "Downloader" + str(i),
                "dir": str(i),
                "url_list": url_list
            }
        )

    d = Downloader(element_list, 5)
    d.get()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Please, give urls as arguments..")
    elif sys.argv[1] in ["--test", "-t"]:
        test()
    else:
        element = {
            "name": "download",
            "url_list": sys.argv[1:]
        }

        d = Downloader([element])
        d.get()
