import colorthief
import os
import sys


class ImageDataExtractor:
    """
    Extracts palettes from the given images
    """

    palette = None

    def __init__(self, file_path):
        """
        Initializer

        Parameters
        ----------
        file_path: str
        """

        self.file_path = file_path
        self.img = open(file_path, 'rb')

    def get_palette(self, color_count=10, quality=1):
        """

        Parameters
        ----------
        color_count: int, optional
            amount of colors for the palette
            (default: 10)

        quality: int, optional
            quality of the computation
            (default: 10)

        Returns
        -------
        list of tuple of int
            a list in the form [(r, g, b), (r, g, b), ...]

        """
        ct = colorthief.ColorThief(self.img)
        self.palette = ct.get_palette(color_count, quality)
        return self.palette

    def print_palette(self, mode='txt', output_path=None):
        """
        Prints a palette according to the given mode

        Parameters
        ----------
        mode: str, optional
            mode of print among ['txt', 'screen', 'html']
            (default: 'txt')

            Note:
            If no output_path is given, the mode becomes screen!

        output_path: str, optional
            the path to the file where to write the palette

        """
        if output_path:
            output_path = output_path
            output = open(output_path, "wt")

        if not self.palette:
            self.get_palette()

        if not output_path or mode == "screen":
            print(str(self.palette))

        elif mode == 'txt':
            output.write(str(self.palette))

        elif mode == 'html':
            html = self.print_HTML_color(self.palette)
            output.write(html)

    def print_HTML_color(self, palette):
        """
        Format the palette as an html document

        Parameters
        ----------
        palette: list of tuple of int
            a palette of colors

        Returns
        -------

        """
        text = "<!DOCTYPE html>\n" + "<html>\n" + "<body>\n\n"

        for color in palette:
            text += ("<h2 style=background-color:"
                    "rgb("
                    ""  + str(color[0]) + ','
                    "" + str(color[1]) + ','
                    "" + str(color[2]) + ')\n'
                    ">" + str(color) + "</h2>"
            )

        text += "</body>\n</html>"
        return text


def test():
    full_path = os.path.join(os.getcwd(), "test", "test.jpeg")
    out_path_html = os.path.join(os.getcwd(), "test", "test.html")
    out_path_txt = os.path.join(os.getcwd(), "test", "test.txt")

    id_ex = ImageDataExtractor(full_path)
    id_ex.print_palette("screen")
    id_ex.print_palette("txt", out_path_txt)
    id_ex.print_palette("html", out_path_html)


def main():
    for path in sys.argv[1:]:
        id_ex = ImageDataExtractor(path)
        id_ex.print_palette("screen")
        id_ex.print_palette("txt", os.path.splitext(path)[0] + ".txt")
        id_ex.print_palette("html", os.path.splitext(path)[0] + ".html")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Missing argument")
    elif sys.argv[1] in ["-t", "--test"]:
        test()
    else:
        main()


