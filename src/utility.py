# Import libraries
from PIL import Image, ImageTk
import os
import io

# Define functions
def img_filter(folder):
    """Create a sub list of image files in the selected folder

    Args:
        folder: The path of the selected folder

    Returns:
        list: The list of filenames
    """

    # Supported image types
    img_types = (".png", ".jpg", "jpeg", ".tif", ".tiff", ".bmp")

    file_list = os.listdir(folder)
    fnames = [f for f in file_list if os.path.isfile(
        os.path.join(folder, f)) and f.lower().endswith(img_types)]

    return fnames


def num_files(folder):
    """Count the number of image files in the selected folder

    Args:
        folder: The path of the selected folder

    Returns:
        int: The number of image files
    """

    fnames = img_filter(folder)
    count = len(fnames)

    return count


def get_img_data(f, maxsize: tuple = (512, 512), first: bool = False):
    """Generate image data using PIL
    Args:
        f: The path of the image file
        maxsize (tuple): The maximum size of the output image
        first (bool): Whether tkinter is active the first time
        first (bool): Whether tkinter is active the first time

    Returns: a Tkinter-compatible photo image
    """
    # Import image
    img = Image.open(f)
    # Limit the image size
    img.thumbnail(maxsize)
    if first:
        # tkinter is inactive the first time
        bio = io.BytesIO()
        img.save(bio, format="PNG")
        del img
        return bio.getvalue()
    return ImageTk.PhotoImage(img)
