# Import libraries
import PySimpleGUI as sg
import os

import pandas as pd

from utility import img_filter, get_img_data

def get_view_win(folder, folder_, type):
    """Create a window to view images

    Args:
        folder: The path of the selected folder
        folder_: The path of the result folder
        type: whether segmented or not

    Returns: a defined window
    """
    # Get the list of image files in the selected folder and count the number of files
    fnames = img_filter(folder)
    num_files = len(fnames)

    # Initialize to the first file in the list
    filename = os.path.join(folder, fnames[0])
    image_elem = sg.Image(data=get_img_data(filename, first=True))
    common_preds = pd.read_csv(folder_ + "/predict_result" + type + ".csv").iloc[:,5]
    filename_display_elem = sg.Text("The prediction result: " + common_preds[0], size=(30, 1), font='Calibri 18')
    file_num_display_elem = sg.Text('File 1 of {}'.format(num_files), size=(15, 1))

    # Define the layout
    col = [
        [filename_display_elem],
        [image_elem]
    ]

    col_files = [
        [sg.Listbox(values=fnames, change_submits=True, size=(40, 30), key='listbox')],
        [sg.Button('Next', size=(8, 2)), sg.Button('Prev', size=(8, 2)), file_num_display_elem]
    ]

    layout = [[sg.Column(col_files), sg.Column(col)]]

    # Create the window
    window = sg.Window('Image Browser', layout, return_keyboard_events=True,
                       use_default_focus=False)

    # Create a loop to read the user input and display images
    i = 0
    while True:
        event, values = window.read()
        print(event, values)

        if event == sg.WIN_CLOSED:
            break
        elif event in ('Next', 'MouseWheel:Down', 'Down:40', 'Next:34'):
            i += 1
            if i >= num_files:
                i -= num_files
            filename = os.path.join(folder, fnames[i])
        elif event in ('Prev', 'MouseWheel:Up', 'Up:38', 'Prior:33'):
            i -= 1
            if i < 0:
                i = num_files + i
            filename = os.path.join(folder, fnames[i])
        elif event == 'listbox':
            # Input from the listbox
            f = values["listbox"][0]
            filename = os.path.join(folder, f)
            i = fnames.index(f)
        else:
            filename = os.path.join(folder, fnames[i])

        # Update the window with a new image
        image_elem.update(data=get_img_data(filename, first=True))
        # Display the prediction
        filename_display_elem.update("The prediction result: " + common_preds[i])
        # Update the page display
        file_num_display_elem.update('File {} of {}'.format(i + 1, num_files))

    window.close()