# Import libraries
import PySimpleGUI as sg
import os
from progress_meter import bridge_window
from utility import img_filter, num_files
from figure_viewer import get_view_win
import time


def main():
    # Change the color scheme of the window
    sg.theme("SystemDefaultForReal")

    # The layout for the window
    layout = [
        [sg.Text("Diagnosis of Covid-19, pneumonia and tuberculosis", font='Calibri 20')],
        [sg.Text("- based on chest X ray images", font='Calibri 18')],
        [sg.Text("Choose a folder containing images to classify:", font='Calibri 14')],
        [sg.InputText(size=(65, 1), enable_events=True, key='-INPUT-'), sg.FolderBrowse(font='Calibri 14')],
        [sg.Checkbox("Segmentation: use segmented lung images for classification", font='Calibri 14', default=False,
                     key="segment")],
        [sg.Text("Save to", font='Calibri 14'), sg.InputText(size=(65, 1), enable_events=True, key='-OUTPUT-'),
         sg.FolderBrowse(font='Calibri 14')],
        [sg.Button("Classify", font='Calibri 14', size=8), sg.Button("Exit", font='Calibri 14', size=8)]
    ]

    # Create the window
    win1 = sg.Window("pyCXR", layout)

    # Define another window, called win2 which is initially inactive., to display classified images
    win2_active = False

    # unet
    unet_dir = "./models/u_net.h5"

    # Read inputs from the window
    while True:
        event, values = win1.read()

        if event == "Exit" or event == sg.WIN_CLOSED:
            win1.close()
            break
        elif event == "Classify":
            if values["-INPUT-"] == "":
                # The input folder has not been chosen yet
                sg.popup("Please choose an input folder first")
            elif not os.path.exists(values["-INPUT-"]):
                sg.popup("The input path does not exist")
            elif num_files(values["-INPUT-"]) == 0:
                # The input folder contains no files or files of wrong types
                sg.popup('No image files in folder')
            else:
                if values["-OUTPUT-"] == "":
                    # The input folder has not been chosen yet
                    sg.popup("Please choose an output folder first")
                elif not os.path.exists(values["-OUTPUT-"]):
                    sg.popup("The output path does not exist")
                elif values["-INPUT-"] == values["-OUTPUT-"]:
                    sg.popup("Please choose an different path for input/output folder")
                else:
                    fnames = img_filter(values["-INPUT-"])
                    if values["segment"] == True:
                        print(0)
                        # start segmented
                        command = "python unet_predict.py" + " --dr " + values["-INPUT-"] + \
                                  " --out_dr " + values['-OUTPUT-'] + " --unet_dr " + unet_dir
                        bridge_window(command, "Segmentation...")
                        if not num_files(values['-OUTPUT-']) == 0:
                            if not win2_active:
                                win2_active = True
                                # Obtain the prediction
                                command = "python classification_predict.py" + " --dr " + values["-INPUT-"] + \
                                          " --out_dr " + values['-OUTPUT-'] + \
                                          " --type " + "_seg"
                                bridge_window(command, "Classfication...")
                                time.sleep(0.01)
                                get_view_win(values['-OUTPUT-'], values['-OUTPUT-'], "_seg")
                                win2_active = False
                    else:
                        if not win2_active:
                            win2_active = True
                            # Obtain the prediction
                            command = "python classification_predict.py" + " --dr " + values["-INPUT-"] + \
                                      " --out_dr " + values['-OUTPUT-']
                            bridge_window(command, "Classfication...")
                            time.sleep(0.01)
                            get_view_win(values["-INPUT-"], values['-OUTPUT-'], "")
                            win2_active = False

    win1.close()


if __name__ == '__main__':
    main()
