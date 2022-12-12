import PySimpleGUI as sg
import subprocess

def bridge_window(command_line,title):
    """The bridge function

    Args:
        command_line: The command_line called
        title: The title of the intermediate window

    Returns: generate new intermediate window
    """
    layout = [[sg.Frame(title, font='Any 15', layout=[[sg.Output(size=(80, 15), font='Courier 10')]])],
              [sg.Button('Continue', size = 20, pad=((0, 0), 5), bind_return_key=False)]]

    window = sg.Window('Intermediate Process Generator', layout, auto_size_text=False, auto_size_buttons=False, default_element_size=(20,1), text_justification='right')

    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED:
            window.close()
            break

        if event == 'Continue':
            try:
                print(command_line)
                window.refresh()
                run_command(command_line, window=window)
                print('**** DONE ****')
                sg.Popup('Process Over or Manually Stop')
                window.close()
            except:
                window.close()



def run_command(cmd, window=None):
    """run shell command

    Args:
        cmd: command to execute
        title: The title of the intermediate window

    Returns: return code from command, command output
    """

    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in p.stdout:
        line = line.decode().rstrip()
        print(line)
        if window:
            window.Refresh()

    return None
