import os
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
import tkinter as tk
import time
import datetime as dt
from itertools import compress
from PIL import Image, ImageTk
import cv2

import sys
sys.path.append("...")
# print(sys.path)
import settings

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
path = ROOT_DIR

class BaggageHandlingApp:
    def __init__(self, master):

        self.frame_style = ttk.Style()
        self.frame_style.configure('new.TFrame', background='#000000')

        self.button_style = ttk.Style()
        self.button_style.configure('B1.TButton', background='black', foreground='black',borderwidth=0, bordercolor='black',
                                    compound="center", highlightthickness=0, relief="flat", activebackground='black')
        self.label_style = ttk.Style()
        self.label_style.configure('L1.TLabel', foreground='white', background='#000000')
        self.logo_style = ttk.Style()
        self.logo_style.configure('L2.TLabel', foreground='black', background='#000000')

        self.combobox_style = ttk.Style()
        self.combobox_style .configure('Combo1.TCombobox', foreground='white', background='#000000')
        self.chckbutton_style = ttk.Style()
        self.chckbutton_style.configure('Chck1.TCheckbutton', foreground='white', background='#000000', font=("helvetica", 18))

        self.content_frame = ttk.Frame(master, style='new.TFrame')
        self.content_frame.pack(fill=BOTH, expand=True)

        self.label1 = ttk.Label(self.content_frame, text="The bag should be repositioned:",font=("helvetica", 20),
                                style='L1.TLabel').grid(row=0, column=0, columnspan=2, padx=5, pady=10,sticky='sw')
#######################################################################################################################
        # read event
        # init_read_event(self)

########################################################################################################################

        self.button1_file = os.path.join(path, "ForceIn_button.png")
        self.ForceIn_image = PhotoImage(file=self.button1_file)
        self.ForceIn_image = self.ForceIn_image.subsample(2,2)
        self.ForceInbutton = ttk.Button(self.content_frame, image=self.ForceIn_image, style='B1.TButton',command=self.reason_window)
        self.ForceInbutton.grid(row=2, column=0,  padx=5, pady=25, sticky='se')


        self.button2_file = os.path.join(path, "check_button.png")
        self.check_image = PhotoImage(file=self.button2_file)
        self.check_image = self.check_image.subsample(2,2)
        self.checkbutton = ttk.Button(self.content_frame, image=self.check_image, style='B1.TButton',command=self.user_checked)
        self.checkbutton.grid(row=2, column=1,  padx=200, pady=25,sticky='se')
        # state = DISABLED
########################################################################################################################
        self.footer_frame = ttk.Frame(master,style='new.TFrame')
        self.footer_frame.pack(fill=BOTH)

        self.time_and_date = ttk.Label(self.footer_frame,
                                text=f"{dt.datetime.now():%a, %b %d %Y} - " + f"{time.strftime('%H:%M:%S')}",
                                style='L1.TLabel',
                                font=("helvetica", 14)).grid(row=0, column=0, padx=20, pady=5,sticky='sw')
        self.mylogo_file = os.path.join(ROOT_DIR, "logo.png")
        self.logo = PhotoImage(file=self.mylogo_file)
        self.logo = self.logo.subsample(1,1)
        self.daifuku = ttk.Label(self.footer_frame, image=self.logo,
                                 style='L2.TLabel')
        self.daifuku.grid(row=0, column=1,rowspan=2,sticky='se', padx=100, pady=0)

        load = Image.open("/home/don/code/BagAnalysis/3DImaging/GPU_based_solution/Deeplearning/detectron2/detectron2/gui/image_gui.png")
        load = load.resize((250, 250), Image.ANTIALIAS)
        render = ImageTk.PhotoImage(load)
        self.img = ttk.Label(self.content_frame, image=render)
        self.img.image = render
        self.img.place(x=500, y=0)


########################################################################################################################

    def user_checked(self):
        # messagebox.showinfo(title="System info", message="Checked by user!")
        # camera 2 detection
        settings.check_button = True

    def user_force_in():
        settings.force_in_button = True
        # messagebox.showinfo(title="System info", message="Force in by user!")

    def reason_window(self):


        def write_log(text, file):
            try:
                f = open(file, "a+")  # 'a' will append to an existing file if it exists
                f.write("{}\n".format(text))  # write the text to the logfile and move to next line
            except FileNotFoundError:
                f = open(file, "w")  # 'a' will append to an existing file if it exists
                f.write("{}\n".format(text))  # write the text to the logfile and move to next line


        def get_reason():
            output = []
            indx = []
            for i in range(len(self.reason)):
                indx.append(self.reason[i].get())
            output = list(compress(self.reasons_values, indx))

            # log_info = str(f"{im_id}:"+f"{dt.datetime.now():%a, %b %d %Y} - " + f"{time.strftime('%H:%M:%S')}: " + f"{output}")
            log_info = str(f"{dt.datetime.now():%a, %b %d %Y} - " + f"{time.strftime('%H:%M:%S')}: " + f"{output}")
            logfile = os.path.join(path, "logfile.txt")
            write_log(log_info, logfile)
            print(log_info)
            # messagebox.showinfo(title="System info", message="Force in by user!")
            settings.conveyable = True #camera2_check()
            settings.force_in_button = True
            # user_force_in()
            window.destroy()
            return log_info

        window = tk.Toplevel()
        window.wm_title("Baggage Handling System")
        window.geometry('800x600')
        window.resizable(False, False)
        window.columnconfigure(0, weight=1)
        window.rowconfigure(0, weight=1)

        self.main_frame = ttk.Frame(window, style='new.TFrame')
        self.main_frame.pack(fill=BOTH, expand=True)
        self.label2 = ttk.Label(self.main_frame, text="Please provide the reason why to force in:",
                  font=("helvetica", 20),style='L1.TLabel').grid(row=0, column=0, columnspan=2, padx=5, pady=10)
########################################################################################################################

        self.reasons_values = []
        with open(os.path.join(path, "reasons.txt")) as my_file:
            for line in my_file:
                self.reasons_values.append(line.rstrip("\n"))
        global output
        self.reason =[]

        for item in range(len(self.reasons_values)):
            self.reason.append(BooleanVar())
            self.reason_checklist = ttk.Checkbutton(self.main_frame, text=self.reasons_values[item],
                                                variable=self.reason[item],style='Chck1.TCheckbutton')

            self.reason_checklist.grid(row=2+item, column=0, sticky='w', padx=5, pady=5)




        self.button3_file = os.path.join(path, "submit.png")
        self.submit_image = PhotoImage(file=self.button3_file)
        self.submit_image = self.submit_image.subsample(2, 2)
        self.Submitbutton = ttk.Button(self.main_frame, image=self.submit_image,
                                       style='B1.TButton',command=get_reason)

        self.Submitbutton.grid(row=3+len(self.reasons_values), column=1, sticky= W + N , padx=5, pady=25)
########################################################################################################################
        self.footer_frame = ttk.Frame(window, style='new.TFrame')
        self.footer_frame.pack(fill=BOTH)

        self.time_and_date = ttk.Label(self.footer_frame,
                                       text=f"{dt.datetime.now():%a, %b %d %Y} - " + f"{time.strftime('%H:%M:%S')}",
                                       style='L1.TLabel',
                                       font=("helvetica", 14)).grid(row=0, column=0, sticky='sw', padx=20, pady=5)
        self.mylogo_file = os.path.join(ROOT_DIR, "logo.png")
        self.logo = PhotoImage(file=self.mylogo_file)
        self.logo = self.logo.subsample(1, 1)
        self.daifuku = ttk.Label(self.footer_frame, image=self.logo,
                                 style='L2.TLabel')
        self.daifuku.grid(row=0, column=1,rowspan=2,sticky='se', padx=100, pady=0)

########################################################################################################################

import timeit

def init_read_event(self):
        # read event
        self.event = []
        with open(os.path.join(path, "event.txt")) as my_file:
            for line in my_file:
                self.event.append(line)

        # self.event = settings.event_list

        self.ibca_message = ttk.Label(self.content_frame, style='L1.TLabel')
        self.ibca_message.config(font=("helvetica", 18))
        self.b = "-"

        def add_option(text):
            if self.ibca_message .cget("text") == "":
                self.ibca_message.config(text=self.b + " " + text)
            else:
                self.ibca_message.config(text=self.ibca_message.cget("text") + "\n" + self.b + " " + text)
        for item in range(len(self.event)):
            add_option(text=self.event[item])
            self.ibca_message.grid(row=1, column=0, padx=5, pady=5)



def read_event(self, root, event_list,image):

        self.ibca_message = ttk.Label(self.content_frame, style='L1.TLabel')
        self.ibca_message.config(font=("helvetica", 18))
        self.b = "-"
        self.ibca_message.grid(row=1, column=0, padx=5, pady=5)


        for item in range(len(event_list)):
            add_option(self, text=event_list[item])
            self.ibca_message.grid(row=1, column=0, padx=5, pady=5)

        # display image
        cv2image = cv2.resize(image,(360,480))

        img = Image.fromarray(cv2image)      # convert image for PIL
        img_detection = ImageTk.PhotoImage(img)  # convert image for tkinter
    
        self.img.image = img_detection           #  anchor imgtk so it does not be deleted by garbage-collector
        self.img.configure(image = img_detection)

        self.img.place(x=500, y=0)
        
        root.update()


def gui_config(self):
    self.geometry('800x600')
    self.resizable(False, False)
    self.title("Baggage Handling System")
    self.columnconfigure(0, weight=1)
    self.rowconfigure(0, weight=1)
    bhs = BaggageHandlingApp(self)
    return bhs


def add_option(self, text):
    if self.ibca_message .cget("text") == "":
        self.ibca_message.config(text=self.b + " " + text)
    else:
        self.ibca_message.config(text=self.ibca_message.cget("text") + "\n" + self.b + " " + text)






def test_GUI():
    gui_root = Tk()
    bhs = gui_config(gui_root)
    # gui_root.mainloop() # avoiding using mainloop to convert the app into function for integration.
    stop = timeit.default_timer()
    while 1:

        gui_root.update()

        start = timeit.default_timer()
        print("fresh time:", start - stop)
        stop = timeit.default_timer()


if __name__ == "__main__":
    test_GUI()


