"""
This program is a survey (questionnaire) containing questions from:
'Analysis and prediction of professional sports consumer
behavior using artificial neural network'
Kyung Hee University
2011.08
Programmer: Joshua Willman
Date: 2019.11.17
"""
from tkinter import (Tk, Label, Button, Radiobutton, Frame, Menu,
                     messagebox, StringVar, Listbox, BROWSE, END, Toplevel, Entry)
from tkinter import ttk
from tkinter import messagebox
import pathlib
import time
import csv
import os.path

# create empty lists used for each set of questions
from main import predict_organization

lifestyle_list = []
sig_trends_list = []
future_trends_list = []
general_answers_list = []

def dialogBox(title, message):
    """
    Basic function to create and display general dialog boxes.
    """
    dialog = Tk()
    dialog.wm_title(title)
    dialog.grab_set()
    dialogWidth, dialogHeight = 225, 125
    positionRight = int(dialog.winfo_screenwidth()/2 - dialogWidth/2)
    positionDown = int(dialog.winfo_screenheight()/2 - dialogHeight/2)
    dialog.geometry("{}x{}+{}+{}".format(
        dialogWidth, dialogHeight, positionRight, positionDown))
    dialog.maxsize(dialogWidth, dialogHeight)
    label = Label(dialog, text=message)
    label.pack(side="top", fill="x", pady=10)
    ok_button = ttk.Button(dialog, text="Ok", command=dialog.destroy)
    ok_button.pack(ipady=3, pady=10)
    dialog.mainloop()

def nextSurveyDialog(title, message, cmd):
    """
    Dialog box that appears before moving onto the next set of questions.
    """
    dialog = Tk()
    dialog.wm_title(title)
    dialog.grab_set()
    dialogWidth, dialogHeight = 225, 125
    positionRight = int(dialog.winfo_screenwidth()/2 - dialogWidth/2)
    positionDown = int(dialog.winfo_screenheight()/2 - dialogHeight/2)
    dialog.geometry("{}x{}+{}+{}".format(
        dialogWidth, dialogHeight, positionRight, positionDown))
    dialog.maxsize(dialogWidth, dialogHeight)
    dialog.overrideredirect(True)
    label = Label(dialog, text=message)
    label.pack(side="top", fill="x", pady=10)
    ok_button = ttk.Button(dialog, text="Begin", command=lambda: [f() for f in [cmd, dialog.destroy]])
    ok_button.pack(ipady=3, pady=10)

    dialog.protocol("WM_DELETE_WINDOW", disable_event) # prevent user from clicking ALT + F4 to close
    dialog.mainloop()

def disable_event():
    pass

def finishedDialog(title, message):
    """
    Display the finished dialog box when user reaches the end of the survey.
    """
    dialog = Tk()
    dialog.wm_title(title)
    dialog.grab_set()
    dialogWidth, dialogHeight = 325, 150
    positionRight = int(dialog.winfo_screenwidth()/2 - dialogWidth/2)
    positionDown = int(dialog.winfo_screenheight()/2 - dialogHeight/2)
    dialog.geometry("{}x{}+{}+{}".format(
        dialogWidth, dialogHeight, positionRight, positionDown))
    dialog.maxsize(dialogWidth, dialogHeight)
    dialog.overrideredirect(True)
    label = Label(dialog, text=message)
    label.pack(side="top", fill="x", pady=10)
    ok_button = ttk.Button(dialog, text="Quit", command=quit)
    ok_button.pack(ipady=3, pady=10)

    dialog.protocol("WM_DELETE_WINDOW", disable_event) # prevent user from clicking ALT + F4 to close
    dialog.mainloop()

def writeToFile(filename, answer_list):
    """
    Called at end of program when user selects finished button,
    write all lists to separate files.
    Parameters: filename: name for save file,
                answer_list: list containing answer from that one of the
                four sections in the survey.
    """
    headers = []
    file_exists = os.path.isfile(filename)

    with open(filename, 'a') as csvfile:
        for i in range(1, len(answer_list) + 1):
            headers.append("Q{}".format(i))
        writer = csv.writer(csvfile, delimiter=',', lineterminator='\n')

        if not file_exists:
            writer.writerow(headers) # file doesn't exist yet, write a header

        writer.writerow(answer_list)

class otherPopUpDialog(object):
    """
    Class for 'other' selections in General Question class.
    When user selects 'other' option, they are able to input
    their answer into an Entry widget.
    self.value: the value of Entry widget.
    """
    def __init__(self, master, text):
        top=self.top=Toplevel(master)
        self.text = text
        top.wm_title("Other Answers")
        top.grab_set()
        dialogWidth, dialogHeight = 200, 150
        positionRight = int(top.winfo_screenwidth()/2 - dialogWidth/2)
        positionDown = int(top.winfo_screenheight()/2 - dialogHeight/2)
        top.geometry("{}x{}+{}+{}".format(
            dialogWidth, dialogHeight, positionRight, positionDown))
        self.label = Label(top, text=self.text)
        self.label.pack(ipady=5)
        self.enter = Entry(top)
        self.enter.pack(ipady=5)
        self.ok_button = Button(top, text="Ok", command=self.cleanup)
        self.ok_button.pack(ipady=5)

    def cleanup(self):
        """
        Get input from Entry widget and close dialog.
        """
        self.value = self.enter.get()
        self.top.destroy()

class Survey(Tk):
    """
    Main class, define the container which will contain all the frames.
    """
    def __init__(self, *args, **kwargs):
        Tk.__init__(self, *args, **kwargs)

        # call closing protocol to create dialog box to ask
        # if user if they want to quit or not.
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        Tk.wm_title(self, "Non-profit Organization Matcher")

        # get position of window with respect to screen
        windowWidth, windowHeight = 750, 400
        positionRight = int(Tk.winfo_screenwidth(self)/2 - windowWidth/2)
        positionDown = int(Tk.winfo_screenheight(self)/2 - windowHeight/2)
        Tk.geometry(self, newGeometry="{}x{}+{}+{}".format(
            windowWidth, windowHeight, positionRight, positionDown))
        Tk.maxsize(self, windowWidth, windowHeight)

        # Create container Frame to hold all other classes,
        # which are the different parts of the survey.
        container = Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # Create menu bar
        menubar = Menu(container)
        filemenu = Menu(menubar, tearoff=0)
        filemenu.add_command(label="Quit", command=quit)
        menubar.add_cascade(label="File", menu=filemenu)

        Tk.config(self, menu=menubar)

        # create empty dictionary for the different frames (the different classes)
        self.frames = {}

        for fr in (StartPage, GenderQuestion, AgeQuestion, MarriageQuestion, LocationQuestion,
                   WorkQuestion, LaborQuestion, ShukHofshiQuestion):
            frame = fr(container, self)
            self.frames[fr] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def on_closing(self):
        """
        Display dialog box before quitting.
        """
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.destroy()

    def show_frame(self, cont):
        """
        Used to display a frame.
        """
        frame = self.frames[cont]
        frame.tkraise() # bring a frame to the "top"

class StartPage(Frame):
    """
    First page that user will see.
    Explains the rules and any extra information the user may need
    before beginning the survey.
    User can either click one of the two buttons, Begin Survey or Quit.
    """
    def __init__(self, master, controller):
        Frame.__init__(self, master)
        self.controller = controller

        # set up start page window
        self.configure(bg="#3A9ED0")
        start_label = Label(self, text="Welcome to NOM - Non-profit Organization Matcher!", font=("Halvetica", 18),
                            borderwidth=2, relief="ridge")
        start_label.pack(pady=10, padx=10, ipadx=5, ipady=3)


        # add labels and buttons to window
        info_text = "This survey asks you some questions in order to get to know you better\nfor recommending the most suitable organization for you..."
        info_label = Label(self, text=info_text, font=("Halvetica", 14),
                           borderwidth=2, relief="ridge")
        info_label.pack(pady=10, padx=10, ipadx=20, ipady=3)

        purpose_text = "We are happy you want to take part.\nAny donation and other help will make our world a better place!"
        purpose_text = Label(self, text=purpose_text, font=("Halvetica", 12),
                             borderwidth=2, relief="ridge")
        purpose_text.pack(pady=10, padx=10, ipadx=5, ipady=3)

        start_button = ttk.Button(self, text="Begin Survey",
                                  command=lambda: controller.show_frame(AgeQuestion))
        start_button.pack(ipadx=10, ipady=15, pady=15)

        quit_button = ttk.Button(self, text="Quit", command=self.on_closing)
        quit_button.pack(ipady=3, pady=10)

    def on_closing(self):
        """
        Display dialog box before quitting.
        """
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.controller.destroy()


class GenderQuestion(Frame):
    """
    Displays gender question from General questions.
    """
    def __init__(self, master, controller):
        Frame.__init__(self, master)
        self.controller = controller
        self.configure(bg="#3A9ED0")

        global general_answers_list

        # Create header label
        ttk.Label(self, text="Question Number 2", font=('Halvetica', 20),
                  borderwidth=2, relief="ridge").pack(padx=10, pady=10)

        self.question = "What is your gender?"

        # Set up labels and checkboxes
        self.question_label = Label(self, text="{}".format(self.question), font=('Halvetica', 16))
        self.question_label.pack(anchor='w', padx=20, pady=10)

        choices = [("Male", "M"), ("Female", "F"), ("Other", "Other")]

        self.var = StringVar()
        self.var.set(0) # initialize

        # Frame to contain checkboxes
        checkbox_frame = Frame(self, borderwidth=2, relief="ridge")
        checkbox_frame.pack(pady=10, anchor='center')

        for text, value in choices:
            b = ttk.Radiobutton(checkbox_frame, text=text,
                                variable=self.var, value=value)
            b.pack(side='left', ipadx=17, ipady=2)

        enter_button = ttk.Button(self, text="Next Question", command=self.nextQuestion)
        enter_button.pack(ipady=5, pady=20)

    def nextQuestion(self):
        '''
        When button is clicked, add user's input to a list
        and display next question.
        '''
        answer = self.var.get()

        if answer == '0':
            dialogBox("No Value Given",
                      "You did not select an answer.\nPlease try again.")
        else:
            selected_answer = self.var.get()
            general_answers_list.append(selected_answer)

            time.sleep(.2) # delay between questions

            self.controller.show_frame(WorkQuestion)

class MarriageQuestion(Frame):
    """
    Displays marriage question from General questions.
    """
    def __init__(self, master, controller):
        Frame.__init__(self, master)
        self.controller = controller
        self.configure(bg="#3A9ED0")

        global general_answers_list

        # Create header label
        ttk.Label(self, text="Question Number 6", font=('Halvetica', 20),
                  borderwidth=2, relief="ridge").pack(padx=10, pady=10)

        self.question = "What is your status?"

        # Set up labels and checkboxes
        self.question_label = Label(self, text="{}".format(self.question), font=('Halvetica', 16))
        self.question_label.pack(anchor='w', padx=20, pady=10)

        choices = [("Married", "Married"), ("Single", "Single")]

        self.var = StringVar()
        self.var.set(0) # initialize

        # Frame to contain checkboxes
        checkbox_frame = Frame(self, borderwidth=2, relief="ridge")
        checkbox_frame.pack(pady=10, anchor='center')

        for text, value in choices:
            b = ttk.Radiobutton(checkbox_frame, text=text,
                                variable=self.var, value=value)
            b.pack(side='left', ipadx=17, ipady=2)

        enter_button = ttk.Button(self, text="Next Question", command=self.nextQuestion)
        enter_button.pack(ipady=5, pady=20)

    def nextQuestion(self):
        '''
        When button is clicked, add user's input to a list
        and display next question.
        '''
        answer = self.var.get()

        if answer == '0':
            dialogBox("No Value Given",
                      "You did not select an answer.\nPlease try again.")
        else:
            selected_answer = self.var.get()
            general_answers_list.append(selected_answer)

            time.sleep(.2) # delay between questions

            self.controller.show_frame(ShukHofshiQuestion)

class AgeQuestion(Frame):
    """
    Displays age question from General questions.
    """
    def __init__(self, master, controller):
        Frame.__init__(self, master)
        self.controller = controller
        self.configure(bg="#3A9ED0")

        global general_answers_list

        # Create header label
        ttk.Label(self, text="Question Number 1", font=('Halvetica', 20),
                  borderwidth=2, relief="ridge").pack(padx=10, pady=10)

        self.question = "What is your age?"

        # Set up labels and checkboxes
        self.question_label = Label(self, text="{}".format(self.question), font=('Halvetica', 16))
        self.question_label.pack(anchor='w', padx=20, pady=10)

        choices = [("Under 20", "20"), ("20-30", "20-30"), ("30-50", "30-50"),
                   ("50-65", "50-65"), ("Above 65", "65")]

        self.var = StringVar()
        self.var.set(0) # initialize

        # Frame to contain checkboxes
        checkbox_frame = Frame(self, borderwidth=2, relief="ridge")
        checkbox_frame.pack(pady=10, anchor='center')

        for text, value in choices:
            b = ttk.Radiobutton(checkbox_frame, text=text,
                                variable=self.var, value=value)
            b.pack(side='left', ipadx=12, ipady=2)

        enter_button = ttk.Button(self, text="Next Question", command=self.nextQuestion)
        enter_button.pack(ipady=5, pady=20)

    def nextQuestion(self):
        '''
        When button is clicked, add user's input to a list
        and display next question.
        '''
        answer = self.var.get()

        if answer == '0':
            dialogBox("No Value Given",
                      "You did not select an answer.\nPlease try again.")
        else:
            selected_answer = self.var.get()
            general_answers_list.append(selected_answer)

            time.sleep(.2) # delay between questions

            self.controller.show_frame(GenderQuestion)

class LocationQuestion(Frame):
    """
    Displays age question from General questions.
    """
    def __init__(self, master, controller):
        Frame.__init__(self, master)
        self.controller = controller
        self.configure(bg="#3A9ED0")

        global general_answers_list

        # Create header label
        ttk.Label(self, text="Question Number 5", font=('Halvetica', 20),
                  borderwidth=2, relief="ridge").pack(padx=10, pady=10)

        self.question = "Where do you live?"

        # Set up labels and checkboxes
        self.question_label = Label(self, text="{}".format(self.question), font=('Halvetica', 16))
        self.question_label.pack(anchor='w', padx=20, pady=10)

        choices = [("Jerusalem", "Jerusalem"), ("Gush Dan", "Dan"), ("North", "North"),
                   ("South", "South"), ("Judea and Samaria", "Judea and samaria"),
                   ("Other", "Other")]

        self.var = StringVar()
        self.var.set(0) # initialize

        # Frame to contain checkboxes
        checkbox_frame = Frame(self, borderwidth=2, relief="ridge")
        checkbox_frame.pack(pady=10, anchor='center')

        for text, value in choices:
            b = ttk.Radiobutton(checkbox_frame, text=text,
                                variable=self.var, value=value)
            b.pack(side='left', ipadx=12, ipady=2)

        enter_button = ttk.Button(self, text="Next Question", command=self.nextQuestion)
        enter_button.pack(ipady=5, pady=20)

    def nextQuestion(self):
        '''
        When button is clicked, add user's input to a list
        and display next question.
        '''
        answer = self.var.get()

        if answer == '0':
            dialogBox("No Value Given",
                      "You did not select an answer.\nPlease try again.")
        else:
            selected_answer = self.var.get()
            general_answers_list.append(selected_answer)

            time.sleep(.2) # delay between questions

            self.controller.show_frame(MarriageQuestion)

class WorkQuestion(Frame):
    """
    Displays work question from General questions.
    """
    def __init__(self, master, controller):
        Frame.__init__(self, master)
        self.controller = controller
        self.configure(bg="#3A9ED0")

        global general_answers_list

        # Create header label
        ttk.Label(self, text="Question Number 3", font=('Halvetica', 20),
                  borderwidth=2, relief="ridge").pack(padx=10, pady=10)

        self.question = "What is your field of employment?"

        # Set up labels and listbox
        self.question_label = Label(self, text="{}".format(self.question), font=('Halvetica', 16))
        self.question_label.pack(anchor='w', padx=20, pady=10)

        choices = ["nature", "social", "humanities", "law", "education", "services",
                   "security", "artist", "medical"]

        self.lb_choices = Listbox(self, selectmode=BROWSE, width=20, borderwidth=3, relief="ridge")
        self.lb_choices.pack(ipady=5, ipadx=5)

        for ch in choices:
            self.lb_choices.insert(END, ch)

        self.enter_button = ttk.Button(self, text="Next Question", command=self.nextQuestion)
        self.enter_button.pack(ipady=5, pady=20)

    def nextQuestion(self):
        '''
        When button is clicked, add user's input to a list
        and display next question.
        '''
        selection = self.lb_choices.curselection()

        if len(selection) == 0:
            dialogBox("No Value Given",
                      "You did not select an answer.\nPlease try again.")
        elif  selection == (8,):
            self.other_window = otherPopUpDialog(self.master, text="Other Occupation:")
            self.enter_button["state"] = "disabled"
            self.master.wait_window(self.other_window.top)
            self.enter_button["state"] = "normal"

            get_other = self.other_window.value
            general_answers_list.append(get_other)

            time.sleep(.2) # delay between questions

            self.controller.show_frame(LaborQuestion)
        else:
            get_selection = self.lb_choices.get(selection)
            general_answers_list.append(get_selection)

            time.sleep(.2) # delay between questions

            self.controller.show_frame(LaborQuestion)


class LaborQuestion(Frame):
    """
    Displays salary question from General questions.
    """
    def __init__(self, master, controller):
        Frame.__init__(self, master)
        self.controller = controller
        self.configure(bg="#3A9ED0")

        global general_answers_list

        # Create header label
        ttk.Label(self, text="Qustion Number 4", font=('Halvetica', 20),
                  borderwidth=2, relief="ridge").pack(padx=10, pady=10)

        self.question = "What is the party you voted for in the last election?"

        # Set up labels and listbox
        self.question_label = Label(self, text="{}".format(self.question), font=('Halvetica', 16))
        self.question_label.pack(anchor='w', padx=20, pady=10)

        choices = ['Yemina', 'Avoda', 'Kachol-lavan', 'Other', 'Tzionut-datit', 'Yesh-atid', 'Tikva-hadasha',
                   'Likud', 'Meretz', 'Israel-beytenu']

        self.lb_choices = Listbox(self, selectmode=BROWSE, width=20, borderwidth=3, relief="ridge")
        self.lb_choices.pack(ipady=5, ipadx=5)

        for ch in choices:
            self.lb_choices.insert(END, ch)

        enter_button = ttk.Button(self, text="Next Question", command=self.nextQuestion)
        enter_button.pack(ipady=5, pady=20)

    def nextQuestion(self):
        '''
        When button is clicked, add user's input to a list
        and display next question.
        '''
        selection = self.lb_choices.curselection()

        if len(selection) == 0:
            dialogBox("No Value Given",
                      "You did not select an answer.\nPlease try again.")
        else:
            get_selection = self.lb_choices.get(selection)
            general_answers_list.append(get_selection)

            time.sleep(.2) # delay between questions

            self.controller.show_frame(LocationQuestion)


class ShukHofshiQuestion(Frame):
    """
    Displays transportation question from General questions.
    """
    def __init__(self, master, controller):
        Frame.__init__(self, master)
        self.controller = controller
        self.configure(bg="#3A9ED0")

        global general_answers_list

        # Create header label
        ttk.Label(self, text="Question Number 7", font=('Halvetica', 20),
                  borderwidth=2, relief="ridge").pack(padx=10, pady=10)

        self.question = "I support a free market, lowering taxes \nand reducing the services I get from the state"

        self.question_label = Label(self, text="{}".format(self.question), font=('Halvetica', 16))
        self.question_label.pack(anchor='w', padx=20, pady=10)

        # Set up labels and checkboxes
        scale = [("1", "1"), ("2", "2"), ("3", "3"),
                 ("4", "4"), ("5", "5")]

        self.var = StringVar()
        self.var.set(0) # initialize

        # Frame to contain checkboxes
        checkbox_frame = Frame(self, borderwidth=2, relief="ridge")
        checkbox_frame.pack(pady=10, anchor='center')

        for text, value in scale:
            b = ttk.Radiobutton(checkbox_frame, text=text,
                                variable=self.var, value=value)
            b.pack(side='left', ipadx=7, ipady=2)

        enter_button = ttk.Button(self, text="Next Question", command=self.nextQuestion)
        enter_button.pack(ipady=5, pady=20)

    def nextQuestion(self):
        '''
        When button is clicked, add user's input to a list
        and display next question.
        '''
        answer = self.var.get()

        if answer == '0':
            dialogBox("No Value Given",
                      "You did not select an answer.\nPlease try again.")
        else:
            selected_answer = self.var.get()
            general_answers_list.append(selected_answer)

            time.sleep(.2) # delay between questions

            predicted_org = predict_organization(general_answers_list)
            finished_text = f'Thank you for using our matcher!\n Our recommendation for you is:\n{predicted_org}'
            finishedDialog("Your match:", finished_text)

