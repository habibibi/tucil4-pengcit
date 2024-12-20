import threading
from tkinter.ttk import Progressbar
import cv2
import CNN
import numpy as np
from ultralytics import YOLO
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk

def putImage(canvas : Canvas, image_array):
    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image_array)
    w, h = image.size
    scale_factor = min(canvas_width / w, canvas_height / h)
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    resized_image = image.resize((new_w, new_h))
    tk_image = ImageTk.PhotoImage(resized_image)
    canvas.itemconfig(1, image=tk_image)
    canvas.image=tk_image

def browseFiles():
    file_path = filedialog.askopenfilename(initialdir="./",
                                           title="Select a File",
                                           filetypes=[("Image Files", "*.png;*.jpg")])
    if file_path:
        try:
            def process_image():
                progress.pack()
                progress.start()
                image = cv2.imread(file_path)
                
                file_location_text.set(file_path)
                putImage(image_input, image)

                conv_class_name = "placeholder" # TODO: conventional output
                putImage(out_conv_image, image)
                out_conv_classname.set(conv_class_name)

                # cnn output
                [cnn_class_name, cnn_img] = CNN.predict(yolo, image)
                putImage(out_cnn_image, cnn_img)
                out_cnn_classname.set(cnn_class_name)
                progress.stop()
                progress.pack_forget()
            threading.Thread(target=process_image).start()
        except Exception as e:
            print(f"Error opening image: {e}")
    else:
        print("No file selected.")

if __name__ == "__main__":
    yolo = YOLO('yolov8s.pt')

    image_width= 307
    image_height = 230
    title_font = ("Arial", 12, "bold")

    window = Tk() 
    window.title('Tugas 4 Pemrosesan Citra Digital')
    window.geometry("1000x650")
    window.config(background = "white smoke")
        
    # Upper frame
    upper_frame = Frame(window)
    upper_frame.pack(side=TOP, fill=X, ipadx=10, ipady=5)

    pilih_gambar_text = Label(upper_frame, text="pilih gambar")
    pilih_gambar_text.pack(side=LEFT, padx=10)


    file_location_text = StringVar()
    file_location_box = Entry(upper_frame, state="readonly", 
                            textvariable=file_location_text, 
                            readonlybackground="white",
                            width=50)
    file_location_box.pack(side=LEFT, padx=5)

    button_explore = Button(upper_frame, 
                            text = "Browse",
                            command = browseFiles) 
    button_explore.pack(side=LEFT, padx=10)

    # Left frame
    left_frame = Frame(window)
    left_frame.pack(side=LEFT, fill=BOTH, expand=True, ipady=5)

    # Right frame
    right_frame = Frame(window)
    right_frame.pack(side=RIGHT, fill=BOTH, expand=True, ipady=5)

    # Left frame content
    input_label = Label(left_frame, text="input", font=title_font)
    input_label.pack(pady=5)
    image_input = Canvas(left_frame, bg="white", height=image_height, width=image_width)
    image_input.create_image(image_width/2, image_height/2, anchor="center", image=None)
    image_input.pack(pady=5)

    # Right frame content
    out_conv_title = Label(right_frame, text="Hasil prediksi metode konvensional", font=title_font)
    out_conv_title.pack(pady=5)
    out_conv_image = Canvas(right_frame, bg="white", height=image_height, width=image_width)
    out_conv_image.create_image(image_width/2, image_height/2, anchor="center", image=None)
    out_conv_image.pack(pady=5)
    out_conv_classname = StringVar()
    out_conv_label = Label(right_frame, textvariable=out_conv_classname)
    out_conv_label.pack(pady=5)

    out_cnn_title = Label(right_frame, text="Hasil prediksi metode CNN", font=title_font)
    out_cnn_title.pack(pady=5)
    out_cnn_image = Canvas(right_frame, bg="white", height=image_height, width=image_width)
    out_cnn_image.create_image(image_width/2, image_height/2, anchor="center", image=None)
    out_cnn_image.pack(pady=5)
    out_cnn_classname = StringVar()
    out_cnn_label = Label(right_frame, textvariable=out_cnn_classname)
    out_cnn_label.pack(pady=5)
    progress = Progressbar(left_frame, mode="indeterminate", length=200)
    progress.pack(pady=10)
    progress.pack_forget()

    window.mainloop()
