import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import simpledialog
import tkinter.font as font

def collect_data():
    root = tk.Tk()
    root.withdraw()  
    
    name = simpledialog.askstring("Input", "Enter name of person:")
    if not name:
        print("No name entered")
        root.destroy()
        return
    
    ids = simpledialog.askstring("Input", "Enter ID:")
    if not ids:
        print("No ID entered")
        root.destroy()
        return
    
    count = 1
    cap = cv2.VideoCapture(0)
    filename = "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(filename)

    while True:
        ret, frm = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.4, 1)

        for x, y, w, h in faces:
            cv2.rectangle(frm, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi = gray[y:y + h, x:x + w]
            if not os.path.exists('persons'):
                os.makedirs('persons')
            cv2.imwrite(f"persons/{name}-{count}-{ids}.jpg", roi)
            count += 1
            cv2.putText(frm, f"{count}", (20, 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
            cv2.imshow("new", roi)

        cv2.imshow("identify", frm)
        if cv2.waitKey(1) == 27 or count > 300:
            break

    cv2.destroyAllWindows()
    cap.release()
    train()
    root.destroy()

def train():
    print("Training part initiated!")
    recog = cv2.face.LBPHFaceRecognizer_create()
    dataset = 'persons'
    paths = [os.path.join(dataset, im) for im in os.listdir(dataset)]
    faces = []
    ids = []
    labels = []
    for path in paths:
        labels.append(path.split('/')[-1].split('-')[0])
        ids.append(int(path.split('/')[-1].split('-')[2].split('.')[0]))
        faces.append(cv2.imread(path, 0))
    recog.train(faces, np.array(ids))
    recog.save('model.yml')

def identify():
    cap = cv2.VideoCapture(0)
    filename = "haarcascade_frontalface_default.xml"
    paths = [os.path.join("persons", im) for im in os.listdir("persons")]
    labelslist = {path.split('/')[-1].split('-')[2].split('.')[0]: path.split('/')[-1].split('-')[0] for path in paths}
    recog = cv2.face.LBPHFaceRecognizer_create()
    recog.read('model.yml')
    cascade = cv2.CascadeClassifier(filename)

    while True:
        ret, frm = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.3, 2)

        for x, y, w, h in faces:
            cv2.rectangle(frm, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi = gray[y:y + h, x:x + w]
            label = recog.predict(roi)
            if label[1] < 100:
                cv2.putText(frm, f"{labelslist[str(label[0])]} + {int(label[1])}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:
                cv2.putText(frm, "unknown", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
        cv2.imshow("identify", frm)
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
    cap.release()

def maincall():
    root = tk.Tk()
    root.geometry("480x100")
    root.title("Identify")
    label = tk.Label(root, text="Select below buttons")
    label.grid(row=0, columnspan=2)
    label_font = font.Font(size=35, weight='bold', family='Helvetica')
    label['font'] = label_font

    btn_font = font.Font(size=25)
    button1 = tk.Button(root, text="Add Member", command=collect_data, height=2, width=20)
    button1.grid(row=1, column=0, pady=(10, 10), padx=(5, 5))
    button1['font'] = btn_font

    button2 = tk.Button(root, text="Start with known", command=identify, height=2, width=20)
    button2.grid(row=1, column=1, pady=(10, 10), padx=(5, 5))
    button2['font'] = btn_font

    root.mainloop()

if __name__ == "__main__":
    maincall()
