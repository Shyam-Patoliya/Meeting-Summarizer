import speech_recognition as sr
import requests
import os
import tkinter as tk
from tkinter import scrolledtext, messagebox
from tkinter import ttk
import sys
import subprocess

# Global variables
GROQ_API_KEY = "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"  # Replace with your actual API key
recognizer = sr.Recognizer()
mic = sr.Microphone()
stop_listening = None
captured_text = []
online_mode = False

# Initialize microphone once at startup
with mic as source:
    recognizer.adjust_for_ambient_noise(source)

def recognize_speech_from_mic():
    global stop_listening
    try:
        stop_listening = recognizer.listen_in_background(mic, process_audio)
    except Exception as e:
        messagebox.showerror("Error", f"Microphone error: {str(e)}")

def process_audio(recognizer, audio):
    global captured_text, online_mode
    try:
        if online_mode:
            text = recognize_online(audio)
        else:
            text = recognize_offline(audio)
        
        if text:
            captured_text.append(text)
            recognized_text_area.insert(tk.END, f"Recognized: {text}\n")
            recognized_text_area.see(tk.END)
    except Exception as e:
        messagebox.showerror("Error", str(e))

def recognize_online(audio):
    try:
        api_key = os.getenv("GROQ_API_KEY") or GROQ_API_KEY
        if not api_key:
            return "Error: Groq API key not found."

        audio_data = audio.get_wav_data()
        response = requests.post(
            "https://api.groq.com/v1/speech-to-text",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"audio": ("audio.wav", audio_data, "audio/wav")}
        )

        return response.json().get("text", "[No text recognized]") if response.status_code == 200 else f"Error: {response.text}"
    except Exception as e:
        return f"Online recognition error: {str(e)}"

def recognize_offline(audio):
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError:
        return "API unavailable"

def summarize_using_groq(text):
    try:
        api_key = os.getenv("GROQ_API_KEY") or GROQ_API_KEY
        if not api_key:
            return "Error: Groq API key not found."

        api_url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama3-70b-8192",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that summarizes text."},
                {"role": "user", "content": f"Summarize the following text into key points: {text}"}
            ],
            "temperature": 0.5
        }

        response = requests.post(api_url, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"Error in summarization: {response.text}"
    except Exception as e:
        return f"Error in summarization: {str(e)}"

def generate_mom(summary):
    return f"*Minutes of Meeting\n\nKey Discussion Points:*\n{summary}"

def generate_summary():
    full_text = " ".join(captured_text)
    if not full_text:
        messagebox.showwarning("Warning", "No text captured for summarization.")
        return

    summary = summarize_using_groq(full_text)
    mom = generate_mom("\n- " + summary.replace("\n", "\n- "))
    generated_summary_area.delete(1.0, tk.END)
    generated_summary_area.insert(tk.END, mom)
    generated_summary_area.see(tk.END)

def start_recognition():
    global online_mode, stop_listening
    online_mode = mode_var.get() == 'Online'
    
    if stop_listening:
        stop_listening(wait_for_stop=False)
    recognize_speech_from_mic()

def stop_recognition():
    global stop_listening
    if stop_listening:
        stop_listening(wait_for_stop=False)
        stop_listening = None

def reset_application():
    global captured_text, stop_listening
    # Stop any ongoing recognition
    stop_recognition()
    # Clear all captured text
    captured_text = []
    # Reset text areas
    recognized_text_area.delete(1.0, tk.END)
    generated_summary_area.delete(1.0, tk.END)
    # Re-insert headers
    recognized_text_area.insert(tk.END, "Recognized Text:\n", 'header')
    generated_summary_area.insert(tk.END, "Minutes of Meeting:\n", 'header')
    messagebox.showinfo("Reset", "Application has been reset successfully!")

# GUI Setup
root = tk.Tk()
root.title("Speech Recognition and Summarization")
root.configure(bg="#2E3440")
root.geometry("1200x900")
root.minsize(1100, 800)
root.resizable(True, True)

# Custom Style Configuration
style = ttk.Style()
style.theme_use('clam')

# Color scheme
bg_color = "#2E3440"
text_bg = "#3B4252"
text_fg = "#ECEFF4"
button_bg = "#434C5E"
active_bg = "#5E81AC"

# Configure styles
style.configure("TFrame", background=bg_color)
style.configure("TRadiobutton", 
                background=bg_color,
                foreground=text_fg,
                font=("Helvetica", 12))
style.map("TRadiobutton",
          background=[('active', bg_color), ('pressed', bg_color)],
          foreground=[('active', text_fg), ('pressed', text_fg)])

style.configure("TButton", 
                font=("Helvetica", 12, "bold"),
                padding=15,
                borderwidth=0,
                background=button_bg,
                foreground=text_fg)
style.map("TButton",
          background=[('active', active_bg), ('pressed', active_bg)],
          foreground=[('active', text_fg)])

# Custom button styles
style.configure("Start.TButton", background="#4CAF50")
style.configure("Stop.TButton", background="#F44336")
style.configure("Summary.TButton", background="#2196F3")
style.configure("Reset.TButton", background="#9C27B0")  # Purple color for reset

# Main container
main_frame = ttk.Frame(root)
main_frame.pack(pady=25, padx=25, fill='both', expand=True)

# Mode selection
mode_frame = ttk.Frame(main_frame)
mode_frame.pack(pady=15, fill='x')

mode_var = tk.StringVar(value='Online')
online_radio = ttk.Radiobutton(mode_frame, 
                              text='Online Mode', 
                              variable=mode_var, 
                              value='Online')
offline_radio = ttk.Radiobutton(mode_frame, 
                               text='Offline Mode', 
                               variable=mode_var, 
                               value='Offline')
online_radio.pack(side=tk.LEFT, padx=25, ipadx=10)
offline_radio.pack(side=tk.LEFT, padx=25, ipadx=10)

# Control buttons
button_frame = ttk.Frame(main_frame)
button_frame.pack(pady=20, fill='x')

start_button = ttk.Button(button_frame, 
                         text="Start Recognition", 
                         command=start_recognition,
                         style="Start.TButton")
start_button.grid(row=0, column=0, padx=15, ipadx=20)

stop_button = ttk.Button(button_frame, 
                        text="Stop Recognition", 
                        command=stop_recognition,
                        style="Stop.TButton")
stop_button.grid(row=0, column=1, padx=15, ipadx=20)

summary_button = ttk.Button(button_frame, 
                           text="Generate Summary", 
                           command=generate_summary,
                           style="Summary.TButton")
summary_button.grid(row=0, column=2, padx=15, ipadx=20)

reset_button = ttk.Button(button_frame, 
                         text="Reset", 
                         command=reset_application,
                         style="Reset.TButton")
reset_button.grid(row=0, column=3, padx=15, ipadx=20)

button_frame.grid_columnconfigure(0, weight=1)
button_frame.grid_columnconfigure(1, weight=1)
button_frame.grid_columnconfigure(2, weight=1)
button_frame.grid_columnconfigure(3, weight=1)

# Text areas configuration
recognized_style = {
    'wrap': tk.WORD,
    'width': 100,
    'height': 10,
    'font': ("Consolas", 12),
    'relief': 'flat',
    'borderwidth': 4,
    'bg': text_bg,
    'fg': text_fg,
    'insertbackground': text_fg
}

summary_style = {
    'wrap': tk.WORD,
    'width': 100,
    'height': 15,
    'font': ("Consolas", 12),
    'relief': 'flat',
    'borderwidth': 4,
    'bg': text_bg,
    'fg': text_fg,
    'insertbackground': text_fg
}

# Recognized text area
recognized_text_area = scrolledtext.ScrolledText(main_frame, **recognized_style)
recognized_text_area.pack(pady=15, fill='both', expand=True)
recognized_text_area.insert(tk.END, "Recognized Text:\n", 'header')
recognized_text_area.tag_configure('header', 
                                 foreground=active_bg, 
                                 font=("Helvetica", 14, "bold"))

# Summary text area
generated_summary_area = scrolledtext.ScrolledText(main_frame, **summary_style)
generated_summary_area.pack(pady=15, fill='both', expand=True)
generated_summary_area.insert(tk.END, "Minutes of Meeting:\n", 'header')
generated_summary_area.tag_configure('header', 
                                  foreground=active_bg, 
                                  font=("Helvetica", 14, "bold"))

# Add subtle borders to text areas
for widget in [recognized_text_area, generated_summary_area]:
    widget.configure(highlightbackground=active_bg,
                    highlightcolor=active_bg,
                    highlightthickness=2)

root.mainloop()
