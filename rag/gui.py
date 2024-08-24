import os
import tkinter as tk
import warnings
from tkinter import scrolledtext, messagebox
from commands import Commands

# Suppress specific deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class RAGApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RAG System")
        self.root.geometry("800x600")

        self.commands = Commands()

        # Initialize button
        self.init_button = tk.Button(root, text="Initialize System", command=self.initialize_system)
        self.init_button.pack(pady=10)

        # Query Entry
        self.query_label = tk.Label(root, text="Enter your query:")
        self.query_label.pack(pady=5)
        self.query_entry = tk.Entry(root, width=50)
        self.query_entry.pack(pady=5)

        # Query Button
        self.query_button = tk.Button(root, text="Send Query", command=self.send_query)
        self.query_button.pack(pady=10)

        # Output Box
        self.output_box = scrolledtext.ScrolledText(root, width=100, height=20, wrap=tk.WORD)
        self.output_box.pack(pady=10)

        # Restore Button
        self.restore_button = tk.Button(root, text="Restore Vector Store", command=self.restore_system)
        self.restore_button.pack(pady=10)

        # Exit Button
        self.exit_button = tk.Button(root, text="Exit", command=self.root.quit)
        self.exit_button.pack(pady=10)

    def initialize_system(self):
        pdf_dir = "pdfs"
        self.commands.init(pdf_dir)
        messagebox.showinfo("Initialization", "System initialized with PDFs from the 'pdfs' directory.")

    def send_query(self):
        query_text = self.query_entry.get()
        if not query_text:
            messagebox.showwarning("Input Error", "Please enter a query.")
            return

        response = self.commands.query(query_text)
        self.output_box.insert(tk.END, f"Query: {query_text}\nResponse: {response}\n\n")

    def restore_system(self):
        self.commands.restore()
        messagebox.showinfo("Restore", "Vector store cleared.")


if __name__ == "__main__":
    root = tk.Tk()
    app = RAGApp(root)
    root.mainloop()
