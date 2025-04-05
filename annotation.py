from PIL import Image, ImageTk
import json
import os
import tkinter as tk
from tkinter import messagebox, Toplevel
import argparse

class CoordinateSaver:
    def __init__(self, master, image_path):
        self.master = master
        self.master.title("Save Coordinates")

        # Load the image
        self.image_path = image_path  # Use the provided image path
        self.image = Image.open(self.image_path)
        self.original_width, self.original_height = self.image.size

        # Calculate dimensions to maintain aspect ratio
        self.max_width = 1200
        self.max_height = 900
        self.image.thumbnail((self.max_width, self.max_height))
        self.photo = ImageTk.PhotoImage(self.image)

        # Create a label to display the image
        self.label = tk.Label(master, image=self.photo)
        self.label.pack()

        # Dictionary to save coordinates
        self.coordinates_data = {}
        self.click_count = 1
        self.max_clicks = 36

        # Bind the click event
        self.label.bind("<Button-1>", self.save_coordinates)

        # Frame for bottom controls
        self.bottom_frame = tk.Frame(master)
        self.bottom_frame.pack(side="bottom", pady=10)

        # Initial instruction message
        self.instruction_label = tk.Label(self.bottom_frame, text=f"Click on point {self.click_count} on the image.")
        self.instruction_label.pack()

        # Button to mark as "occluded/missing"
        self.missing_button = tk.Button(self.bottom_frame, text="Occluded/Missing", command=self.mark_missing)
        self.missing_button.pack(side="left", padx=5)

        # Button to show reference image
        self.reference_button = tk.Button(self.bottom_frame, text="Show Reference", command=self.show_reference)
        self.reference_button.pack(side="right", padx=5)

    def show_instruction(self):
        self.instruction_label.config(text=f"Click on point {self.click_count} on the image.")

    def save_coordinates(self, event):
        # Calculate coordinates relative to the original image
        x = int(event.x * (self.original_width / self.photo.width()))
        y = int(event.y * (self.original_height / self.photo.height()))

        # Save the coordinates in the dictionary
        self.coordinates_data[self.click_count] = {
            "coordinates": {
                "x": x,
                "y": y
            },
            "status": "ok"
        }

        # Print the dictionary in JSON format
        print(json.dumps(self.coordinates_data, indent=4))

        self.click_count += 1

        # Show the message for the next click
        if self.click_count <= self.max_clicks:
            self.show_instruction()
        else:
            self.save_to_file()  # Save to file when done
            messagebox.showinfo("Finished", "You have clicked all the points!")
            self.master.quit()  # Close the application after saving

    def mark_missing(self):
        self.coordinates_data[self.click_count] = {
            "status": "occluded/missing"
        }
        print(json.dumps(self.coordinates_data, indent=4))
        self.click_count += 1  # Move to the next point
        if self.click_count <= self.max_clicks:
            self.show_instruction()
        else:
            self.save_to_file()  # Save to file when done
            messagebox.showinfo("Finished", "You have clicked all the points!")
            self.master.quit()  # Close the application after saving

    def show_reference(self):
        # Open a new top-level window
        ref_window = Toplevel(self.master)
        ref_window.title("Reference Image")
        
        # Load and display the reference image
        ref_image_path = "court.png"  # Reference image file
        if os.path.exists(ref_image_path):
            ref_image = Image.open(ref_image_path)
            ref_photo = ImageTk.PhotoImage(ref_image)
            ref_label = tk.Label(ref_window, image=ref_photo)
            ref_label.image = ref_photo  # Keep a reference
            ref_label.pack()
        else:
            messagebox.showerror("Error", "Reference image not found!")

    def save_to_file(self):
        # Create the output filename
        base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        output_filename = f"annotation-dist/{base_name}-ann.json"

        # Ensure the output directory exists
        os.makedirs("annotation-dist", exist_ok=True)

        # Save the coordinates data to a JSON file
        with open(output_filename, 'w') as json_file:
            json.dump(self.coordinates_data, json_file, indent=4)
        print(f"Coordinates saved to {output_filename}")

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Annotate image coordinates.")
    parser.add_argument("number", type=int, help="The number to append to the image filename.")
    args = parser.parse_args()

    # Construct the image path based on the input number
    image_path = f"annotation-images/out{args.number}.jpg"  # Adjust the filename pattern as needed

    # Check if the image file exists
    if not os.path.isfile(image_path):
        print(f"Error: The file {image_path} does not exist.")
    else:
        root = tk.Tk()
        app = CoordinateSaver(root, image_path)
        root.mainloop()
