from PIL import Image, ImageTk
import json
import os
import tkinter as tk
from tkinter import messagebox
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
        self.max_clicks = 14

        # Bind the click event
        self.label.bind("<Button-1>", self.save_coordinates)

        # Button to mark as "occluded/missing"
        self.missing_button = tk.Button(master, text="Occluded/Missing", command=self.mark_missing)
        self.missing_button.pack()

        # Initial instruction message
        self.show_instruction()

    def show_instruction(self):
        messagebox.showinfo("Instruction", f"Click on point {self.click_count} on the image.")

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
        messagebox.showinfo("Info", f"Point {self.click_count} marked as 'occluded/missing'.")
        self.click_count += 1  # Move to the next point
        if self.click_count <= self.max_clicks:
            self.show_instruction()
        else:
            self.save_to_file()  # Save to file when done
            messagebox.showinfo("Finished", "You have clicked all the points!")
            self.master.quit()  # Close the application after saving

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

    # Construct the image
    # Construct the image path based on the input number
    image_path = f"annotation-images/out{args.number}.jpg"  # Adjust the filename pattern as needed

    # Check if the image file exists
    if not os.path.isfile(image_path):
        print(f"Error: The file {image_path} does not exist.")
    else:
        root = tk.Tk()
        app = CoordinateSaver(root, image_path)
        root.mainloop()
