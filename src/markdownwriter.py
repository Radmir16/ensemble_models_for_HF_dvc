import os

def describe_dataset(path):
    """
    Describes the datasets located at a given path by listing the number of images per class,
    and provides a summary of the total number of images and classes.
    
    Args:
    path (str): The path to the dataset directory.

    Returns:
    dict: A dictionary containing the counts and names of images per class.
    """
    classes = os.listdir(path)  # Get all subdirectories/classes
    description = {}
    
    for class_name in classes:
        class_path = os.path.join(path, class_name)
        if os.path.isdir(class_path):  # Check if it is a directory
            images = [img for img in os.listdir(class_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
            description[class_name] = {
                'image_count': len(images),
                'example_images': images[:5]  # Show up to 5 example image names
            }
    
    return description

class MarkdownWriter:
    def __init__(self, filename):
        self.filename = filename
        self.headings = []  # To store headings for TOC
        # Open the file in write mode to create it or clear it if it already exists
        with open(self.filename, 'w') as file:
            pass

    def header1(self, text):
        """Writes a level 1 header to the markdown file."""
        self.headings.append((1, text))  # Store for TOC
        with open(self.filename, 'a') as file:
            file.write(f'# {text}\n\n')

    def header2(self, text):
        """Writes a level 2 header to the markdown file."""
        self.headings.append((2, text))  # Store for TOC
        with open(self.filename, 'a') as file:
            file.write(f'## {text}\n\n')

    def header3(self, text):
        """Writes a level 3 header to the markdown file."""
        self.headings.append((3, text))  # Store for TOC
        with open(self.filename, 'a') as file:
            file.write(f'### {text}\n\n')

    def print_data(self, text):
        """Writes a paragraph of text to the markdown file."""
        with open(self.filename, 'a') as file:
            file.write(f'{text}\n\n')

    def print_config(self, config):
        """Formats and writes a configuration dictionary to the markdown file."""
        if 'dataset' in config:
            self.header3('Dataset')
            for key, value in config['dataset'].items():
                self.print_data(f"- **{key.capitalize()}**: `{value}`")
        
        if 'model' in config:
            self.header3('Model Configuration')
            for key, value in config['model'].items():
                self.print_data(f"- **{key.capitalize()}**: `{value}`")
        
        if 'train' in config:
            self.header3('Training Configuration')
            for key, value in config['train'].items():
                self.print_data(f"- **{key.capitalize()}**: {value}")
        
        if 'metrics' in config:
            self.header3('Metrics')
            for metric in config['metrics']:
                self.print_data(f"- {metric.capitalize().replace('_', ' ')}")

    def write_toc(self, report):
        """Generates and writes the Table of Contents based on stored headings."""
        toc = []
        toc.append(f"# {report}\n")
        for level, text in self.headings:
            indent = '  ' * (level - 1)
            # Create a slug from the heading text - lowercase and replace spaces with dashes
            slug = text.replace(' ', '-').lower()
            toc.append(f"{indent}- [{text}](#{slug})\n")

        # Write TOC at the beginning of the file by reading the current content first
        with open(self.filename, 'r+') as file:
            content = file.read()
            file.seek(0, 0)
            file.write(''.join(toc) + '\n' + content)
            
    def add_image(self, image_path, alt_text="Image", title=""):
        """Writes an image link to the markdown file."""
        with open(self.filename, 'a') as file:
            title_str = f' "{title}"' if title else ""
            file.write(f'![{alt_text}]({image_path}{title_str})\n\n')
            
    def describe_dataset_markdown(self, description, title="Dataset Summary"):
        """Formats and writes a dataset description to the markdown file as a table, including total image count."""
        total_images = sum(info['image_count'] for info in description.values())
        self.header2(title)
        self.print_data(f"**Total Number of Images**: {total_images}\n")

        # Start the table without example images
        table_header = "| Class Name | Number of Images |\n"
        table_divider = "| --- | --- |\n"
        table_rows = ""
        for class_name, info in description.items():
            table_rows += f"| {class_name} | {info['image_count']} |\n"
        
        # Write the table
        self.print_data(table_header + table_divider + table_rows)