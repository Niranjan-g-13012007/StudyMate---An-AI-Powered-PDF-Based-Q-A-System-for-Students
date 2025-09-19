from fpdf import FPDF

# Create instance of FPDF class
pdf = FPDF()

# Add a page
pdf.add_page()

# Set font
pdf.set_font("Arial", size=12)

# Read the text file
with open("test_document.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Add the text to the PDF
pdf.multi_cell(0, 10, txt=text)

# Save the PDF
pdf.output("test_document.pdf")

print("PDF created successfully: test_document.pdf")
