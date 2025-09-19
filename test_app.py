import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class TestStudyMate(unittest.TestCase):
    """Test cases for StudyMate application."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a test uploads directory if it doesn't exist
        self.upload_dir = os.path.join(os.path.dirname(__file__), 'uploads')
        os.makedirs(self.upload_dir, exist_ok=True)
        
        # Create a test PDF file
        self.test_pdf_path = os.path.join(self.upload_dir, 'test_document.pdf')
        with open(self.test_pdf_path, 'wb') as f:
            f.write(b'%PDF-1.4\n1 0 obj\n<< /Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<< /Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<< /Type /Page\n/Parent 2 0 R\n/Resources << /Font << /F1 4 0 R >> >>\n/MediaBox [0 0 612 792]\n/Contents 5 0 R\n>>\nendobj\n4 0 obj\n<< /Type /Font\n/Subtype /Type1\n/BaseFont /Helvetica\n>>\nendobj\n5 0 obj\n<< /Length 44 >>\nstream\nBT\n/F1 24 Tf\n100 700 Td\n(Test Document) Tj\nET\nendstream\nendobj\nxref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000109 00000 n \n0000000223 00000 n \n0000000258 00000 n \ntrailer\n<< /Size 6\n/Root 1 0 R\n>>\nstartxref\n389\n%%EOF')

    def tearDown(self):
        """Clean up after each test method."""
        # Remove test files
        if os.path.exists(self.test_pdf_path):
            os.remove(self.test_pdf_path)

    @patch('app.app')
    def test_app_creation(self, mock_app):
        """Test that the Flask app is created correctly."""
        from app import create_app
        app = create_app()
        self.assertIsNotNone(app)

    @patch('pdf_utils.PDFProcessor')
    def test_pdf_processing(self, mock_pdf_processor):
        """Test PDF processing functionality."""
        from pdf_utils import PDFProcessor
        
        # Mock the PDFProcessor
        mock_processor = MagicMock()
        mock_processor.extract_text.return_value = "Test document content"
        mock_processor.chunk_text.return_value = ["Test document content"]
        mock_pdf_processor.return_value = mock_processor
        
        # Test text extraction
        processor = PDFProcessor()
        text = processor.extract_text(self.test_pdf_path)
        self.assertEqual(text, "Test document content")
        
        # Test text chunking
        chunks = processor.chunk_text(text)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], "Test document content")

    @patch('llm_granite.GraniteLLM')
    def test_llm_integration(self, mock_llm):
        """Test LLM integration."""
        from llm_granite import GraniteLLM
        
        # Mock the LLM
        mock_llm_instance = MagicMock()
        mock_llm_instance.generate_answer.return_value = "Test answer"
        mock_llm.return_value = mock_llm_instance
        
        # Test answer generation
        llm = GraniteLLM()
        answer = llm.generate_answer("Test question", ["Test context"])
        self.assertEqual(answer, "Test answer")

if __name__ == '__main__':
    unittest.main()
