import unittest

def add_numbers(a, b):
    """Simple function to add two numbers"""
    return a + b

def multiply_numbers(a, b):
    """Simple function to multiply two numbers"""
    return a * b

class TestMathOperations(unittest.TestCase):
    """Test class for math operations"""
    
    def test_add_positive_numbers(self):
        """Test adding positive numbers"""
        result = add_numbers(2, 3)
        self.assertEqual(result, 5)
    
    def test_add_negative_numbers(self):
        """Test adding negative numbers"""
        result = add_numbers(-2, -3)
        self.assertEqual(result, -5)
    
    def test_multiply_positive_numbers(self):
        """Test multiplying positive numbers"""
        result = multiply_numbers(4, 5)
        self.assertEqual(result, 20)
    
    def test_multiply_by_zero(self):
        """Test multiplying by zero"""
        result = multiply_numbers(10, 0)
        self.assertEqual(result, 0)

# Run the tests
if __name__ == '__main__':
    unittest.main()
