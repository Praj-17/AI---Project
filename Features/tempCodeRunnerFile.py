import cv2
import pytesseract
import numpy as np
from PIL import ImageGrab
import time

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
img = cv2.imread('test\\try.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
pytesseract