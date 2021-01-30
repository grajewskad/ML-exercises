import cv2
import numpy as np
import time
"""
# CZĘŚĆ 1
# wczytywanie mediów
# wczytaj zdjęcie
img = cv2.imread("photos/dominika.jpg")
# wyświetl zdjęcie
cv2.imshow("output", img)
# wyświetl je dopóki go nie zamknę, 
# w przeciwnym wypadku wyświetli się na 1 milisekundę.
cv2.waitKey(0)
# ----------------
# wyswietl video z podanej ściezki
cap = cv2.VideoCapture("photos/video.mp4")
while True:
    success, img = cap.read()
    cv2.imshow("video", img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

# ----------------
# wyświetl obraz z kamery
cap = cv2.VideoCapture(0)
time.sleep(2)
# szerokość
cap.set(3, 640)
# wysokość
cap.set(4, 480)
# ustaw jasność obrazu
cap.set(10, 100)
while True:
    success, img = cap.read()
    cv2.imshow("video", img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

# ----------------
"""
# CZĘŚĆ 2
# podstawowe funkcje obrazu - transformacje

# wczytaj zdjęcie
img = cv2.imread("photos/dominika.jpg")
"""
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# wyświetl zdjęcie
cv2.imshow("Gray", imgGray)
cv2.waitKey(0)
# ----------------
imgBlur = cv2.GaussianBlur(imgGray, (9,9), 0)
#wyświetl zdjęcie
cv2.imshow("Blur", imgBlur)
cv2.waitKey(5000)

# ----------------
imgBlur = cv2.GaussianBlur(imgGray, (9,9), 0)
#wyświetl zdjęcie
cv2.imshow("Blur", imgBlur)
cv2.waitKey(5000)

# ----------------
# Kanny krawędzie
imgCanny = cv2.Canny(img, 100, 100)

cv2.imshow("Canny", imgCanny)
cv2.waitKey(5000)

# ----------------
# dylatacja obrazu do lepszego wyznaczania krawędzi
# grube krawędzie

# jądro do dylatacji - maciez jednostkowa 5 x 5
kernel = np.ones((5, 5), np.uint8)

imgDilatation = cv2.dilate(imgCanny, kernel, iterations=1)

cv2.imshow("Dilatation", imgDilatation)
cv2.waitKey(5000)

# ----------------
# erocja - zrobi krawędzie cieńsze

imgEroded = cv2.erode(imgDilatation, kernel, iterations=1)
cv2.imshow("Eroded", imgEroded)
cv2.waitKey(5000)

# ----------------
#CZĘŚĆ 3 
# zmiana rozmiarów, wycinanie

imgResize = cv2.resize(img, (300,500))
cv2.imshow("ImageResize", imgResize)
cv2.waitKey(5000)
# ----------------

# wycinanie - traktujemy obraz jak macierz

imgCroped = img[300:600, 200:500]
cv2.imshow("ImgCropped", imgCroped)
cv2.waitKey(5000)

"""
# ----------------
#CZĘŚĆ 4
# rysowanie kształtów
# 512, 512 to wymiary, a 3 to kolory rgb
img = np.zeros((512, 512, 3), np.uint8)
#ustawienie koloru w rgb - cały obraz
"""
img[:] = 255, 255, 0
cv2.imshow("Image", img)
cv2.waitKey(5000)
"""
# ----------------
# rysowanie prostokątów

img[100:200, 50:150] = 0, 255, 0
"""
cv2.imshow("Image", img)
cv2.waitKey(5000)

# ----------------
#rysowanie linii na obrazku - start, koniec, kolor, grubość

cv2.line(img, (0,0), (300, 300), (0, 0, 130), 3)
cv2.imshow("Image", img)
cv2.waitKey(5000)

# ----------------
#rysowanie linii na obrazku - przekątna

cv2.line(img, (0,0), (img.shape[1], img.shape[0]), (0, 0, 130), 3)
cv2.imshow("Image", img)
cv2.waitKey(5000)

# ----------------
#rysowanie prostokąta - wypełniony

cv2.rectangle(img, (0,0), (240,100), (0, 0, 130), cv2.FILLED)

#rysowanie koła, definiowane centrum i promień, kolor
cv2.circle(img, (200, 200), (30), (120, 255, 30))
#text - text, miejsce, font, skala, kolor, grubość
cv2.putText(img, "kocham Cie", (300,300), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
cv2.imshow("Image", img)
cv2.waitKey(5000)

# ----------------
#CZĘŚĆ 5

#perspektywa

img = cv2.imread("photos/cards.jpg")
width, height = 200,300
pts1 = np.float32([[345,106],[231,261], [136,190], [244, 41]])
pts2 = np.float32([[0,0],[width,0], [0,height], [width, height]])
matrix = cv2.getPerspectiveTransform(pts1, pts2)
imgOutput = cv2.warpPerspective(img, matrix, (width, height))
cv2.imshow("Image", imgOutput)
cv2.waitKey(5000)


# ----------------
#CZĘŚĆ 6
# ładowanie kilku obrazów w jeden
img = cv2.imread("photos/dominika.jpg")
#wyświetl dwa razy to samo zdjęcie horyzontalnie
imgHor = np.hstack([img, img])
#wyświtl dwa razy ro samo zdjęcie wertykalnie
# (zdjęcia muszą mieć ten sam numer channelu (kolor))
imgVer = np.vstack([img, img])
cv2.imshow("Image", imgVer)
cv2.waitKey(5000)


# ----------------
#CZĘŚĆ 7
# detekcja koloru ze zdjęcia
img = cv2.imread("photos/light.jpg")
def empty(a):
   pass

#pasek do wyboru wartości, nazwa, okno, pozycja startowa wyboru, zakres wyboru
cv2.namedWindow("Trackbar")
cv2.resizeWindow("Trackbar", 600, 200)
cv2.createTrackbar("HueMin", "Trackbar", 0, 179, empty)
cv2.createTrackbar("HueMax", "Trackbar", 179, 179, empty)
cv2.createTrackbar("SatMin", "Trackbar", 0, 255, empty)
cv2.createTrackbar("SatMax", "Trackbar", 255, 255, empty)
cv2.createTrackbar("ValMin", "Trackbar", 0, 255, empty)
cv2.createTrackbar("ValMax", "Trackbar", 255, 255, empty)

#odwrócone kolory
imgVSH = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#pobieraj ciągle od uzytkownika wartosci
while True:
   h_min = cv2.getTrackbarPos("HueMin", "Trackbar")
   h_max = cv2.getTrackbarPos("HueMax", "Trackbar")
   s_min = cv2.getTrackbarPos("SatMin", "Trackbar")
   s_max = cv2.getTrackbarPos("SatMax", "Trackbar")
   v_min = cv2.getTrackbarPos("ValMin", "Trackbar")
   v_max = cv2.getTrackbarPos("ValMax", "Trackbar")
   print(h_min, h_max, s_min, s_max, v_min, v_max)
   lower = np.array([h_min, s_min, v_min])
   higher = np.array([h_max, s_max, v_max])
#maska - zaaplikuj te wartości na zdjęciu
   mask = cv2.inRange(imgVSH, lower, higher)
#połącz zdjęcie z maską, tam gdzie jest białe zamaskowane, tam będzie kolor w outputs
   imgResult = cv2.bitwise_and(img, img, mask=mask)
   cv2.imshow("Image", imgResult)
   cv2.waitKey(1)



# detekcja koloru z wideo
cap = cv2.VideoCapture(0)
time.sleep(2)
# szerokość
cap.set(3, 640)
# wysokość
cap.set(4, 480)
# ustaw jasność obrazu
cap.set(10, 100)


def empty(a):
   pass

#pasek do wyboru wartości, nazwa, okno, pozycja startowa wyboru, zakres wyboru
cv2.namedWindow("Trackbar")
cv2.resizeWindow("Trackbar", 600, 200)
cv2.createTrackbar("HueMin", "Trackbar", 0, 179, empty)
cv2.createTrackbar("HueMax", "Trackbar", 179, 179, empty)
cv2.createTrackbar("SatMin", "Trackbar", 0, 255, empty)
cv2.createTrackbar("SatMax", "Trackbar", 255, 255, empty)
cv2.createTrackbar("ValMin", "Trackbar", 0, 255, empty)
cv2.createTrackbar("ValMax", "Trackbar", 255, 255, empty)

#odwrócone kolory
imgVSH = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#pobieraj ciągle od uzytkownika wartosci
while True:
   _, img = cap.read()
   imgVSH = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

   h_min = cv2.getTrackbarPos("HueMin", "Trackbar")
   h_max = cv2.getTrackbarPos("HueMax", "Trackbar")
   s_min = cv2.getTrackbarPos("SatMin", "Trackbar")
   s_max = cv2.getTrackbarPos("SatMax", "Trackbar")
   v_min = cv2.getTrackbarPos("ValMin", "Trackbar")
   v_max = cv2.getTrackbarPos("ValMax", "Trackbar")

   print(h_min, h_max, s_min, s_max, v_min, v_max)

   lower = np.array([h_min, s_min, v_min])
   higher = np.array([h_max, s_max, v_max])
   #maska - zaaplikuj te wartości na zdjęciu
   mask = cv2.inRange(imgVSH, lower, higher)
   #połącz zdjęcie z maską, tam gdzie jest białe zamaskowane, tam będzie kolor w outputs
   imgResult = cv2.bitwise_and(img, img, mask=mask)
   cv2.imshow("Image", imgResult)
   if cv2.waitKey(1) & 0xFF==ord('q'):
      break



# ----------------
#CZĘŚĆ 8
#rozpoznawanie kształtów
img = cv2.imread("photos/shapes.jpg")
imgCont = img.copy()
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (3,3), 1)
imgVer = np.vstack([imgGray, imgBlur])
#cv2.imshow("Image", imgVer)
#cv2.waitKey(5000)
imgCanny = cv2.Canny(imgBlur, 50, 50)

cv2.imshow("Image", imgCanny)
cv2.waitKey(5000)

def getContours(img):
   #contours daje nam po kolei kazdy ksztalt i miejsce corners 
   contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
   for cnt in contours:
      #policz pole tej figury
      area = cv2.contourArea(cnt)
      
      print(area)
      
      #narysuj kształty - zdjęcie oryginalne, kontur, -1 bo chcemy wszystkie boki, kolor, grubość
      cv2.drawContours(imgCont, cnt, -1, (0, 0, 255), 5)
      #obwód, True bo kształt zamknięty
      peri = cv2.arcLength(cnt, True)
      #wygładzenie konturów - o ile przyblienie? o 2% peri w poniszym przypadku
      approx = cv2.approxPolyDP(cnt, 0.1*peri, True)
      # ilu kątny jest kształt?
      objCor = (len(approx))
      #zamykanie kształtów w ramce prostokątnej
      x, y, width, height = cv2.boundingRect(approx)

      if objCor == 3: objType="Triangle"
      elif objCor==4: 
         #sprawdmy czy kwadrat czy prostokąt
         aspRatio = width/float(height)
         if aspRatio>0.95 or aspRatio<1.05:
            objType="square"
         else:
            objType="Rectangle"
      else:
         objType="Noname"

      #rysowanie - na jakim, początek, koniec, kolor, grubość
      cv2.rectangle(imgCont, (x, y), (x + width, y + height), (0, 255, 0), 2 )
      cv2.putText(imgCont, objType, (int(x+(width/2)), int(y+(height/2))), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 2)

      

getContours(imgCanny)

cv2.imshow("Image", imgCont)
cv2.waitKey(20000)

# CZĘŚĆ 9
#rozpoznawania twarzy
#pobranie zbioru wyuczonego na twarz
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
img = cv2.imread("photos/dominika.jpg")
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY )
#pobranie współrzęcnych na twarze
faces = faceCascade.detectMultiScale(img, 1.1, 4)

for x, y, w, h in faces:
   #zaznaczenie twarzy
   cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
   cv2.imshow("Image", img)
   cv2.waitKey(20000)

"""

#PROJEKT 1
#VIRTUALNE MALOWANIE

#przechwyć obraz z kamery

cap = cv2.VideoCapture(0)
time.sleep(2)
# szerokość
cap.set(3, 640)
# wysokość
cap.set(4, 480)
# ustaw jasność obrazu
cap.set(10, 100)
myColors = [18,100, 189,255, 89, 255]
def find_colors(img, myColors):
   imgVSH = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
   lower = np.array(myColors[0:3])
   higher = np.array(myColors[3:6])
   mask = cv2.inRange(imgVSH, lower, higher)
   cv2.imshow("img", mask)

while True:
   _, img = cap.read()
   find_colors(img, myColors)
   cv2.imshow("Image", img)
   if cv2.waitKey(1) & 0xFF==ord('q'):
      break
