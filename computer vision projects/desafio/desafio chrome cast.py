# Aplicação utilizando OpenCV que abre a stream da camera
# e permite ao usuário clicar com o botão esquerdo do
# mouse sobre um ponto na área do video, selecionando os pixels baseado na cor
# de onde foi clicado.
#
# O programa compara o valor da cor (ou tom de cinza) de todos os pixels
# do frame com o valor da cor (ou tom de cinza) de onde foi clicado.
# Se a diferença entre esses valores for menor que 13, o pixel é marcado com a cor
# vermelha e exiba o resultado na tela.

# Imports
import cv2
import os
import numpy as np
import time

# Global variables (window name, the actual selected color and actual frame of video)
window_name = "Video CAM"
selectedColor = None
actual_frame = None
imagem = cv2.imread("test.jpg")

# Run the video opened, and show pixels nearest of selected color
# Press Q to exit
def runVideo(video):
    global actual_frame

    # delay of frame in miliseconds
    delay_time = 25

    # Set window name
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, handlerMouseEvent)

    # Read until video is completed
    while(video.isOpened()):
        # Capture frame-by-frame
        actual_frame = video.read()[1]

        if(selectedColor):
            # Display the frame with selected pixels
            start = time.time()
            selectIqualColors(selectedColor, 13)
            delay_time = 25 - (time.time() - start)*1000

            # keeps delay positive greater than 1
            if(delay_time < 1):
                delay_time = 1
        else:
            # Display the resulting frame
            cv2.imshow(window_name,actual_frame)

        # Press Q on keyboard to  exit
        if cv2.waitKey(int(delay_time)) & 0xFF == ord('q'):
            break


# Receive an frame as numpy matrix, and select all pixels that have similar color
def selectIqualColors(color, range):
    global imagem
    new_frame = actual_frame.copy()

    # Calculate the distance of pixels to selected color
    distances = (new_frame - color) ** 2
    distances = np.sum(distances, axis = 2)
    range = range ** 2

    # Get pixels that is nearest of selected color, and set as red
    new_frame[distances < range] = imagem

    # Show frame with red pixels at place of selected color
    cv2.imshow(window_name, new_frame)


# Handler with mouse event over the frame
# Setting the selected color
def handlerMouseEvent(event, x, y, flags, param):
    global selectedColor

    # if the left mouse button was clicked
    if(event == cv2.EVENT_LBUTTONDOWN):
        # Set the new color selected
        b,g,r = int(actual_frame[y,x,0]), int(actual_frame[y,x,1]), int(actual_frame[y,x,2])
        selectedColor = (b,g,r)


# Get an video from input, and open her
def OpenAnVideo():
    # Read movie
    video = cv2.VideoCapture(0)

    # Run video with selected pixels
    print("Rodando Camera. Aperte 'Q' para sair!")
    runVideo(video)

    # Release video and destroy video window
    video.release()
    cv2.destroyAllWindows()


def main():
    OpenAnVideo()

if __name__ == "__main__":
    main()
