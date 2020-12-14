README.txt:

Autor: George Oliveir Barros
Disciplina: Visão Computacional
PPGI - UnB

Bibliotecas utilizadas:
-matplotlib
-numpy
-cv2
-glob
-sys

*Para executar o Requisito 1:
	(1) Inicialmente é necessáio inserir no código o caminho do diretório onde se encontram as imagens da câmera 1 e câmera 2, para realização das respectivas calibrações.
		- Coloque o caminha das imagens da câmera 1 na linha 89.
		- Coloque o caminha das imagens da câmera 2 na linha 94.
	(2) Após iserir o caminho correto, execute: python trabalho.py --r1.
	(3) Os resultados (matrizes de intrísecos e coeficientes de distorção) serão impressos no terminal e salvos em um arquivo .yaml.

Detalhes de implementação: cv2.findChessboardCorners e cv2.cornerSubPix (Para encontrar pontos e esquinas do chessboard); cv2.calibrateCamera (para calibrar e obter intrísecos).


*Para executar o Requisito 2:
O requisito 2 realiza a calibração antes de ser executado, logo, é possível testar o requisito 2 antes de testar o requisito 1, se for o caso.
	(1) Após iserir o caminho correto (do requisito 1), execute: python trabalho.py --r2.
	(2) Os resultados (vetores de rotação, translação e poses das câmeras) serão impressos no terminal.

Detalhes de implementação: cv2.solvePnP(objP, imgP, mtx, dist) para econtrar posição dos objetos em relação as câmeras e -np.matrix(rotM2).T * np.matrix(tvec2) para obter a pose de cada uma das câmeras.


*Para executar o Requisito 3:
As imagens utilizadas nos requisitos 1 e 2, para calibração, foram redimensionadas e salvas em outros dois diretórios, que se encontram dentro da pasta data, os diretório são: "Calibration1 resize" e "Calibration2 resize". Adicionalmente, os frames utilizados como referência para o cálculo da disparidade se encontram na pasta /data/disparity. Portanto, a execução desta função necessita da preservação da estrutura desses diretórios. Essas informações estão nas linhas 179 e 180 do código fonte.

	(1) Execute: python trabalho.py --r3.
	(2) O mapa de dispaidade aparecerá na tela ao fim da execução e também será salvo na mesma pasta do arquivo trabalho.py.

Detalhes de implementação: cv2.getOptimalNewCameraMatrix para conseguir melhor matriz de intrísecos, cv2.undistort para corrigir distorção, cv2.StereoSGBM_create para casamnto de pontos e cálculo de disparidade, cv2.StereoSGBM_create.stereo.compute para gerar o mapa de disparidade.
