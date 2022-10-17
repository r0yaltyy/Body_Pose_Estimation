import cv2                                          # подключаем openCV2
import mediapipe as mp                              # подключаем mediapipe

cam = cv2.VideoCapture(0)                           # подключаем изображение с камеры в режиме реального времени
mpDraw = mp.solutions.drawing_utils                 # подключаем инструмент для рисования
mp_pose = mp.solutions.pose                         # подключаем раздел распознавания тела
pose = mp_pose.Pose(                                # объект класса "поза"
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

py = [0 for i in range(33)]                         #массив для хранения точек скелета по оси Y
px = [0 for i in range(33)]                         #массив для хранения точек скелета по оси X

hand_raised = [0 for i in range(2)]                 #индикатор поднятых рук
hand_on_head = [0 for i in range(2)]                #индикатор рук на голове
hand_on_shoulder = [0 for i in range(2)]            #индикатор рук на плечах (для перекрещенных рук)
elbow_near_hip = [0 for i in range(2)]              #индикатор правильного положения локтей (для перекрещенных рук)

def distance(point1, point2):                       #функция, возвращающая расстояние между точками по модулю
    return abs(point1 - point2)

while True:                             
    good, img = cam.read()                         #покадрово читаем изображение                     
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  #преобразуем в RGB
    results = pose.process(imgRGB)                 #получаем результат
    
    if results.pose_landmarks:                                      #если удалось распознать руки
        for bodyLms in results.pose_landmarks.landmark:             #читаем координаты каждой точки
            mpDraw.draw_landmarks(img, results.pose_landmarks,      #Соединяем точки линиями
                                   mp_pose.POSE_CONNECTIONS)
           
            for id, point in enumerate(results.pose_landmarks.landmark) :                           #создаем список с координатами точек
                width, height, color = img.shape                                                    #получаем размеры изображения с камеры и масштабируем
                width, height = int(point.x * height), int(point.y * width)

                py[id] = height             #Заполняем массив координатами по оси Y
                px[id] = width              #Заполняем массив координатами по оси X
                
            Good_distance_for_raised_hands = distance(py[12], py[24]) * 5/3     # получаем расстояние, с которым будем сравнивать каждую руку 
                                                                                #(берем расстояние от плеча до бедра и умножаем на кооэффициент, необходимый для заданной точности распознавания)       
            # 0 - правая, 1 - левая
            hand_raised[0] = 1 if distance(py[24], py[16]) > Good_distance_for_raised_hands else 0 #Распознаем, поднята ли правая рука
            hand_raised[1] = 1 if distance(py[23], py[15]) > Good_distance_for_raised_hands else 0 #Распознаем, поднята ли левая рука
            #Вывод сообщений на экран в зависимости от комбинаций поднятых рук
            if (hand_raised[0]) and (hand_raised[1]):
                print("hands up")
            if (hand_raised[0]) and not (hand_raised[1]):
                print("right hand raised")
            if not (hand_raised[0]) and (hand_raised[1]):
               print("left hand raised")

            # получаем расстояние, с которым будем сравнивать руки на голове по горизонтали и вертикали 
            Good_distance_for_hands_on_head_X = distance(px[8], px[7]) * 8/15 
            Good_distance_for_hands_on_head_Y = distance(py[6], py[9]) * 2
            #Распознаем руки на голове (0 - правая , 1 - левая)
            hand_on_head[0] = 1 if (distance(px[8], px[16]) < Good_distance_for_hands_on_head_X and distance(py[8], py[16]) < Good_distance_for_hands_on_head_Y) else 0
            hand_on_head[1] = 1 if (distance(px[7], px[15]) < Good_distance_for_hands_on_head_X and distance(py[7], py[15]) < Good_distance_for_hands_on_head_Y) else 0
            #если обе руки на голове, выводим сообщение
            if (hand_on_head[0]) and (hand_on_head[1]):
                print("hands on head")
       
            #Получаем расстояние по осям X и Y для сравнения кистей на противоположных плечах
            Good_distance_for_cross_hands_X = (distance(px[12], px[11]) / 3)
            Good_distance_for_cross_hands_Y = (distance(px[12], px[11]) / 3)
            #Распознаем руки (0 - правая, 1 - левая) на противоположных плечах по обеим координатам
            hand_on_shoulder[0] = 1 if (distance(px[20], px[11]) < Good_distance_for_cross_hands_X and distance(py[19], py[11]) < Good_distance_for_cross_hands_Y)  else 0
            hand_on_shoulder[1] = 1 if (distance(px[15], px[12]) < Good_distance_for_cross_hands_X and distance(py[15], py[12]) < Good_distance_for_cross_hands_Y)  else 0
            #Получаем расстояние по оси Y для правильного положения локтя (рядом с соответствующим бедром)
            Good_distance_for_elbow_Y = distance(px[24], px[23]) * 3/2
            #Убеждаемся, что локти (0 - правый, 1 - левый) находятся рядом с соответствующими бедрами
            elbow_near_hip[0] = 1 if distance(py[14], py[24]) < Good_distance_for_elbow_Y else 0
            elbow_near_hip[1] = 1 if distance(py[13], py[23]) < Good_distance_for_elbow_Y else 0
            #Выводим на экран распознанные перекрещенные руки
            if (hand_on_shoulder[0] and hand_on_shoulder[1] and elbow_near_hip[0] and elbow_near_hip[1]):
                print("CROSS")
    
    cv2.imshow("Image", img)           # выводим окно с нашим изображением
    if cv2.waitKey(1) == ord('q'):     # ждем нажатия клавиши q в течение 1 мс для прерывания процесса
        break    
