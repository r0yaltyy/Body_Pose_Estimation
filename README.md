1.В данной программе описано распознавание неестественных поз для нескольких людей с помощью Python3. Все действия выполнены на Ubuntu 20.04.5 LTS 
Данная программа на основе геометрического анализа кинематического изображения человека с хорошей точностью распознает такие положения как: поднятые руки (отдельно правая, отдельно левая, одновременно правая и левая), руки на голове и скрещенные на груди руки.

2.Необходимые компоненты - устройство, на котором будет запущен python3 код и видеокамера, с которой будет считываться изображение в режиме реального времени.

3.Установка OpenCV и MediaPipe.

OpenCV (Open Source Computer Vision Library) — это открытая библиотека для работы с алгоритмами компьютерного зрения, машинным обучением и обработкой изображений. 
Она нам необходима для считывания изображения и работы с ним в дальнейшем.
Подробнее: https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html
Для установки необходимо ввести в терминал: 
    
    pip3 install opencv-python

MediaPipe — это фреймворк с открытым исходным кодом, представленный Google, который помогает создавать мультимодальные конвейеры машинного обучения. Эта структура может использоваться для различных приложений для обработки изображений и мультимедиа, таких как обнаружение объектов, распознавание лиц, отслеживание рук, распознавание тела человека. С помощью него будет отслеживаться кинематическая модель человека, с достаточно хорошими показателями точности.
Подробнее: https://google.github.io/mediapipe/
Для установки необходимо ввести в терминал: 

    pip3 install mediapipe

4.Запуск программы.
Для запуска программы необходимо через терминал перейти к директории с исполняемым файлом multi_body_estimation.py и ввести команду: python3 multi_body_estimation.py 

5.Объяснение программы распознавания неестественных поз человека. 
Разобьем программу на несколько Блоков и подробно опишем каждый.

Блок 1. - основной:
Этап 1. (стр 1-2)
Необходимо подключить предварительно установленные модули OpenCV и MediaPipe

    import cv2
    import mediapipe as mp

Этап 2. (стр 119-127)
    Указывается, хотим ли мы выводить изображение и рисовать на нем.
    Подключение из mediapipe таких компонент как: инструмент для рисования точек и их связей на выведенном видеоизображении, раздел для распознавания тела человека в кинематическом формате.
    И создание объектна класса "поза" из модуля mediapipe
    А также, подключение изображения в режиме реального времени (0 - порядковый номер камеры, вместо него можно указать mp3 файл, если имеется необходимость проверить работу программы на предварительно записанном видеоряде).
    
    
    show_video = True                                  #ключ показа видео
    draw_mode = True                                   #ключ рисования на видео

    mp_pose = mp.solutions.pose                         #подключаем раздел распознавания тела
    pose = mp_pose.Pose(static_image_mode=True)         #объект класса "поза"
    if draw_mode:
        mp_draw = mp.solutions.drawing_utils            #подключаем инструмент для рисования

    cap = cv2.VideoCapture(0)                           #подключаем изображение с камеры в режиме реального времени 


Этап 3.(стр 129-137)
    В данном этапе запускается бесконечный цикл, внутри которого читается каждый кадр изображения, для него вызывается функция распознавания позы нескольких людей, о которой будет сказано далее. Выводится изображение и создается ключ для выхода из программы - в данном случае клавиша ESC

    while True:                                         
        _, Image = cap.read()                           #в бесконечном цикле читаем изображение
        Multi_people_estimation(Image)                  #вызываем функцию распознавания людей и их поз
        if show_video:
            cv2.imshow("cam", Image)                    #выводим изображение на экран
        k = cv2.waitKey(1)                              #завершаем работу программы на ESC
        if k == 27:  # close on ESC key
            break
    cv2.destroyAllWindows()


Блок 2: (стр 115-117) Функция Multi_people_estimation 
    Данная функция создаст копию изображения и вызовет распознавание всех людей на кадре, передавая исходное и скопированное изображение (функция pose_recursive).

    def Multi_people_estimation(Image):                    #Функция распознавания позы нескольких людей (получает на вход изображение)    
        img = cv2.cvtColor(Image, cv2.COLOR_BGR2RGB)       #делаем копию изображения
        pose_recursive(Image, img)                         #вызываем функцию распознавания нескольких людей, внутри нее для каждого человека вызовется функция распознавания позы
Блок 3: (стр 66-112 )Функция pose_recursive
    Данная функция получит на вход изображение, его копию и глубину рекурсии, после создаст необходимые переменные и если удастся распознать человека, она вызовет функцию распознавания позы(BP_estimation см блок 4).
    Далее, если глубина рекурсии позволяет, она закрасит область, в которой находится человек в копию изображения и снова вызовет саму себя для распознавания следующего человека. 
    
    def pose_recursive(origin_image, image, recursion_depth = 5):   #функция распознавания нескольких людей (глубина рекурсии показывает максимальное количество людей)
                                                                    #В оригинальном изображении хранится наш видеоряд, который в конечном итоге выводится, 
                                                                    #в копии хранится картинка, в которой распознанные люди закрашиваются с каждым новым вызовом функции
        results = pose.process(image)        #получаем результат
        X , Y = [], []                       #создаем массивы, которые в дальнейшем заполним координатами X и Y
        h, w = image.shape[:2]               #получаем масштаб изображения (height и weight)
        person_detected = False              #переменная, показывающая считался ли человек
    
        pose_state = 0                       # состояние позы человека
        body_pose = [                        #массив с расшифровкой состояний
        'BP_UNKNOWN',
        'BP_HANDS_UP',
        'BP_RIGHT_HAND_RAISED',
        'BP_LEFT_HAND_RAISED',
        'BP_HANDS_ON_HEAD',
        'BP_CROSS']
    
        if results.pose_landmarks:                                        #если удалось получить точки скелета человека
            pose_state = BP_estimation(results.pose_landmarks.landmark)   #вызываем функцию распознавания человека и присваиваем вернувшееся значение переменной состояния
            print( 6 - recursion_depth, " : ", body_pose[pose_state])     #выводим на экран номер человека и расшифровку его позы
        
            if draw_mode:
                mp_draw.draw_landmarks(origin_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS) #рисуем точки и соединяем их, получая скелет на изображении
        
            for i in range(len(results.pose_landmarks.landmark)):         #получаем два массива координат по x и y
                X.append(results.pose_landmarks.landmark[i].x)
                Y.append(results.pose_landmarks.landmark[i].y)
            person_detected = True                                        #говорим, что удалось распознать человека
        
        if person_detected and recursion_depth > 0:                       #если распознан человек и глубина рекурсии, заданая в аргументе функции больше нуля
            #получаем координаты наиольших и наименьших точек по обеим осям (В данном случае наименьшая координата по оси Y будет самой выскокой на изображении, наибольшая координата - самой низкой)
            x1 = int(min(X) * w)
            y1 = int(min(Y) * h)
            x2 = int(max(X) * w)
            y2 = int(max(Y) * h)
        
            padding = 20                                                   #добавляем необходимый отступ вокруг человека 
            x1 = x1 - padding if x1 - padding > 0 else 0
            y1 = y1 - 3 * padding if y1 - 3 * padding > 0 else 0
            x2 = x2 + padding if x2 + padding < w else w
            y2 = y2 + padding if y2 + padding < h else h
        
            if draw_mode:
                cv2.rectangle(origin_image, (x1,y2), (x2,y1), color = (0,255,0), thickness = 1) #рисуем прямоугольник вокруг человека
            image[y1:y2, x1:x2] = 0                                                             #Делаем каждый пиксель прямоугольника, где есть человек, черным 
            pose_recursive(origin_image, image, recursion_depth - 1)                            #вызываем функцию снова, передавая исходное изображение и изображение с закрашенным человеком, глубину рекурсии уменьшаем

Блок 4:(стр 4-62) Функция BP_estimation
Данную функцию разобьем на несколько этапов.

Этап 0. (стр 4-5) Функция, которая вернет расстояние между двумя точками по модулю.

    def distance(point1, point2):                       #функция, возвращающая расстояние между точками по модулю
        return abs(point1 - point2)

Этап 1. (стр 9-15) Создание нескольких необходимых переменных

    py = [0 for i in range(33)]                         #массив для хранения точек скелета по оси Y
    px = [0 for i in range(33)]                         #массив для хранения точек скелета по оси X

    hand_raised = [0 for i in range(2)]                 #индикатор поднятых рук
    hand_on_head = [0 for i in range(2)]                #индикатор рук на голове
    hand_on_shoulder = [0 for i in range(2)]            #индикатор рук на плечах (для перекрещенных рук)
    elbow_near_hip = [0 for i in range(2)]              #индикатор правильного положения локтей (для перекрещенных рук)

Этап 2. стр(18-23) Получаем координаты каждой точки и масштабируем их (нумерация точек в файле Points.png)

    for id, point in enumerate(landmarks) :                           #создаем список с координатами точек
        width, height, color = Image.shape                            #получаем размеры изображения с камеры и масштабируем
        width, height = int(point.x * height), int(point.y * width)

        py[id] = height             #Заполняем массив координатами по оси Y
        px[id] = width              #Заполняем массив координатами по оси X

Этап 3.(стр 25-36)
В данном этапе реализовано распознавание поднятых рук.
Необходимо получить допустимое расстояние по оси Y, с которым мы будем сравнивать расстояние между какой-либо точкой тела и кистью для заполнения индикаторов поднятой или опущенной руки.
В данном примере использовано расстоние от плеча до бедра (точки 12 и 24 (см. Points.png)), умноженное на коэффициент, удовлетворяющий необходимой точности.
После чего для каждой руки выполняется проверка: Находится ли необходимая рука (точка кисти) на достаточном расстоянии от соответствующего ей бедра.
В зависимости от полученных результатов функция вернет число, которое является состоянием позы.

    Good_distance_for_raised_hands = distance(py[12], py[24]) * 5/3     # получаем расстояние, с которым будем сравнивать каждую руку 
                                                                        #(берем расстояние от плеча до бедра и умножаем на кооэффициент, необходимый для заданной точности распознавания)       
    # 0 - правая, 1 - левая                                             #по этому принципу будем работать с каждой позой
    hand_raised[0] = 1 if distance(py[24], py[16]) > Good_distance_for_raised_hands else 0 #Распознаем, поднята ли правая рука
    hand_raised[1] = 1 if distance(py[23], py[15]) > Good_distance_for_raised_hands else 0 #Распознаем, поднята ли левая рука
    #возвращаем значения в зависимости от комбинаций поднятых рук
    if (hand_raised[0]) and (hand_raised[1]):
        return 1
    if (hand_raised[0]) and not (hand_raised[1]):
        return 2
    if not (hand_raised[0]) and (hand_raised[1]):
        return 3


Этап 4. (стр 38-46) 
На данном этапе реализовано распознавание человека, который держит обе руки на голове.
По тому же принципу получаем допустимое расстояние по обеим осям, с которым будем сравнивать расстояние между точками кисти и различными точками лица.
Заполняем индикаторы для обеих рук и возвращаем состояние позы, если ее удалось распознать.

    # получаем расстояние, с которым будем сравнивать руки на голове по горизонтали и вертикали 
    Good_distance_for_hands_on_head_X = distance(px[8], px[7]) * 8/15 
    Good_distance_for_hands_on_head_Y = distance(py[6], py[9]) * 2
    #Распознаем руки на голове (0 - правая , 1 - левая)
    hand_on_head[0] = 1 if (distance(px[8], px[16]) < Good_distance_for_hands_on_head_X and distance(py[8], py[16]) < Good_distance_for_hands_on_head_Y) else 0
    hand_on_head[1] = 1 if (distance(px[7], px[15]) < Good_distance_for_hands_on_head_X and distance(py[7], py[15]) < Good_distance_for_hands_on_head_Y) else 0
    #если обе руки на голове, выводим сообщение
    if (hand_on_head[0]) and (hand_on_head[1]):
        return 4
                
Этап 5. (стр 48-62)
На данном этапе реализованно распознавание позы, в которой человек держит на груди перекрещенные руки.
По тому же принципу нам необходимо получить несколько допустимых расстояний, с которым мы будем сравнивать расстояние между точкой кисти и точкой противоположного плеча для каждой руки по обеим осям.
И Заполняем индикаторы, находятся ли руки на противоположных плечах. 
Далее необходимо проверить, находятся ли локти снизу, иначе если поместить кисти на противоположные плечи и задрать руки вверх, то программа выведет ошибочный результат.
Получаем необходимое расстояние, с которым будем сравнивать расстояние между каждой рукой и соответствующим ей бедром.
Заполняем индикаторы. Если обе руки находятся на противоположных плечах и оба локтя находятся на допустимом расстоянии к соответствующим им бедрам - возвращам состояние распознанной позы.

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
        return 5
    return 0 
Если никакую позу не удалось распознать - возвращаем 0. Ноль - состояние, в котором поза человека неизвестна. 
                
