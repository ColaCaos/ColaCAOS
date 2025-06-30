# Autor: Rubén Torre Merino
import cv2
import numpy as np
import time
import csv

def find_centroid(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None
    c = max(cnts, key=cv2.contourArea)
    M = cv2.moments(c)
    if M['m00'] == 0:
        return None, None
    return int(M['m10']/M['m00']), int(M['m01']/M['m00'])

def main():
    L1_cm, L2_cm = 23.5, 20.0

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 60)
    gpu_frame = cv2.cuda_GpuMat()

    # sliders de calibración
    cv2.namedWindow("Ajustes", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("R_LO","Ajustes",  0,180, lambda x:None)
    cv2.createTrackbar("R_HI","Ajustes", 10,180, lambda x:None)
    cv2.createTrackbar("R2_LO","Ajustes",160,180,lambda x:None)
    cv2.createTrackbar("R2_HI","Ajustes",180,180,lambda x:None)
    cv2.createTrackbar("G_LO","Ajustes", 34,180, lambda x:None)
    cv2.createTrackbar("G_HI","Ajustes", 62,180, lambda x:None)
    cv2.createTrackbar("B_LO","Ajustes",100,180, lambda x:None)
    cv2.createTrackbar("B_HI","Ajustes",140,180, lambda x:None)
    cv2.createTrackbar("S_MIN","Ajustes", 88,255, lambda x:None)
    cv2.createTrackbar("V_MIN","Ajustes", 30,255, lambda x:None)

    # CSV
    csv_f = open('coordenadas_cm.csv','w', newline='')
    writer = csv.writer(csv_f)
    writer.writerow([
        't_s',
        'red_x_cm','red_y_cm',
        'green_x_cm','green_y_cm',
        'blue_x_cm','blue_y_cm'
    ])

    # ventanas de máscara
    cv2.namedWindow("Mask Red",   cv2.WINDOW_NORMAL)
    cv2.namedWindow("Mask Green", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Mask Blue",  cv2.WINDOW_NORMAL)

    def detect():
        ret, frame = cap.read()
        if not ret:
            return None, None, None, None
        gpu_frame.upload(frame)
        hsv = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2HSV)
        s_min = cv2.getTrackbarPos("S_MIN","Ajustes")
        v_min = cv2.getTrackbarPos("V_MIN","Ajustes")
        masks = {}
        for color in ('red','green','blue'):
            if color == 'red':
                lo1 = (cv2.getTrackbarPos("R_LO","Ajustes"), s_min, v_min)
                hi1 = (cv2.getTrackbarPos("R_HI","Ajustes"),255,255)
                lo2 = (cv2.getTrackbarPos("R2_LO","Ajustes"), s_min, v_min)
                hi2 = (cv2.getTrackbarPos("R2_HI","Ajustes"),255,255)
                m1 = cv2.cuda.inRange(hsv, lo1, hi1)
                m2 = cv2.cuda.inRange(hsv, lo2, hi2)
                acc = cv2.cuda.bitwise_or(m1, m2)
            else:
                tag = color[0].upper()
                lo = (cv2.getTrackbarPos(f"{tag}_LO","Ajustes"), s_min, v_min)
                hi = (cv2.getTrackbarPos(f"{tag}_HI","Ajustes"),255,255)
                acc = cv2.cuda.inRange(hsv, lo, hi)
            morph = cv2.cuda.createMorphologyFilter(
                cv2.MORPH_OPEN, cv2.CV_8UC1,
                cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
            )
            masks[color] = morph.apply(acc).download()
        cr = find_centroid(masks['red'])
        cg = find_centroid(masks['green'])
        cb = find_centroid(masks['blue'])
        cv2.imshow("Mask Red",   masks['red'])
        cv2.imshow("Mask Green", masks['green'])
        cv2.imshow("Mask Blue",  masks['blue'])
        return cr, cg, cb, frame

    # calibración 2 segundos
    d1s, d2s = [], []
    t0 = time.time()
    while time.time() - t0 < 2.0:
        cr, cg, cb, _ = detect()
        if None not in (*cr,*cg,*cb):
            d1 = np.hypot(cg[0]-cb[0], cg[1]-cb[1])
            d2 = np.hypot(cr[0]-cg[0], cr[1]-cg[1])
            if d1>5 and d2>5:
                d1s.append(d1)
                d2s.append(d2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            csv_f.close()
            return
    if not d1s or not d2s:
        print("Falló calibración")
        cap.release()
        csv_f.close()
        return
    s1 = L1_cm / np.mean(d1s)
    s2 = L2_cm / np.mean(d2s)
    print(f"Calibrado: s1={s1:.3f} cm/px, s2={s2:.3f} cm/px")

    origin = None
    # Umbrales de salto en cm para cada segmento
    MAX_JUMP_GREEN = 10.0   # cm
    MAX_JUMP_RED   = 20.0   # cm

    prev = {'red':None,'green':None,'blue':None}
    t_start = time.time()

    while True:
        cr, cg, cb, frame = detect()
        t = time.time() - t_start
        if None in (*cr,*cg,*cb):
            cv2.putText(frame, "Tracking lost", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        else:
            # desplazamientos en píxeles
            dx1, dy1 = cg[0]-cb[0], cb[1]-cg[1]
            dx2, dy2 = cr[0]-cg[0], cg[1]-cr[1]

            # convertir a cm
            blue_cm  = (0.0, 0.0)
            green_cm = (dx1 * s1, dy1 * s1)
            red_cm   = (green_cm[0] + dx2 * s2,
                        green_cm[1] + dy2 * s2)

            # inicializar origen
            if origin is None:
                origin = blue_cm

            # coordenadas relativas al origen
            rel = {
                'red':   (red_cm[0] - origin[0],   red_cm[1] - origin[1]),
                'green': (green_cm[0] - origin[0], green_cm[1] - origin[1]),
                'blue':  (0.0,                     0.0)
            }

            # filtrar saltos por color usando umbrales distintos
            # verde
            cur_g = rel['green']
            p_g   = prev['green']
            if p_g is None or np.hypot(cur_g[0]-p_g[0], cur_g[1]-p_g[1]) < MAX_JUMP_GREEN:
                prev['green'] = cur_g
            else:
                cur_g = p_g
            # rojo
            cur_r = rel['red']
            p_r   = prev['red']
            if p_r is None or np.hypot(cur_r[0]-p_r[0], cur_r[1]-p_r[1]) < MAX_JUMP_RED:
                prev['red'] = cur_r
            else:
                cur_r = p_r

            # preparar valores para CSV
            blx, bly   = rel['blue']
            grx, gry   = cur_g
            redx, redy = cur_r

            writer.writerow([
                f"{t:.4f}",
                f"{redx:.3f}", f"{redy:.3f}",
                f"{grx:.3f}", f"{gry:.3f}",
                f"{blx:.3f}", f"{bly:.3f}"
            ])

            # dibujar pivotes
            for pt, col in zip((cr,cg,cb), ((0,0,255),(0,255,0),(255,0,0))):
                cv2.circle(frame, pt, 6, col, -1)

        cv2.imshow("Tracking + Trayectorias", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    csv_f.close()

if __name__ == "__main__":
    main()
